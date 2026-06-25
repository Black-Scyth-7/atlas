"""Stage 14: local screen vision via a small VLM (Qwen2.5-VL-3B in llama.cpp).

Lets Atlas look at the screen and answer questions about it. The model is
lazy-loaded (only on the first screen question) and runs on CPU by default, so
it coexists with the GPU-resident Qwen3 on an 8 GB card.

Best-effort: if the model isn't present or fails to load, `look()` returns a
clear "unavailable" message and the rest of Atlas keeps working.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re

from config import VisionConfig


def capture_camera(index: int = 0, warmup_s: float = 0.6):
    """Grab a single frame from a webcam as a PIL RGB image.

    Reads frames for a short warmup period first: webcams return dark frames
    right after opening because auto-exposure needs ~half a second of wall-clock
    time (not just a few frames) to settle. Raises RuntimeError if no camera or
    frame is available.
    """
    import time

    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python isn't installed (pip install opencv-python).")
    from PIL import Image

    # CAP_DSHOW is the most reliable backend on Windows.
    backend = getattr(cv2, "CAP_DSHOW", 0)
    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"couldn't open camera {index}.")
        frame = None
        reads = 0
        deadline = time.time() + max(0.0, warmup_s)
        # Keep grabbing until the warmup time elapses AND we've read a few frames,
        # so the sensor has time to adjust exposure before we keep the last frame.
        while time.time() < deadline or reads < 3:
            ok, f = cap.read()
            if ok:
                frame, reads = f, reads + 1
            elif reads == 0 and time.time() >= deadline:
                break
        if frame is None:
            raise RuntimeError("camera opened but returned no frame.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def _parse_box(text: str):
    """Extract [x1, y1, x2, y2] from a Qwen2.5-VL grounding reply, or None."""
    # Preferred: a JSON object with "bbox_2d".
    m = re.search(r'\{[^{}]*"bbox_2d"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            box = json.loads(m.group(0)).get("bbox_2d")
            if isinstance(box, list) and len(box) >= 4:
                return [float(v) for v in box[:4]]
        except Exception:
            pass
    # Fallback: a bare [x1, y1, x2, y2] list.
    m = re.search(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
        r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if m:
        return [float(g) for g in m.groups()]
    # Last resort: the first four numbers anywhere (handles "(x1,y1),(x2,y2)").
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 4:
        return [float(n) for n in nums[:4]]
    return None


class Vision:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self._llm = None
        if not cfg.enable_vision:
            self.available, self.reason = False, "disabled in config"
        elif not (os.path.exists(cfg.model_path) and os.path.exists(cfg.mmproj_path)):
            self.available, self.reason = False, "vision model not downloaded"
        else:
            self.available, self.reason = True, ""

    def _load(self):
        """Load the VLM + vision projector on first use (CPU)."""
        if self._llm is None:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            handler = Qwen25VLChatHandler(
                clip_model_path=self.cfg.mmproj_path, verbose=False)
            self._llm = Llama(
                model_path=self.cfg.model_path, chat_handler=handler,
                n_ctx=self.cfg.n_ctx, n_gpu_layers=self.cfg.n_gpu_layers,
                verbose=False)
        return self._llm

    def _prep(self, image, max_px: int = 0) -> tuple[str, int, int]:
        """A PIL image or path -> (downscaled PNG data URI, width, height).

        The width/height are the dimensions of the image actually sent to the
        model — coordinates it returns are in this space, so callers divide by
        them to get scale-invariant fractions. `max_px` overrides the longest-
        side cap (0 = use max_image_px).
        """
        from PIL import Image

        cap = max_px or self.cfg.max_image_px
        img = Image.open(image) if isinstance(image, str) else image
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > cap:
            scale = cap / longest
            img = img.resize((max(1, int(img.size[0] * scale)),
                              max(1, int(img.size[1] * scale))))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return uri, img.size[0], img.size[1]

    def look(self, image, question: str) -> str:
        """Answer a question about an image (PIL image or file path)."""
        if not self.available:
            return f"Screen vision is unavailable ({self.reason})."
        try:
            uri, _, _ = self._prep(image)
            resp = self._load().create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": question},
                    ],
                }],
                max_tokens=self.cfg.max_tokens,
                temperature=0.2,
            )
            return (resp["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            return f"Vision failed: {e}"

    def locate(self, image, description: str):
        """Find a UI element and return its location as fractions of the image.

        Returns {"x", "y"} (the element's center, each 0..1), plus the raw
        "box" [x1,y1,x2,y2] in sent-image pixels — or None if not found.
        Qwen2.5-VL is a grounding model: it outputs a bounding box for a
        described object. We divide by the sent image's dimensions so the result
        is scale-independent and maps straight onto the real screen.
        """
        if not self.available:
            return None
        try:
            uri, w, h = self._prep(image, max_px=self.cfg.ground_image_px)
            prompt = (
                f'Locate the UI element described as: "{description}". Respond '
                "with ONLY a JSON object giving its bounding box in pixel "
                'coordinates of THIS image: {"bbox_2d": [x1, y1, x2, y2]}. '
                'If it is not visible, respond {"bbox_2d": null}.'
            )
            resp = self._load().create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=128,
                temperature=0.0,
            )
            text = resp["choices"][0]["message"].get("content") or ""
        except Exception:
            return None
        box = _parse_box(text)
        if not box:
            return None
        x1, y1, x2, y2 = box
        cx = min(max((x1 + x2) / 2.0 / w, 0.0), 1.0)
        cy = min(max((y1 + y2) / 2.0 / h, 0.0), 1.0)
        return {"x": cx, "y": cy, "box": box, "sent": (w, h)}


if __name__ == "__main__":
    # Standalone test: describe the screen, then locate an element on it.
    import sys

    from PIL import ImageGrab

    v = Vision(VisionConfig())
    print("available:", v.available, "|", v.reason or "ok")
    if v.available:
        print("loading VLM (first run)...")
        shot = ImageGrab.grab()
        print(v.look(shot, "Describe what is on the screen, concisely."))
        # Pass a description to test grounding, e.g.:
        #   python vision.py "the close button"
        target = sys.argv[1] if len(sys.argv) > 1 else "the Start button"
        loc = v.locate(shot, target)
        if loc:
            px, py = round(loc["x"] * shot.size[0]), round(loc["y"] * shot.size[1])
            print(f"located {target!r} at screen ~({px}, {py})  box={loc['box']}")
        else:
            print(f"could not locate {target!r}")
