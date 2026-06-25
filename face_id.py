"""Local face recognition via InsightFace (ArcFace embeddings, ONNX, CPU).

The visual parallel to speaker_id.py: enroll known people by name, then identify
who is in front of the camera by cosine-matching 512-d face embeddings. The
detector + recognizer are InsightFace's `buffalo_l` ONNX models, run on CPU, and
lazy-loaded on first use so they don't slow startup.

Best-effort: if InsightFace (or its models) aren't available, `available` is
False and Atlas runs without face recognition.

NOTE: like the voice speaker gate, this is **personalization, not security** —
it can be fooled by a photo and must not gate access to anything sensitive.

Run standalone to identify whoever is at the webcam right now:
    python face_id.py
"""

from __future__ import annotations

import os

import numpy as np

from config import FaceConfig


class FaceRecognizer:
    def __init__(self, cfg: FaceConfig):
        self.cfg = cfg
        self._app = None
        self.db: dict[str, np.ndarray] = {}  # name -> (N, 512) embeddings
        if not cfg.enable_faces:
            self.available, self.reason = False, "disabled in config"
        else:
            try:
                import insightface  # noqa: F401
                self.available, self.reason = True, ""
            except Exception as e:
                self.available, self.reason = False, f"insightface unavailable ({e})"
        if self.available:
            self._load_db()

    # ---- model (lazy) ----------------------------------------------------
    def _app_lazy(self):
        if self._app is None:
            import warnings
            warnings.filterwarnings("ignore")
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self.cfg.model_pack,
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_thresh=self.cfg.min_det_score,
                        det_size=(self.cfg.det_size, self.cfg.det_size))
            self._app = app
        return self._app

    # ---- enrolled-face database -----------------------------------------
    def _load_db(self) -> None:
        path = self.cfg.db_path
        if not os.path.exists(path):
            return
        try:
            data = np.load(path, allow_pickle=True)
            names = list(data["names"])
            embs = data["embeddings"]
            self.db = {str(n): np.asarray(e, dtype=np.float32)
                       for n, e in zip(names, embs)}
        except Exception:
            self.db = {}

    def _save_db(self) -> None:
        names = list(self.db.keys())
        embs = np.empty(len(names), dtype=object)
        for i, n in enumerate(names):
            embs[i] = self.db[n]
        np.savez(self.cfg.db_path,
                 names=np.array(names, dtype=object), embeddings=embs)

    def names(self) -> list[str]:
        return list(self.db.keys())

    def forget(self, name: str) -> str:
        if name in self.db:
            del self.db[name]
            self._save_db()
            return f"Forgot {name}'s face."
        return f"I don't have a face saved for {name}."

    def reset(self) -> bool:
        """Forget every enrolled face (clear memory + delete the database file)."""
        self.db = {}
        try:
            if os.path.exists(self.cfg.db_path):
                os.remove(self.cfg.db_path)
            return True
        except Exception:
            return False

    # ---- detection / embedding ------------------------------------------
    @staticmethod
    def _to_bgr(image) -> np.ndarray:
        from PIL import Image

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        else:
            arr = np.asarray(image)
        return np.ascontiguousarray(arr)

    def _detect(self, image) -> list:
        if not self.available:
            return []
        faces = self._app_lazy().get(self._to_bgr(image))
        return [f for f in faces if f.det_score >= self.cfg.min_det_score]

    @staticmethod
    def _largest(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))

    # ---- enroll / identify ----------------------------------------------
    def enroll(self, image, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return "What name should I save this face under?"
        faces = self._detect(image)
        if not faces:
            return "I couldn't see a clear face to enroll."
        emb = np.asarray(self._largest(faces).normed_embedding,
                         dtype=np.float32).reshape(1, -1)
        self.db[name] = (np.vstack([self.db[name], emb])
                         if name in self.db else emb)
        self._save_db()
        n = len(self.db[name])
        return f"Saved {name}'s face ({n} sample{'s' if n != 1 else ''})."

    def _match(self, emb: np.ndarray) -> tuple[str, float]:
        best_name, best = None, -1.0
        for name, embs in self.db.items():
            score = float((embs @ emb).max())  # normed -> cosine similarity
            if score > best:
                best, best_name = score, name
        if best_name is None or best < self.cfg.match_threshold:
            return "unknown", best
        return best_name, best

    def identify(self, image) -> list[dict]:
        """Return [{name, score, bbox}] for each detected face (largest first)."""
        faces = sorted(self._detect(image),
                       key=lambda f: (f.bbox[2] - f.bbox[0])
                       * (f.bbox[3] - f.bbox[1]), reverse=True)
        out = []
        for f in faces:
            name, score = self._match(
                np.asarray(f.normed_embedding, dtype=np.float32))
            out.append({"name": name, "score": round(score, 3),
                        "bbox": [int(v) for v in f.bbox]})
        return out


if __name__ == "__main__":
    from vision import capture_camera

    fr = FaceRecognizer(FaceConfig())
    print("available:", fr.available, "|", fr.reason or "ok")
    print("enrolled:", fr.names() or "(none — run enroll_face.py <name>)")
    if fr.available:
        print("loading models + capturing (first run downloads ~300 MB)...")
        results = fr.identify(capture_camera(0))
        if not results:
            print("  (no face detected)")
        for r in results:
            print(f"  {r['name']}  (score {r['score']})  bbox={r['bbox']}")
