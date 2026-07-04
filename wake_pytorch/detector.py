"""Runtime wake-word detector — replaces openWakeWord in the Atlas app.

Loads the exported MatchboxNet (models/atlas.onnx) with onnxruntime and scores a
rolling 1.5 s mic buffer using the SAME MFCC front-end the model was trained on
(features.py). Exposes an openWakeWord-compatible surface so the app's existing
gating logic (wait_for_wake_word / _bargein_watcher) needs only trivial edits:

    det = WakeDetector("models/atlas.onnx")
    score = det.predict(chunk_int16)   # float in [0, 1]  (P("Atlas"))
    det.reset()

`predict` accepts any-length int16 mono chunk; the detector keeps its own rolling
window internally, so callers can feed it 80 ms (1280-sample) chunks as before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

# Support running both as a package module and standalone (sys.path shim).
try:
    from .features import SAMPLE_RATE, WINDOW_SAMPLES, mfcc
except ImportError:  # imported as a top-level module
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import SAMPLE_RATE, WINDOW_SAMPLES, mfcc


class WakeDetector:
    def __init__(self, model_path: str, providers: list[str] | None = None):
        self.model_path = str(model_path)
        self.sess = ort.InferenceSession(
            self.model_path,
            providers=providers or ["CPUExecutionProvider"])
        self._in = self.sess.get_inputs()[0].name
        self._buf = np.zeros(WINDOW_SAMPLES, dtype=np.int16)

    def reset(self) -> None:
        """Clear the rolling audio window (call between listening sessions)."""
        self._buf[:] = 0

    def predict(self, chunk: np.ndarray) -> float:
        """Append an int16 mono chunk and return P("Atlas") over the last 1.5 s."""
        chunk = np.asarray(chunk).reshape(-1).astype(np.int16)
        if chunk.shape[0] >= WINDOW_SAMPLES:
            self._buf = chunk[-WINDOW_SAMPLES:].copy()
        else:
            self._buf = np.concatenate([self._buf[chunk.shape[0]:], chunk])
        feat = mfcc(self._buf).unsqueeze(0).numpy().astype(np.float32)  # (1,64,frames)
        prob = self.sess.run(None, {self._in: feat})[0]
        return float(np.asarray(prob).reshape(-1)[0])


if __name__ == "__main__":
    # Smoke test against the exported model, if present.
    p = Path(__file__).resolve().parent.parent / "models" / "atlas.onnx"
    if not p.exists():
        raise SystemExit(f"No model at {p}; run export_onnx.py first.")
    det = WakeDetector(str(p))
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(1280) * 500).astype(np.int16)
    print(f"noise chunk score: {det.predict(noise):.4f}  (should be low)")
    det.reset()
    print("reset OK")
