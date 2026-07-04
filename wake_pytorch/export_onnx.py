"""Export the trained MatchboxNet to a self-contained ONNX for runtime inference.

Wraps the net in a softmax head so onnxruntime returns P("Atlas") directly, then
verifies PyTorch vs onnxruntime parity and copies the model to models/atlas.onnx
where the app's WakeDetector loads it.

    wake_pytorch/.venv/Scripts/python wake_pytorch/export_onnx.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# torch's ONNX exporter prints a '✅' progress message; the default Windows
# console codepage (cp1252) can't encode it and crashes the process. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    _rc = getattr(_stream, "reconfigure", None)
    if _rc is not None:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from features import N_MFCC, N_FRAMES
from model import build

BASE = Path(__file__).resolve().parent
CKPT = BASE / "checkpoints" / "atlas_matchboxnet.pt"
ONNX_OUT = BASE / "atlas.onnx"
APP_MODEL = BASE.parent / "models" / "atlas.onnx"


class ProbWrapper(nn.Module):
    """MatchboxNet + softmax -> P(atlas), output shape (batch, 1)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)[:, 1:2]


def main() -> None:
    if not CKPT.exists():
        raise SystemExit(f"No checkpoint at {CKPT}; run train.py first.")
    ckpt = torch.load(CKPT, map_location="cpu")
    net = build(num_classes=2)
    net.load_state_dict(ckpt["state_dict"])
    model = ProbWrapper(net).eval()
    print(f"loaded checkpoint (val={ckpt.get('val')})")

    dummy = torch.randn(1, N_MFCC, N_FRAMES)
    torch.onnx.export(
        model, dummy, str(ONNX_OUT),
        input_names=["mfcc"], output_names=["prob"],
        dynamic_axes={"mfcc": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=18,
    )
    # Fold any external-data sidecar into a single self-contained file.
    onnx.save_model(onnx.load(str(ONNX_OUT)), str(ONNX_OUT),
                    save_as_external_data=False)
    print(f"exported -> {ONNX_OUT}")

    # Parity check: PyTorch vs onnxruntime on random inputs.
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    max_err = 0.0
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(1, N_MFCC, N_FRAMES)
            pt = float(model(x).reshape(-1)[0])
            on = float(np.asarray(sess.run(None, {"mfcc": x.numpy()})[0]).reshape(-1)[0])
            max_err = max(max_err, abs(pt - on))
    print(f"torch vs onnxruntime max|Δ|={max_err:.2e}")
    if max_err > 1e-4:
        raise SystemExit("ABORT: ONNX parity check failed.")

    APP_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ONNX_OUT, APP_MODEL)
    print(f"copied -> {APP_MODEL} (app runtime model)")


if __name__ == "__main__":
    main()
