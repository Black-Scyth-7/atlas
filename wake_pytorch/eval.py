"""Evaluate the trained model: sweep the detection threshold, report recall vs
false-wakes per hour — the metric that actually matters for a wake word.

    wake_pytorch/.venv/Scripts/python wake_pytorch/eval.py
    ... wake_pytorch/eval.py --neg-windows 8000 --target-fp 0.5

Scores every positive clip and a large set of non-overlapping 1.5 s negative
windows (MUSAN speech/noise/music + quiet), then prints, for each threshold, the
true-accept rate and estimated false-wakes/hour, and recommends a threshold.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset import _read_window, POS_DIR, REAL_DIR, REAL_NEG_DIR, MANIFEST
from features import WINDOW_SAMPLES, mfcc
from model import build

BASE = Path(__file__).resolve().parent
CKPT = BASE / "checkpoints" / "atlas_matchboxnet.pt"
WINDOW_HOURS = WINDOW_SAMPLES / 16000 / 3600.0


@torch.no_grad()
def _scores(model, wavs, device, batch=256) -> np.ndarray:
    out = []
    buf = []
    for w in wavs:
        buf.append(mfcc(w))
        if len(buf) == batch:
            x = torch.stack(buf).to(device)
            out.append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
            buf = []
    if buf:
        x = torch.stack(buf).to(device)
        out.append(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-windows", type=int, default=8000)
    ap.add_argument("--target-fp", type=float, default=0.5, help="max false-wakes/hour")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not CKPT.exists():
        raise SystemExit(f"No checkpoint at {CKPT}; run train.py first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device)
    model = build(num_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rng = random.Random(args.seed)
    # Positives: TTS + your own recordings (so recall reflects your voice).
    pos = sorted(str(p) for p in POS_DIR.glob("*.wav")) + \
        sorted(str(p) for p in REAL_DIR.glob("*.wav"))
    pos_wav = [_read_window(p, rng=rng) for p in pos]

    man = json.loads(MANIFEST.read_text())
    neg_pool = man.get("speech", []) + man.get("noise", []) + \
        man.get("music", []) + man.get("quiet", [])
    neg_wav = [_read_window(rng.choice(neg_pool), rng=rng)
               for _ in range(args.neg_windows)]

    # Your own NON-"Atlas" speech: score every window and report a SEPARATE
    # false-wake rate on it — the metric that matters for your day-to-day chatter.
    real_neg = sorted(str(p) for p in REAL_NEG_DIR.glob("*.wav"))
    rneg_wav = [_read_window(p, rng=rng) for p in real_neg]

    print(f"scoring {len(pos_wav)} positives, {len(neg_wav)} MUSAN/quiet windows "
          f"({len(neg_wav) * WINDOW_HOURS:.2f} h), "
          f"{len(rneg_wav)} of your own speech windows...")
    ps = _scores(model, pos_wav, device)
    ns = _scores(model, neg_wav, device)
    rns = _scores(model, rneg_wav, device) if rneg_wav else np.zeros(0)
    neg_hours = len(ns) * WINDOW_HOURS

    # 'your_fp' = fraction of YOUR non-Atlas speech windows that would false-wake.
    hdr = f"\n{'thr':>5} {'recall':>8} {'fp/hour':>9} {'neg_fires':>10}"
    if len(rns):
        hdr += f" {'your_fp%':>9}"
    print(hdr)
    best = None
    for thr in np.arange(0.10, 0.96, 0.05):
        recall = float(np.mean(ps >= thr))
        fires = int(np.sum(ns >= thr))
        fph = fires / (neg_hours + 1e-9)
        row = f"{thr:5.2f} {recall:8.3f} {fph:9.2f} {fires:10d}"
        if len(rns):
            row += f" {float(np.mean(rns >= thr)) * 100:8.2f}%"
        print(row)
        if fph <= args.target_fp and (best is None or recall > best[1]):
            best = (float(thr), recall, fph)

    if best:
        print(f"\nRecommended wake_threshold={best[0]:.2f} "
              f"(recall={best[1]:.3f}, {best[2]:.2f} false-wakes/hour "
              f"<= target {args.target_fp}).")
    else:
        print(f"\nNo threshold meets <= {args.target_fp} fp/hour; "
              "train longer / add negatives.")


if __name__ == "__main__":
    main()
