"""Train MatchboxNet on the Kokoro positives vs MUSAN/quiet negatives.

    wake_pytorch/.venv/Scripts/python wake_pytorch/train.py
    ... wake_pytorch/train.py --epochs 30 --epoch-size 20000 --batch 256

Loader workers return only raw int16 waveforms; MFCC (via the shared features.py
transform, placed on the GPU) and SpecAugment run batched on-device in the loop.
That keeps memory flat (no CUDA torch copy per worker — the earlier OOM) and is
faster than per-item CPU features. Saves the best checkpoint (by validation
score) to checkpoints/atlas_matchboxnet.pt.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import features
from dataset import WakeDataset, POS_DIR, REAL_DIR, REAL_NEG_DIR
from model import build

BASE = Path(__file__).resolve().parent
CKPT_DIR = BASE / "checkpoints"
CKPT = CKPT_DIR / "atlas_matchboxnet.pt"


def _split(files: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle + split a file list into (train, val)."""
    files = list(files)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_frac)) if files else 0
    return files[n_val:], files[:n_val]

def _split_positives(val_frac: float, seed: int):
    """Split TTS and (optional) real positives into disjoint train/val sets."""
    tts = sorted(str(p) for p in POS_DIR.glob("*.wav"))
    if not tts:
        raise SystemExit(f"No positives in {POS_DIR}; run gen_positives.py first.")
    real = sorted(str(p) for p in REAL_DIR.glob("*.wav"))
    real_neg = sorted(str(p) for p in REAL_NEG_DIR.glob("*.wav"))
    tts_tr, tts_va = _split(tts, val_frac, seed)
    real_tr, real_va = _split(real, val_frac, seed + 7)
    rneg_tr, rneg_va = _split(real_neg, val_frac, seed + 13)
    return (tts_tr, tts_va), (real_tr, real_va), (rneg_tr, rneg_va)


def compute_feats(wav_i16: torch.Tensor, transform, device) -> torch.Tensor:
    """(B, T) int16 waveform tensor -> (B, n_mfcc, frames) normalized MFCC on device.

    Uses the SAME transform/normalization as features.mfcc (and thus the runtime
    detector), so training and inference see identical features.
    """
    x = wav_i16.to(device, non_blocking=True).float() / 32768.0
    feat = transform(x)                          # (B, n_mfcc, frames)
    return features.normalize(feat)


def spec_augment(feat: torch.Tensor, n_freq: int = 2, n_time: int = 2,
                 max_f: int = 12, max_t: int = 25) -> torch.Tensor:
    """SpecAugment: zero random frequency + time bands (batched, on-device)."""
    _, n_mels, n_frames = feat.shape
    for _ in range(n_freq):
        f = random.randint(0, max_f)
        f0 = random.randint(0, max(0, n_mels - f))
        feat[:, f0:f0 + f, :] = 0.0
    for _ in range(n_time):
        t = random.randint(0, max_t)
        t0 = random.randint(0, max(0, n_frames - t))
        feat[:, :, t0:t0 + t] = 0.0
    return feat


@torch.no_grad()
def evaluate(model, loader, transform, device) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    for wav, label in loader:
        feat = compute_feats(wav, transform, device)
        label = label.to(device)
        pred = model(feat).argmax(dim=1)
        tp += int(((pred == 1) & (label == 1)).sum())
        fp += int(((pred == 1) & (label == 0)).sum())
        tn += int(((pred == 0) & (label == 0)).sum())
        fn += int(((pred == 0) & (label == 1)).sum())
    n = tp + fp + tn + fn
    recall = tp / (tp + fn + 1e-9)               # true-accept rate on "Atlas"
    fpr = fp / (fp + tn + 1e-9)                  # false-wake rate on negatives
    acc = (tp + tn) / (n + 1e-9)
    return {"acc": acc, "recall": recall, "fpr": fpr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, default=20000)
    ap.add_argument("--val-size", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0,
                    help="loader workers (0 = in-process; keep low — each worker "
                         "imports torch). Data is light (waveforms), so 0-2 is plenty.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--real-frac", type=float, default=0.5,
                    help="fraction of positive samples drawn from your own recordings "
                         "(if present) — heavy oversampling so your voice isn't drowned "
                         "out by the ~540 TTS clips.")
    ap.add_argument("--real-neg-frac", type=float, default=0.35,
                    help="fraction of NEGATIVE samples drawn from your own non-'Atlas' "
                         "speech (if present) — hard same-speaker negatives.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    (tts_tr, tts_va), (real_tr, real_va), (rneg_tr, rneg_va) = \
        _split_positives(args.val_frac, args.seed)
    rf = args.real_frac if real_tr else 0.0
    rnf = args.real_neg_frac if rneg_tr else 0.0
    print(f"positives(tts): {len(tts_tr)} train / {len(tts_va)} val  |  "
          f"real(voice): {len(real_tr)} train / {len(real_va)} val  "
          f"(real_frac={rf})", flush=True)
    print(f"real NEG(your speech): {len(rneg_tr)} train / {len(rneg_va)} val  "
          f"(real_neg_frac={rnf})", flush=True)
    if not real_tr:
        print("  note: no real-voice clips — run record_positives.py to add your voice.",
              flush=True)
    if not rneg_tr:
        print("  note: no real-negative clips — run record_negatives.py to add hard "
              "same-speaker negatives.", flush=True)

    train_ds = WakeDataset(epoch_size=args.epoch_size, train=True,
                           positives=tts_tr, real_positives=real_tr,
                           real_negatives=rneg_tr, real_frac=rf,
                           real_neg_frac=rnf, seed=args.seed)
    # Validation draws real + TTS positives and real negatives the same way, so
    # val recall/fpr reflect your voice too (fracs clamp to 0 if a pool is empty).
    val_ds = WakeDataset(epoch_size=args.val_size, train=False,
                         positives=tts_va, real_positives=real_va,
                         real_negatives=rneg_va, real_frac=rf,
                         real_neg_frac=rnf, seed=args.seed + 1)
    print(f"negatives[speech/music/noise/quiet]="
          f"{len(train_ds.neg_speech)}/{len(train_ds.neg_music)}/"
          f"{len(train_ds.neg_noise)}/{len(train_ds.neg_quiet)}", flush=True)

    dl_kw = dict(batch_size=args.batch, num_workers=args.workers,
                 pin_memory=(device == "cuda"),
                 persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=False, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    transform = features.make_transform(device)
    model = build(num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = correct = seen = 0
        for wav, label in train_dl:
            feat = spec_augment(compute_feats(wav, transform, device))
            label = label.to(device)
            opt.zero_grad()
            logits = model(feat)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            running += loss.item() * feat.size(0)
            correct += int((logits.argmax(1) == label).sum())
            seen += feat.size(0)
        sched.step()
        m = evaluate(model, val_dl, transform, device)
        # Score favors a wake word: high recall with low false-wake rate.
        score = m["recall"] - 2.0 * m["fpr"]
        flag = ""
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(),
                        "val": m, "args": vars(args)}, CKPT)
            flag = "  <- best (saved)"
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={running/seen:.4f} train_acc={correct/seen:.3f}  "
              f"val_acc={m['acc']:.3f} recall={m['recall']:.3f} "
              f"fpr={m['fpr']:.4f}{flag}", flush=True)

    print(f"\nBest score={best_score:.4f}. Checkpoint -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
