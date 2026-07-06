"""Training dataset: balanced positives/negatives with on-the-fly augmentation.

Each item is a 1.5 s window turned into normalized MFCC (features.py) with a
label (1 = "Atlas", 0 = not). Diversity comes from augmentation applied here, not
from more raw clips:
  - positives: random time-shift + MUSAN background mixed at random SNR.
  - negatives: MUSAN speech (hard babble), music, noise (random crops) + the
    synthetic quiet clips.
  - both: SpecAugment (time + frequency masking) on the MFCC at train time.

An "epoch" is a fixed number of randomly-sampled, class-balanced examples.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from features import SAMPLE_RATE, WINDOW_SAMPLES

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
POS_DIR = DATA / "positive"            # Kokoro TTS positives
REAL_DIR = DATA / "positive_real"      # your own recorded "Atlas" (record_positives.py)
REAL_NEG_DIR = DATA / "negative_real"  # your own NON-"Atlas" speech (record_negatives.py)
NEG_WORDS_DIR = DATA / "negative_words"  # TTS isolated single-word negatives (gen_negatives_tts.py)
MANIFEST = DATA / "negatives_manifest.json"


def _read_window(path: str, length: int = WINDOW_SAMPLES,
                 rng: random.Random | None = None) -> np.ndarray:
    """Read a random `length`-sample mono crop (int16) from a WAV, resampling to
    16 kHz if needed and looping/padding short files."""
    info = sf.info(path)
    need_src = int(length * info.samplerate / SAMPLE_RATE) + 1
    total = info.frames
    if total <= need_src:
        data, sr = sf.read(path, dtype="int16", always_2d=True)
    else:
        start = (rng or random).randint(0, total - need_src)
        data, sr = sf.read(path, start=start, frames=need_src,
                           dtype="int16", always_2d=True)
    x = data[:, 0]
    if sr != SAMPLE_RATE:
        xf = torch.from_numpy(x.astype(np.float32) / 32768.0)
        x = (AF.resample(xf, sr, SAMPLE_RATE).numpy() * 32768.0).astype(np.int16)
    if x.shape[0] < length:                       # loop-pad short clips
        reps = length // x.shape[0] + 1
        x = np.tile(x, reps)
    return x[:length]


def _mix_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `noise` to `sig` (int16) at the given SNR; returns int16."""
    s = sig.astype(np.float32)
    n = noise.astype(np.float32)
    sp = float(np.mean(s ** 2)) + 1e-9
    npow = float(np.mean(n ** 2)) + 1e-9
    scale = np.sqrt(sp / (npow * (10 ** (snr_db / 10.0))))
    out = s + n * scale
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 32767:                              # avoid clipping
        out = out * (32767.0 / peak)
    return out.astype(np.int16)


class WakeDataset(Dataset):
    def __init__(self, epoch_size: int = 20000, train: bool = True,
                 pos_aug_prob: float = 0.7, snr_range=(0.0, 20.0),
                 positives: list[str] | None = None,
                 real_positives: list[str] | None = None,
                 real_negatives: list[str] | None = None,
                 neg_words: list[str] | None = None,
                 real_frac: float = 0.5, real_neg_frac: float = 0.35,
                 seed: int = 0):
        self.train = train
        self.epoch_size = epoch_size
        self.pos_aug_prob = pos_aug_prob
        self.snr_range = snr_range

        self.positives = (list(positives) if positives is not None
                          else sorted(str(p) for p in POS_DIR.glob("*.wav")))
        if not self.positives:
            raise SystemExit(f"No positives in {POS_DIR}; run gen_positives.py first.")
        # Your own recorded "Atlas" clips. Far fewer than the TTS pool, so when
        # present they're drawn a fixed `real_frac` of the time (not proportional
        # to file count) — heavy oversampling so your voice isn't drowned out.
        # Augmentation (time-shift + MUSAN mixing) supplies the missing variety.
        self.real_positives = (list(real_positives) if real_positives is not None
                               else sorted(str(p) for p in REAL_DIR.glob("*.wav")))
        self.real_frac = real_frac if self.real_positives else 0.0
        man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        self.neg_speech = man.get("speech", [])
        self.neg_music = man.get("music", [])
        self.neg_noise = man.get("noise", [])
        self.neg_quiet = man.get("quiet", [])
        self.bg_pool = self.neg_music + self.neg_noise    # augmentation backgrounds
        # ISOLATED single-word negatives (TTS, same format as positives). These
        # are the critical hard negatives: without them the model shortcuts to
        # "any short word in quiet = Atlas" and wakes on hello/hey/yellow/help.
        # Heavily weighted so that shortcut can't survive training.
        self.neg_words = (list(neg_words) if neg_words is not None
                          else sorted(str(p) for p in NEG_WORDS_DIR.glob("*.wav")))
        # Negative sources with sampling weights. Isolated words dominate when
        # present (they fix the false-wake-on-any-word failure); otherwise the
        # weights fall back to the hard MUSAN speech babble.
        if self.neg_words:
            self.neg_sources = [
                (self.neg_words, 0.45), (self.neg_speech, 0.25),
                (self.neg_noise, 0.12), (self.neg_music, 0.08),
                (self.neg_quiet, 0.10),
            ]
        else:
            self.neg_sources = [
                (self.neg_speech, 0.45), (self.neg_noise, 0.20),
                (self.neg_music, 0.15), (self.neg_quiet, 0.20),
            ]
        self.neg_sources = [(s, w) for s, w in self.neg_sources if s]
        if not self.neg_sources:
            raise SystemExit("No negatives; run prepare_negatives.py first.")
        # Your own NON-"Atlas" speech: the hardest negatives (same speaker as the
        # positives). When present, drawn a fixed `real_neg_frac` of the time so
        # your voice is well represented despite the huge MUSAN pool.
        self.real_negatives = (list(real_negatives) if real_negatives is not None
                               else sorted(str(p) for p in REAL_NEG_DIR.glob("*.wav")))
        self.real_neg_frac = real_neg_frac if self.real_negatives else 0.0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.epoch_size

    def _sample_negative(self, rng: random.Random) -> np.ndarray:
        if self.real_neg_frac and rng.random() < self.real_neg_frac:
            return _read_window(rng.choice(self.real_negatives), rng=rng)
        pools, weights = zip(*self.neg_sources)
        pool = rng.choices(pools, weights=weights, k=1)[0]
        return _read_window(rng.choice(pool), rng=rng)

    def _sample_positive(self, rng: random.Random) -> np.ndarray:
        pool = (self.real_positives if (self.real_frac and rng.random() < self.real_frac)
                else self.positives)
        x = _read_window(rng.choice(pool), rng=rng)
        if self.train:
            shift = rng.randint(-3200, 3200)          # +/- 200 ms time-shift
            x = np.roll(x, shift)
            if self.bg_pool and rng.random() < self.pos_aug_prob:
                bg = _read_window(rng.choice(self.bg_pool), rng=rng)
                snr = rng.uniform(*self.snr_range)
                x = _mix_snr(x, bg, snr)
        return x

    def __getitem__(self, idx: int):
        # Returns a raw int16 waveform + label; MFCC is computed on the GPU in
        # batches by train.py (keeps loader workers light — no torch/CUDA per
        # worker — and avoids the OOM from many feature-extracting processes).
        rng = random.Random((self._rng.randint(0, 1 << 30) ^ idx) + idx)
        label = idx % 2                               # exact 50/50 balance
        wav = self._sample_positive(rng) if label == 1 else self._sample_negative(rng)
        return (torch.from_numpy(np.ascontiguousarray(wav.astype(np.int16))),
                torch.tensor(label, dtype=torch.long))


if __name__ == "__main__":
    ds = WakeDataset(epoch_size=8, train=True)
    print(f"positives(tts)={len(ds.positives)} real={len(ds.real_positives)} "
          f"(real_frac={ds.real_frac}) "
          f"neg[speech/music/noise/quiet]="
          f"{len(ds.neg_speech)}/{len(ds.neg_music)}/{len(ds.neg_noise)}/{len(ds.neg_quiet)} "
          f"real_neg={len(ds.real_negatives)} (real_neg_frac={ds.real_neg_frac})")
    wav, lab = ds[1]
    print(f"item: wav {tuple(wav.shape)} (expected ({WINDOW_SAMPLES},)) "
          f"dtype={wav.dtype} label={int(lab)}")
    wav0, lab0 = ds[0]
    print(f"item0 label={int(lab0)} (expected 0)")
