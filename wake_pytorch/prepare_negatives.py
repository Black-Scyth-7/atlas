"""Prepare negative + background audio: MUSAN download/index + synthetic quiet.

Two jobs:
  1. Ensure the MUSAN corpus (music / speech / noise) is present under
     data/musan/, downloading + extracting it (~11 GB) if missing, then write a
     manifest of its WAV paths. MUSAN speech = hard negatives (non-"Atlas"
     babble); music + noise are both negatives AND the background pool that
     dataset.py mixes into positives for augmentation.
  2. Synthesize quiet/ambient negative clips (silence, mic floor, colored noise,
     mains hum, tones) so the model learns an idle mic is NOT a wake. Pure numpy.

    wake_pytorch/.venv/Scripts/python wake_pytorch/prepare_negatives.py
    wake_pytorch/.venv/Scripts/python wake_pytorch/prepare_negatives.py --skip-download

MUSAN: openslr.org/17 (Snyder, Chen, Povey 2015).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from features import SAMPLE_RATE, WINDOW_SAMPLES

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MUSAN_DIR = DATA / "musan"
QUIET_DIR = DATA / "negative_quiet"
MANIFEST = DATA / "negatives_manifest.json"

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_TAR = DATA / "musan.tar.gz"

N_QUIET = 3000    # synthetic quiet/ambient clips


# --- MUSAN download / extract / index ------------------------------------------

def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  {dest.name} already downloaded ({dest.stat().st_size/1e9:.1f} GB).")
        return
    print(f"  downloading {url} -> {dest} (~11 GB, slow)...")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def ensure_musan(skip_download: bool) -> None:
    if (MUSAN_DIR / "speech").is_dir():
        print("MUSAN already extracted.")
        return
    if skip_download:
        raise SystemExit(f"MUSAN not found at {MUSAN_DIR} and --skip-download set.")
    DATA.mkdir(parents=True, exist_ok=True)
    _download(MUSAN_URL, MUSAN_TAR)
    print("  extracting (this takes a while)...")
    with tarfile.open(MUSAN_TAR) as t:
        t.extractall(DATA)                     # creates data/musan/{music,speech,noise}
    print(f"  extracted -> {MUSAN_DIR}")


def index_musan() -> dict[str, list[str]]:
    manifest: dict[str, list[str]] = {}
    for cat in ("music", "speech", "noise"):
        d = MUSAN_DIR / cat
        wavs = sorted(str(p) for p in d.rglob("*.wav")) if d.is_dir() else []
        manifest[cat] = wavs
        print(f"  MUSAN {cat}: {len(wavs)} files")
    return manifest


# --- Synthetic quiet/ambient negatives (numpy only) ----------------------------

def _colored_noise(n: int, rng: np.random.Generator, beta: float) -> np.ndarray:
    """1/f**beta noise via FFT shaping (beta=1 pink, 2 brown), unit std."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    freqs[0] = freqs[1]
    spec = spec / (freqs ** (beta / 2.0))
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-9)


def _synth_quiet_clip(rng: np.random.Generator) -> np.ndarray:
    """One 1.5 s int16 clip of quiet/ambient non-speech that must NOT read as 'atlas'."""
    n = WINDOW_SAMPLES
    t = np.arange(n) / SAMPLE_RATE
    kind = rng.integers(0, 6)
    if kind == 0:                                  # near-silence / mic floor
        x = rng.normal(0, rng.uniform(1, 25), n)
    elif kind == 1:                                # white noise, quiet..moderate
        x = rng.normal(0, rng.uniform(25, 500), n)
    elif kind == 2:                                # pink noise (fans / AC / hiss)
        x = _colored_noise(n, rng, 1.0) * rng.uniform(60, 700)
    elif kind == 3:                                # brown noise (deep rumble)
        x = _colored_noise(n, rng, 2.0) * rng.uniform(60, 700)
    elif kind == 4:                                # mains hum 50/60 Hz + harmonics
        f0 = float(rng.choice([50.0, 60.0]))
        x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.283))
                for k in (1, 2, 3, 4))
        x = x * rng.uniform(40, 400) + rng.normal(0, 15, n)
    else:                                          # steady tone/whine + hiss
        f = rng.uniform(180, 5000)
        x = (np.sin(2 * np.pi * f * t) * rng.uniform(25, 250)
             + rng.normal(0, rng.uniform(5, 60), n))
    return np.clip(x, -32768, 32767).astype(np.int16)


def make_quiet_negatives(n: int = N_QUIET) -> None:
    QUIET_DIR.mkdir(parents=True, exist_ok=True)
    if any(QUIET_DIR.glob("*.wav")):
        print(f"Quiet negatives already present in {QUIET_DIR}.")
        return
    rng = np.random.default_rng(0)
    for i in tqdm(range(n), desc="  quiet negatives"):
        sf.write(str(QUIET_DIR / f"quiet_{i:05d}.wav"),
                 _synth_quiet_clip(rng), SAMPLE_RATE)
    print(f"  {n} quiet clips -> {QUIET_DIR}")


def main(skip_download: bool) -> None:
    print("1) MUSAN")
    ensure_musan(skip_download)
    manifest = index_musan()

    print("2) Synthetic quiet/ambient negatives")
    make_quiet_negatives()
    manifest["quiet"] = sorted(str(p) for p in QUIET_DIR.glob("*.wav"))

    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest -> {MANIFEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true",
                    help="Fail instead of downloading if MUSAN is missing.")
    args = ap.parse_args()
    main(skip_download=args.skip_download)
