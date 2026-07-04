"""Synthesize "Atlas" positive clips with Kokoro TTS (Kokoro-82M).

Loops over Kokoro's English voices x prosody variants x speeds, resamples each
24 kHz utterance to 16 kHz, trims silence, peak-normalizes, and centers it in a
1.5 s window saved as 16-bit PCM WAV in data/positive/.

    wake_pytorch/.venv/Scripts/python wake_pytorch/gen_positives.py            # all
    wake_pytorch/.venv/Scripts/python wake_pytorch/gen_positives.py --limit 6  # quick test

Diversity beyond this base set comes from on-the-fly augmentation at train time
(MUSAN noise mixing + SpecAugment + time-shift), not from more raw TTS clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from features import SAMPLE_RATE, WINDOW_SAMPLES, fit_window

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "positive"
KOKORO_SR = 24000

# Kokoro English voices. lang_code 'a' = American, 'b' = British; the pipeline's
# lang_code must match the voice's a*/b* prefix.
VOICES_A = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck",
]
VOICES_B = ["bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]

TEXTS = ["Atlas", "Atlas.", "Atlas?", "Atlas!"]
SPEEDS = [0.85, 0.95, 1.0, 1.1, 1.2]


def _trim_silence(x: np.ndarray, thresh_frac: float = 0.02,
                  margin: int = 800) -> np.ndarray:
    """Drop leading/trailing near-silence (energy below thresh_frac of peak)."""
    peak = float(np.max(np.abs(x))) or 1.0
    loud = np.where(np.abs(x) > thresh_frac * peak)[0]
    if loud.size == 0:
        return x
    lo = max(0, loud[0] - margin)
    hi = min(x.shape[0], loud[-1] + margin)
    return x[lo:hi]


def _postprocess(audio: np.ndarray) -> np.ndarray:
    """24 kHz float -> 16 kHz, trimmed, peak-normalized, centered in 1.5 s -> int16."""
    wav = torch.from_numpy(np.ascontiguousarray(audio)).float()
    wav = AF.resample(wav, KOKORO_SR, SAMPLE_RATE).numpy()
    wav = _trim_silence(wav)
    peak = float(np.max(np.abs(wav))) or 1.0
    wav = 0.95 * wav / peak                       # peak-normalize
    wav = fit_window(wav, WINDOW_SAMPLES)         # center in 1.5 s
    return np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)


def main(limit: int | None = None) -> None:
    from kokoro import KPipeline

    OUT.mkdir(parents=True, exist_ok=True)
    pipelines = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}
    voices = [("a", v) for v in VOICES_A] + [("b", v) for v in VOICES_B]

    made = 0
    for lang, voice in voices:
        pipe = pipelines[lang]
        for text in TEXTS:
            for speed in SPEEDS:
                if limit is not None and made >= limit:
                    print(f"\nReached --limit {limit}; stopping.")
                    print(f"Done: {made} positive clips in {OUT}")
                    return
                # Kokoro yields (graphemes, phonemes, audio) per chunk; a single
                # word is one chunk. Concatenate defensively.
                chunks = [a for _, _, a in pipe(text, voice=voice, speed=speed)]
                if not chunks:
                    continue
                audio = np.concatenate([
                    (c.detach().cpu().numpy() if isinstance(c, torch.Tensor)
                     else np.asarray(c)).astype(np.float32) for c in chunks])
                pcm = _postprocess(audio)
                tag = text.strip(".?!").lower() + \
                    ("_q" if "?" in text else "_e" if "!" in text
                     else "_p" if "." in text else "")
                fn = f"{voice}_{tag}_s{int(speed * 100)}.wav"
                sf.write(str(OUT / fn), pcm, SAMPLE_RATE)
                made += 1
                if made % 25 == 0:
                    print(f"  {made} clips...")

    print(f"Done: {made} positive clips in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate at most N clips (quick smoke test).")
    args = ap.parse_args()
    main(limit=args.limit)
