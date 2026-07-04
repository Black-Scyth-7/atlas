"""The audio-feature contract shared by training AND runtime.

This is the ONE module both sides must agree on. `train.py`/`dataset.py` and the
runtime `detector.py` both import `mfcc()` from here so the model never sees a
different front-end at inference than it was trained on.

Front-end: 64 MFCC @ 16 kHz (matches the MatchboxNet paper), 25 ms window /
10 ms hop, per-feature (per-coefficient) mean/std normalization over the window.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

# --- Locked constants (see PLAN.md §1). Changing any of these invalidates a
# trained model; retrain if you touch them. ---
SAMPLE_RATE = 16000
N_FFT = 400          # 25 ms window
WIN_LENGTH = 400
HOP_LENGTH = 160     # 10 ms hop
N_MELS = 64
N_MFCC = 64
F_MIN = 20.0
F_MAX = 8000.0

WINDOW_SAMPLES = 24000   # 1.5 s detection window
# ~ (WINDOW_SAMPLES / HOP_LENGTH) + 1 = 151 frames
N_FRAMES = WINDOW_SAMPLES // HOP_LENGTH + 1

_EPS = 1e-5

# One shared transform instance (kept on CPU by default; callers may .to(device)
# a copy for batched GPU feature extraction during training).
_MFCC = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs=dict(
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        center=True,
        power=2.0,
    ),
)


def _to_float_tensor(wav) -> torch.Tensor:
    """Accept int16/float ndarray or tensor -> float32 tensor in [-1, 1], shape (..., T)."""
    if isinstance(wav, np.ndarray):
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        else:
            wav = wav.astype(np.float32)
        wav = torch.from_numpy(np.ascontiguousarray(wav))
    elif isinstance(wav, torch.Tensor):
        if wav.dtype == torch.int16:
            wav = wav.float() / 32768.0
        else:
            wav = wav.float()
    else:
        raise TypeError(f"unsupported waveform type: {type(wav)}")
    return wav


def normalize(feat: torch.Tensor) -> torch.Tensor:
    """Per-feature (per-coefficient) mean/std normalization over the time axis.

    feat: (..., n_mfcc, frames). Normalizes each MFCC coefficient across time,
    the same 'per_feature' scheme NeMo uses for MatchboxNet.
    """
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    return (feat - mean) / (std + _EPS)


def mfcc(wav, transform: torchaudio.transforms.MFCC | None = None) -> torch.Tensor:
    """Waveform -> normalized MFCC features.

    wav: int16/float ndarray or tensor, shape (T,) or (B, T).
    transform: optional pre-placed MFCC transform (e.g. on GPU) for batched use;
               defaults to the module-level CPU transform.
    Returns: (n_mfcc, frames) for a 1-D input, or (B, n_mfcc, frames) for batched.
    """
    x = _to_float_tensor(wav)
    tf = transform if transform is not None else _MFCC
    if transform is not None:
        # Match the input to the (possibly GPU-placed) transform. dct_mat is a
        # registered buffer on torchaudio's MFCC, so it carries the device.
        x = x.to(transform.dct_mat.device)
    feat = tf(x)                     # (..., n_mfcc, frames)
    return normalize(feat)


def make_transform(device: str = "cpu") -> torchaudio.transforms.MFCC:
    """A fresh MFCC transform on `device` (for batched GPU feature extraction)."""
    tf = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs=dict(
            n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, center=True, power=2.0,
        ),
    )
    return tf.to(device)


def fit_window(wav: np.ndarray, length: int = WINDOW_SAMPLES) -> np.ndarray:
    """Center-pad or crop a mono clip to exactly `length` samples (int16 or float)."""
    n = wav.shape[-1]
    if n == length:
        return wav
    if n > length:                              # center crop
        start = (n - length) // 2
        return wav[..., start:start + length]
    pad = length - n                            # center pad
    left = pad // 2
    right = pad - left
    return np.pad(wav, (left, right))


if __name__ == "__main__":
    # Shape smoke test: 1.5 s of noise -> (64, N_FRAMES).
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(WINDOW_SAMPLES) * 3000).astype(np.int16)
    f = mfcc(clip)
    print(f"mfcc shape: {tuple(f.shape)}  (expected (64, {N_FRAMES}))")
    print(f"per-feature mean~0: {float(f.mean()):+.4f}  std~1: {float(f.std()):.4f}")
