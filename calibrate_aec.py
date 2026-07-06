"""Calibrate `DspConfig.aec_ref_delay_ms` for this machine.

Atlas cancels its own voice from the mic during barge-in with an adaptive echo
filter (audio_dsp.EchoCanceller). That filter only models ~160 ms of echo path,
so if the *reference* (what Atlas played) isn't time-aligned to when the mic
actually hears it, cancellation fails and Atlas interrupts itself — cutting a
reply off at the first sentence. The alignment knob is `aec_ref_delay_ms`, and
its right value is your speaker->air->mic round-trip latency, which varies by
machine and audio backend.

This tool measures it directly:
  1. Plays a short broadband burst out the speaker while recording the mic
     (sounddevice keeps the two time-aligned), and cross-correlates to get the
     coarse round-trip latency.
  2. Sweeps candidate delays through the *real* EchoCanceller on the recording
     and reports ERLE (dB of echo removed) for each, so the pick is the value
     that actually cancels best on your hardware — not just the raw latency.

Run it in a quiet room, speakers on (NOT headphones — we want the echo), at a
normal volume:

    python calibrate_aec.py

Then set the printed value in config.py (DspConfig.aec_ref_delay_ms).
"""

from __future__ import annotations

import sys

import numpy as np

from config import DspConfig
from audio_dsp import EchoCanceller

_SR = 16000            # mic rate the AEC runs at
_BURST_S = 0.6         # length of the broadband probe
_LEAD_S = 0.25         # silence before the burst (lets the stream settle)
_TAIL_S = 0.4          # silence after (captures the full echo tail)


def _probe() -> np.ndarray:
    """A lead-in silence, a white-noise burst, then a tail of silence."""
    rng = np.random.default_rng(0)
    lead = np.zeros(int(_LEAD_S * _SR), dtype=np.float32)
    burst = 0.5 * rng.standard_normal(int(_BURST_S * _SR)).astype(np.float32)
    tail = np.zeros(int(_TAIL_S * _SR), dtype=np.float32)
    return np.concatenate([lead, burst, tail])


def _play_and_record(ref: np.ndarray) -> np.ndarray:
    """Play `ref` and capture the mic, time-aligned to playback start."""
    import sounddevice as sd

    print("Playing a probe tone and recording the mic... stay quiet.")
    rec = sd.playrec(ref, samplerate=_SR, channels=1, dtype="float32")
    sd.wait()
    return np.asarray(rec, dtype=np.float32).reshape(-1)


def _coarse_delay_samples(ref: np.ndarray, mic: np.ndarray) -> int:
    """Cross-correlate mic against ref; return the lag (round-trip) in samples."""
    n = min(ref.size, mic.size)
    r = ref[:n] - ref[:n].mean()
    m = mic[:n] - mic[:n].mean()
    corr = np.correlate(m, r, mode="full")
    lags = np.arange(-(n - 1), n)
    # Physical round-trip delay is a non-negative lag (mic hears ref later).
    pos = lags >= 0
    lag = int(lags[pos][np.argmax(corr[pos])])
    return max(0, lag)


def _erle_for_delay(ref: np.ndarray, mic: np.ndarray, delay: int,
                    cfg: DspConfig) -> float:
    """Echo Return Loss Enhancement (dB) with the reference delayed by `delay`.

    Higher = more of Atlas's own voice removed. Runs a fresh filter so each
    candidate delay is judged from cold, exactly like a new reply.
    """
    r16 = np.clip(ref, -1.0, 1.0) * 32767.0
    m16 = np.clip(mic, -1.0, 1.0) * 32767.0
    ref_d = np.concatenate([np.zeros(delay, dtype=np.float32), r16])[:m16.size]
    ref_i16 = ref_d.astype(np.int16)
    mic_i16 = m16.astype(np.int16)

    aec = EchoCanceller(cfg)
    B = cfg.aec_block
    out = np.concatenate([
        aec.cancel(mic_i16[i:i + B], ref_i16[i:i + B])
        for i in range(0, (mic_i16.size // B) * B, B)
    ]).astype(np.float64)

    # Judge on the echo region only (skip the lead-in silence), and give the
    # filter the first third to adapt before scoring.
    start = int((_LEAD_S + _BURST_S * 0.33) * _SR)
    end = min(out.size, int((_LEAD_S + _BURST_S + _TAIL_S) * _SR))
    if end - start < B:
        return 0.0
    before = np.std(mic_i16[start:end].astype(np.float64))
    after = np.std(out[start:end])
    return 20.0 * np.log10(before / max(after, 1e-6))


def main() -> int:
    try:
        import sounddevice  # noqa: F401
    except Exception as e:
        print(f"sounddevice not available ({e}); cannot calibrate.", file=sys.stderr)
        return 1

    cfg = DspConfig()
    ref = _probe()
    mic = _play_and_record(ref)

    echo = np.std(mic[int(_LEAD_S * _SR):int((_LEAD_S + _BURST_S) * _SR)])
    if echo < 20 / 32767:   # ~ -64 dBFS: essentially silence
        print("\nBarely any echo was captured. Are you on headphones, or is the "
              "speaker muted / volume very low? AEC only matters for open "
              "speakers — on headphones you can just leave aec_ref_delay_ms=0.")
        return 1

    coarse = _coarse_delay_samples(ref, mic)
    print(f"\nCoarse round-trip latency: {coarse} samples "
          f"({coarse * 1000 / _SR:.0f} ms)")

    # Sweep delays around the coarse estimate and keep the best-cancelling one.
    filt_ms = cfg.aec_filter_blocks * cfg.aec_block * 1000 // _SR
    center_ms = coarse * 1000 // _SR
    lo = max(0, center_ms - filt_ms)
    hi = center_ms + filt_ms
    print(f"Filter reach: ~{filt_ms} ms. Sweeping {lo}-{hi} ms for best ERLE:\n")

    best_ms, best_erle = 0, -1e9
    for ms in range(lo, hi + 1, 10):
        erle = _erle_for_delay(ref, mic, ms * _SR // 1000, cfg)
        bar = "#" * max(0, int(erle))
        print(f"  {ms:4d} ms  ERLE {erle:5.1f} dB  {bar}")
        if erle > best_erle:
            best_ms, best_erle = ms, erle

    print(f"\nBest: aec_ref_delay_ms = {best_ms}  (ERLE {best_erle:.1f} dB)")
    if best_erle < 6:
        print("Warning: even the best delay removes little echo (<6 dB). Your "
              "round-trip latency may exceed the filter's reach — consider "
              "raising DspConfig.aec_filter_blocks, or lowering output latency.")
    print(f"\nSet this in config.py:\n    aec_ref_delay_ms: int = {best_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
