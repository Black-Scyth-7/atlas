"""Mic-side audio DSP: RNNoise noise suppression + acoustic echo cancellation.

Two best-effort processors that clean the microphone signal:

  NoiseSuppressor  - RNNoise (a small recurrent denoiser, native 48 kHz) wrapped
                     to run on the project's 16 kHz frames. Removes background
                     room noise so the wake word, VAD, and STT keep working in a
                     noisy room. RNNoise ships as a tiny DLL inside the
                     `pyrnnoise` package; we load that low-level ctypes binding
                     by file path to avoid the package's heavy (and on Windows,
                     broken) PyAV import chain.

  EchoCanceller    - a partitioned frequency-domain NLMS adaptive filter. Given
                     the signal being played out the speaker (the "reference"),
                     it estimates and subtracts the echo picked up by the mic, so
                     Atlas's own voice doesn't false-trigger the wake word during
                     playback (barge-in without headphones).

Both degrade gracefully: if RNNoise or SciPy isn't available the relevant stage
becomes a pass-through and `available` is False.

Standalone self-test (synthetic signals, no mic needed):
    python audio_dsp.py
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from config import DspConfig

_SR = 16000
_RNN_SR = 48000


# ---- RNNoise low-level loader (bypasses pyrnnoise/__init__ -> audiolab/av) ----
def _load_rnnoise():
    """Return the pyrnnoise.rnnoise ctypes module, or None if unavailable."""
    try:
        spec = importlib.util.find_spec("pyrnnoise")  # no __init__ execution
        if spec is None or not spec.submodule_search_locations:
            return None
        path = os.path.join(spec.submodule_search_locations[0], "rnnoise.py")
        mod_spec = importlib.util.spec_from_file_location("atlas_rnnoise", path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


class NoiseSuppressor:
    """RNNoise denoiser for int16 16 kHz frames (resamples to/from 48 kHz).

    process() returns exactly as many samples as it's given, using overlap-save
    resampling so there are no per-frame boundary clicks.
    """

    _HIST16 = 120          # input history kept for overlap-save (16 kHz)
    _HIST48 = 360          # output history kept for overlap-save (48 kHz)

    def __init__(self, sample_rate: int = _SR):
        self.sr = sample_rate
        self.available, self.reason = False, ""
        self._rnn = None
        self._state = None
        try:
            from scipy.signal import resample_poly  # noqa: F401
        except Exception:
            self.reason = "scipy not available"
            return
        rnn = _load_rnnoise()
        if rnn is None:
            self.reason = "rnnoise (pyrnnoise) not available"
            return
        try:
            self._rnn = rnn
            self._state = rnn.create()
            self._frame = rnn.FRAME_SIZE      # 480 samples @ 48 kHz
            self._hist16 = np.zeros(self._HIST16, dtype=np.float32)
            self._hist48 = np.zeros(self._HIST48, dtype=np.float32)
            self._pending48 = np.zeros(0, dtype=np.float32)
            self.available = True
        except Exception as e:
            self.reason = f"rnnoise init failed ({e})"

    def __del__(self):
        try:
            if self._rnn is not None and self._state is not None:
                self._rnn.destroy(self._state)
        except Exception:
            pass

    def process(self, frame_i16: np.ndarray) -> np.ndarray:
        """Denoise one int16 mono frame; returns int16 of the same length."""
        if not self.available or frame_i16.size == 0:
            return frame_i16
        from scipy.signal import resample_poly

        n = frame_i16.size
        x = frame_i16.astype(np.float32)

        # 16 kHz -> 48 kHz with left-context (overlap-save) to avoid transients.
        ext = np.concatenate([self._hist16, x])
        up = resample_poly(ext, 3, 1)
        up = up[-n * 3:]                      # valid region (drops the lead-in)
        self._hist16 = x[-self._HIST16:]

        # Denoise in 480-sample (10 ms @48 kHz) RNNoise frames.
        self._pending48 = np.concatenate([self._pending48, up])
        den_parts = []
        while self._pending48.size >= self._frame:
            blk = self._pending48[: self._frame]
            self._pending48 = self._pending48[self._frame:]
            blk_i16 = np.clip(blk, -32768, 32767).astype(np.int16)
            out_i16, _ = self._rnn.process_mono_frame(self._state, blk_i16)
            den_parts.append(out_i16.astype(np.float32))
        if not den_parts:
            return np.zeros(n, dtype=np.int16)
        den48 = np.concatenate(den_parts)

        # 48 kHz -> 16 kHz, again with left-context.
        ext48 = np.concatenate([self._hist48, den48])
        down = resample_poly(ext48, 1, 3)
        self._hist48 = den48[-self._HIST48:]
        out = down[-n:] if down.size >= n else np.pad(down, (n - down.size, 0))
        return np.clip(out, -32768, 32767).astype(np.int16)

    def reset(self):
        if self.available:
            self._hist16[:] = 0
            self._hist48[:] = 0
            self._pending48 = np.zeros(0, dtype=np.float32)


class EchoCanceller:
    """Partitioned frequency-domain NLMS adaptive filter (overlap-save).

    cancel(mic, ref) returns the echo-cancelled mic signal. `ref` is the audio
    being played out the speaker, resampled to the mic rate and roughly aligned
    in time. Works on int16; processes internally in float blocks of B samples.
    """

    def __init__(self, cfg: DspConfig, sample_rate: int = _SR):
        self.available = True
        self.B = max(64, int(cfg.aec_block))
        self.N = 2 * self.B
        self.P = max(1, int(cfg.aec_filter_blocks))
        self.mu = float(cfg.aec_mu)
        self._bins = self.N // 2 + 1
        self._W = [np.zeros(self._bins, dtype=np.complex128) for _ in range(self.P)]
        self._Xh = [np.zeros(self._bins, dtype=np.complex128) for _ in range(self.P)]
        self._xprev = np.zeros(self.B, dtype=np.float64)
        self._eps = 1e-6

    def _block(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        # x, d: B float samples (reference, mic). Returns B residual samples.
        xblk = np.concatenate([self._xprev, x])           # length N
        X = np.fft.rfft(xblk)
        self._xprev = x
        self._Xh.pop()
        self._Xh.insert(0, X)
        Y = np.zeros(self._bins, dtype=np.complex128)
        for w, xh in zip(self._W, self._Xh):
            Y += w * xh
        y = np.fft.irfft(Y, n=self.N)[self.B:]            # overlap-save tail
        e = d - y
        E = np.fft.rfft(np.concatenate([np.zeros(self.B), e]))
        norm = self._eps
        for xh in self._Xh:
            norm = norm + np.abs(xh) ** 2
        step = self.mu * E / norm
        for i, xh in enumerate(self._Xh):
            self._W[i] += np.conj(xh) * step
        return e

    def cancel(self, mic_i16: np.ndarray, ref_i16: np.ndarray) -> np.ndarray:
        n = mic_i16.size
        if not self.available or n == 0:
            return mic_i16
        mic = mic_i16.astype(np.float64)
        ref = np.zeros(n, dtype=np.float64)
        m = min(n, ref_i16.size)
        ref[:m] = ref_i16[:m].astype(np.float64)
        # pad to a whole number of B-blocks
        pad = (-n) % self.B
        if pad:
            mic = np.concatenate([mic, np.zeros(pad)])
            ref = np.concatenate([ref, np.zeros(pad)])
        out = np.empty_like(mic)
        for s in range(0, mic.size, self.B):
            out[s:s + self.B] = self._block(ref[s:s + self.B], mic[s:s + self.B])
        out = out[:n]
        return np.clip(out, -32768, 32767).astype(np.int16)

    def reset(self):
        for w in self._W:
            w[:] = 0
        for xh in self._Xh:
            xh[:] = 0
        self._xprev[:] = 0


class MicStream:
    """Wraps a sounddevice InputStream so reads come back noise-suppressed.

    read()      -> noise-suppressed frames (used by the main wake/record loop).
    read_raw()  -> the untouched frames (used by the barge-in watcher, which
                   runs its own echo canceller and wants the raw echo intact).
    """

    def __init__(self, stream, suppressor: "NoiseSuppressor | None" = None):
        self._s = stream
        self._ns = suppressor

    def read(self, frames):
        block, overflowed = self._s.read(frames)
        if self._ns is not None and self._ns.available:
            pcm = self._ns.process(block[:, 0]).reshape(-1, 1)
            return pcm, overflowed
        return block, overflowed

    def read_raw(self, frames):
        return self._s.read(frames)

    def drain(self):
        """Discard any already-buffered input frames.

        While Atlas is speaking, the mic keeps capturing into the stream's
        buffer — including Atlas's own voice (which may contain the wake word,
        e.g. the 'Atlas is online' greeting). Draining that backlog before we
        start listening for the wake word stops Atlas from waking itself.
        """
        try:
            avail = self._s.read_available
            while avail > 0:
                self._s.read(min(avail, 4096))
                avail = self._s.read_available
        except Exception:
            pass

    def stop(self):
        try:
            self._s.stop()
        except Exception:
            pass

    def close(self):
        try:
            self._s.close()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._s, name)


class ReferenceBuffer:
    """Thread-safe store of recently-played audio, resampled to the mic rate.

    The playback thread pushes chunks it writes to the speaker; the barge-in
    watcher pulls the matching number of reference samples (with an optional
    bulk delay) to feed the echo canceller.
    """

    def __init__(self, cfg: DspConfig, sample_rate: int = _SR):
        import threading

        self.sr = sample_rate
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.int16)
        self._delay = max(0, int(cfg.aec_ref_delay_ms) * sample_rate // 1000)
        self._maxlen = sample_rate * 4   # keep ~4 s

    def push(self, chunk: np.ndarray, chunk_rate: int) -> None:
        """Add played audio (float32 [-1,1] at chunk_rate) to the reference."""
        try:
            data = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk_rate != self.sr:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(chunk_rate), self.sr)
                data = resample_poly(data, self.sr // g, int(chunk_rate) // g)
            i16 = np.clip(data, -1.0, 1.0) * 32767.0
            with self._lock:
                self._buf = np.concatenate([self._buf, i16.astype(np.int16)])
                if self._buf.size > self._maxlen:
                    self._buf = self._buf[-self._maxlen:]
        except Exception:
            pass

    def take(self, n: int) -> np.ndarray:
        """Pop the next n reference samples aligned to the mic (int16)."""
        with self._lock:
            avail = self._buf.size - self._delay
            if avail < n:
                out = np.zeros(n, dtype=np.int16)
                k = max(0, avail)
                if k:
                    out[:k] = self._buf[:k]
                    self._buf = self._buf[k:]
                return out
            out = self._buf[:n].copy()
            self._buf = self._buf[n:]
            return out

    def clear(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.int16)


if __name__ == "__main__":
    # --- Synthetic self-test: prove NS reduces noise and AEC reduces echo. ---
    rng = np.random.default_rng(0)

    ns = NoiseSuppressor()
    print("NoiseSuppressor available:", ns.available, "|", ns.reason or "ok")
    if ns.available:
        t = np.arange(_SR) / _SR
        speech = 0.25 * np.sin(2 * np.pi * 200 * t)
        noise = 0.25 * rng.standard_normal(_SR)
        noisy = ((speech + noise) * 32767).astype(np.int16)
        clean = np.concatenate([ns.process(noisy[i:i + 480])
                                for i in range(0, _SR - 480, 480)])
        # Compare noise floor in a silent gap (use the trailing pure-noise part).
        sil = slice(0, 480 * 4)
        in_noise = np.std((noise[: clean.size][sil] * 32767))
        out_noise = np.std(clean[sil].astype(float))
        print(f"  noise std in={in_noise:.0f} out={out_noise:.0f} "
              f"(len in/out 480 -> {clean.size // ((_SR - 480)//480)} per frame)")

    cfg = DspConfig()
    aec = EchoCanceller(cfg)
    # Reference = noise burst; echo = delayed, filtered ref; mic = echo + faint near-end.
    ref = (rng.standard_normal(_SR) * 8000).astype(np.int16)
    h = np.array([0.0] * 20 + [0.8, -0.4, 0.2, -0.1])    # echo path (delay+decay)
    echo = np.convolve(ref.astype(float), h)[: _SR]
    nearend = (rng.standard_normal(_SR) * 300)
    mic = (echo + nearend).astype(np.int16)
    out = np.concatenate([aec.cancel(mic[i:i + 480], ref[i:i + 480])
                          for i in range(0, _SR - 480, 480)])
    half = out.size // 2
    echo_before = np.std(mic[:out.size][half:].astype(float))
    echo_after = np.std(out[half:].astype(float))
    erle = 20 * np.log10(echo_before / max(echo_after, 1e-6))
    print(f"AEC: residual std {echo_before:.0f} -> {echo_after:.0f}  "
          f"(ERLE {erle:.1f} dB; higher = more echo removed)")
