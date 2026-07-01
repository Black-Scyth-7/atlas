"""Stage 1: shared mic stream, wake word detection, and VAD-gated recording.

Small functions that operate on one long-lived InputStream so the wake-word
loop and the command recorder share a single open mic.

Run this file directly for a standalone smoke test:
    python audio_input.py
It waits for the wake word, records your command until you stop talking, and
saves it to command.wav so you can play it back.
"""

import wave
from collections import deque
from typing import Protocol, Tuple

import numpy as np
import sounddevice as sd
import webrtcvad
import openwakeword
from openwakeword.model import Model

from config import Config


class MicLike(Protocol):
    """A readable mic stream — the raw sd.InputStream or a DSP wrapper of it."""
    def read(self, frames: int) -> Tuple[np.ndarray, bool]: ...


def open_stream(cfg: Config) -> sd.InputStream:
    """Open and start the single shared 16 kHz mono input stream."""
    stream = sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=1,
        dtype="int16",
        blocksize=cfg.frame_samples,
    )
    stream.start()
    return stream


def load_wake_model(cfg: Config) -> Model:
    """Load the openWakeWord model, downloading base + pretrained models once."""
    # Idempotent: fetches the melspectrogram/embedding base models and the
    # bundled pretrained wake words on first run, then no-ops. Needs internet
    # the first time only.
    openwakeword.utils.download_models() # type: ignore
    return Model(
        wakeword_models=[cfg.wake_model],
        inference_framework=cfg.wake_framework,
    )


def wait_for_wake_word(stream: MicLike, oww: Model, cfg: Config,
                       interrupt=None) -> bool:
    """Block until the wake word is detected. Returns True when it is.

    If `interrupt` (a threading.Event) is given and gets set — e.g. the user
    pressed the text-input key — this returns False instead, so the caller can
    switch to typed input.

    Reads RAW (un-denoised) audio: the wake-word model is trained on raw mic
    audio, and noise suppression shifts the signal enough to hurt detection.
    Noise suppression is applied only to the recorded command (read()).
    """
    read = getattr(stream, "read_raw", None) or stream.read
    buf = np.empty(0, dtype=np.int16)
    warmup = max(0, getattr(cfg, "wake_warmup_chunks", 0))
    need = max(1, getattr(cfg, "wake_consecutive", 1))
    oww.reset()
    seen = 0    # chunks scored since reset (to skip the warm-up spike window)
    run = 0     # consecutive chunks currently over threshold
    while True:
        if interrupt is not None and interrupt.is_set():
            return False
        frame, _ = read(cfg.frame_samples)
        buf = np.concatenate([buf, frame[:, 0]])
        while len(buf) >= cfg.wake_chunk:
            chunk, buf = buf[: cfg.wake_chunk], buf[cfg.wake_chunk:]
            scores = oww.predict(np.ascontiguousarray(chunk))
            seen += 1
            if seen <= warmup:      # let the buffers fill; ignore early spikes
                run = 0
                continue
            if max(scores.values()) >= cfg.wake_threshold:  # type: ignore
                run += 1
                if run >= need:
                    oww.reset()
                    return True
            else:
                run = 0


def record_until_silence(stream: MicLike, vad: webrtcvad.Vad,
                         cfg: Config,
                         start_timeout_ms: int | None = None) -> np.ndarray:
    """Record a command until silence_tail_ms of trailing quiet. Returns float32.

    Leading silence is dropped (only a short pre-roll is kept) so the clip starts
    at speech — important for the speaker gate. Returns an empty array if no
    speech is detected, so callers can tell "nothing said" from "said something".

    start_timeout_ms: if set, give up and return empty when no speech begins
    within that window. Used for the hands-free follow-up listen; left None for a
    normal post-wake-word command (waits up to max_command_ms).
    """
    recorded: list[np.ndarray] = []
    preroll: deque[np.ndarray] = deque(maxlen=cfg.preroll_frames)
    silence_run = 0
    spoke_at_all = False
    silence_limit = cfg.silence_tail_ms // cfg.frame_ms
    max_frames = cfg.max_command_ms // cfg.frame_ms
    start_limit = (start_timeout_ms // cfg.frame_ms) if start_timeout_ms else None

    for i in range(max_frames):
        frame, _ = stream.read(cfg.frame_samples)
        pcm = frame[:, 0]
        is_speech = vad.is_speech(pcm.tobytes(), cfg.sample_rate)

        if not spoke_at_all:
            preroll.append(pcm)
            if is_speech:
                spoke_at_all = True
                recorded.extend(preroll)  # include the short lead-in
                silence_run = 0
            elif start_limit is not None and i >= start_limit:
                return np.empty(0, dtype=np.float32)  # no follow-up within window
        else:
            recorded.append(pcm)
            silence_run = 0 if is_speech else silence_run + 1
            if silence_run >= silence_limit:
                break

    if not spoke_at_all:
        return np.empty(0, dtype=np.float32)
    audio_int16 = np.concatenate(recorded)
    return audio_int16.astype(np.float32) / 32768.0


def record_fixed(stream: MicLike, seconds: float, cfg: Config) -> np.ndarray:
    """Record a fixed duration (used for enrollment). Returns float32."""
    frames = int(seconds * 1000 / cfg.frame_ms)
    chunks = [stream.read(cfg.frame_samples)[0][:, 0] for _ in range(frames)]
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> str:
    """Write a float32 [-1, 1] mono clip to a 16-bit PCM WAV. Returns the path."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    # Standalone smoke test for build-order Step 1:
    #   wake word -> record command -> save a WAV you can play back.
    cfg = Config()

    print("Loading wake-word model (first run downloads it)...")
    oww = load_wake_model(cfg)
    vad = webrtcvad.Vad(cfg.vad_aggressiveness)

    try:
        stream = open_stream(cfg)
    except Exception as e:  # mic unavailable / bad device / driver error
        raise SystemExit(f"Could not open the microphone: {e}")

    print(f"Ready. Say the wake word ('{cfg.wake_model}'). Ctrl+C to quit.\n")
    try:
        while True:
            wait_for_wake_word(stream, oww, cfg)
            print("Wake word detected — recording your command...")
            audio = record_until_silence(stream, vad, cfg)
            if audio.size == 0:
                print("  No speech detected. Try again.\n")
                continue
            path = save_wav("command.wav", audio, cfg.sample_rate)
            dur = audio.size / cfg.sample_rate
            print(f"  Saved {dur:.1f}s to {path}. Play it back to verify.\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()
