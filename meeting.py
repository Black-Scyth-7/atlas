"""Meeting recorder: capture audio in the background for later transcription.

Records from the microphone on its own sounddevice stream (separate from the
main wake-word stream, so the assistant keeps listening for "stop meeting"). The
mic captures your voice, anyone on open speakers, and in-person rooms. It does
NOT capture system audio when you're on headphones — that needs WASAPI loopback,
which this sounddevice version doesn't expose (a future add via the `soundcard`
library).

Used by the start_meeting / stop_meeting tools (see tools.py).
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from config import Config


class MeetingRecorder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.active = False

    def start(self) -> bool:
        """Begin recording in the background. Returns False if already recording
        or the stream can't open."""
        if self.active:
            return False
        self._frames = []

        def callback(indata, frames, time_info, status):
            with self._lock:
                self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.cfg.sample_rate, channels=1, dtype="float32",
                callback=callback)
            self._stream.start()
            self.active = True
            return True
        except Exception:
            self._stream = None
            return False

    def stop(self) -> np.ndarray:
        """Stop and return the recorded float32 mono clip (empty if nothing)."""
        if not self.active:
            return np.empty(0, dtype=np.float32)
        self.active = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(frames)[:, 0].astype(np.float32)

    def duration_s(self) -> float:
        with self._lock:
            n = sum(len(f) for f in self._frames)
        return n / self.cfg.sample_rate
