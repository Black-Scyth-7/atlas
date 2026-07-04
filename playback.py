"""Stage 6b: audio playback, with streaming so speech starts early.

play()        - play a finished clip and block until done.
play_stream() - play a generator of audio chunks. A background producer thread
                pulls/synthesizes the next chunk while the current one plays, so
                the first sentence is heard with minimal latency.

Barge-in: play_stream accepts a `stop_event`. When set, playback stops between
chunks. Wiring the wake word to set that event during playback is left as a
clearly marked hook in main.py (see TODO there).
"""

import queue
import threading
import time
from typing import Iterable, Iterator, Optional

import numpy as np
import sounddevice as sd

import ui_events as ux


_SENTINEL = object()  # marks end of the chunk stream in the queue


def play(audio: np.ndarray, sample_rate: int, tail_padding_ms: int = 300) -> None:
    """Play a single float32 clip and block until it finishes.

    Appends a short trailing silence so the last word isn't clipped when the
    device stream closes (some Windows backends drop whatever is still buffered).
    """
    if audio.size == 0:
        return
    if tail_padding_ms > 0:
        pad = np.zeros(int(sample_rate * tail_padding_ms / 1000), dtype=audio.dtype)
        audio = np.concatenate([audio, pad])
    sd.play(audio, sample_rate)
    sd.wait()


def play_stream(
    chunks: Iterable[np.ndarray],
    sample_rate: int,
    stop_event: Optional[threading.Event] = None,
    reference_sink=None,
    tail_padding_ms: int = 250,
) -> None:
    """Play chunks as they become available; overlap production with playback.

    `chunks` is consumed in a producer thread (e.g. Piper synthesizing later
    sentences) and fed through a small queue to the output stream, so audio
    starts as soon as the first chunk is ready. If `stop_event` is set, playback
    halts at the next chunk boundary (barge-in).

    `reference_sink`, if given, receives every chunk written to the speaker (via
    `.push(chunk, sample_rate)`) so the echo canceller has a reference signal.
    """
    q: "queue.Queue[object]" = queue.Queue(maxsize=8)

    def producer() -> None:
        try:
            for chunk in chunks:
                if stop_event is not None and stop_event.is_set():
                    break
                q.put(chunk)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
    stream.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            if stop_event is not None and stop_event.is_set():
                break
            chunk: np.ndarray = item  # type: ignore[assignment]
            if chunk.size:
                if reference_sink is not None:
                    reference_sink.push(chunk, sample_rate)
                # Live loudness for the GUI orb while Atlas speaks (chunk is
                # float32 in [-1, 1]); no-op with no UI subscribed.
                ux.audio_level(float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))))
                stream.write(chunk)
    finally:
        # On a clean finish (not a barge-in stop), make sure the last samples
        # actually reach the speaker before we stop the stream. On some Windows
        # audio backends stream.stop() clips whatever is still buffered, so we
        # (1) write trailing silence and (2) explicitly wait out the device's
        # output latency + the pad, so nothing at the end of the reply is cut.
        if stop_event is None or not stop_event.is_set():
            try:
                pad_s = max(0, tail_padding_ms) / 1000.0
                if pad_s:
                    stream.write(np.zeros(int(sample_rate * pad_s), dtype="float32"))
                latency = getattr(stream, "latency", 0.1)
                if isinstance(latency, (tuple, list)):
                    latency = latency[-1]
                time.sleep(float(latency or 0.1) + pad_s + 0.05)
            except Exception:
                pass
        stream.stop()
        stream.close()


if __name__ == "__main__":
    # Standalone Step 5 test: synthesize a multi-sentence reply and stream it.
    from config import TTSConfig
    from tts import TTS

    tts = TTS(TTSConfig())
    text = ("Hello, I am Atlas. This sentence plays while the next one is still "
            "being synthesized. Streaming keeps the response feeling instant.")
    print("Speaking (streaming)...")
    play_stream(tts.stream(text), tts.sample_rate)
    print("Done.")
