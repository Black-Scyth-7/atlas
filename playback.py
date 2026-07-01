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
from typing import Iterable, Iterator, Optional

import numpy as np
import sounddevice as sd

_SENTINEL = object()  # marks end of the chunk stream in the queue


def play(audio: np.ndarray, sample_rate: int) -> None:
    """Play a single float32 clip and block until it finishes."""
    if audio.size == 0:
        return
    sd.play(audio, sample_rate)
    sd.wait()


def play_stream(
    chunks: Iterable[np.ndarray],
    sample_rate: int,
    stop_event: Optional[threading.Event] = None,
    reference_sink=None,
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
                stream.write(chunk)
    finally:
        # On a clean finish (not a barge-in stop), write a little trailing
        # silence so the last samples of the final sentence aren't clipped when
        # the output stream stops.
        if stop_event is None or not stop_event.is_set():
            try:
                pad = np.zeros(int(sample_rate * 0.15), dtype="float32")
                stream.write(pad)
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
