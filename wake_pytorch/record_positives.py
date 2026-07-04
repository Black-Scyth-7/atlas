r"""Record YOUR voice saying "Atlas" -> real positive clips for training.

The first-pass model learned only from Kokoro TTS voices. Adding a few dozen
recordings of your own voice (your mic, your room, your accent) is the single
biggest accuracy win for real-world triggering. This captures them in the exact
same 1.5 s / 16 kHz format as the TTS positives.

    wake_pytorch\.venv\Scripts\python wake_pytorch\record_positives.py            # 40 takes
    ... record_positives.py --takes 60 --device 2      # more takes / pick a mic

For each take: press Enter, say "Atlas" once, clearly. It auto-detects speech,
stops on silence, trims + centers the word, and saves to data/positive_real/.
Vary it a little across takes — normal, quiet, fast, slightly different distance —
so the model generalizes. Say 'r' then Enter to redo the last take, 'q' to stop.
Then retrain:  python wake_pytorch/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from features import SAMPLE_RATE, WINDOW_SAMPLES, fit_window

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "positive_real"

FRAME = 320                     # 20 ms @ 16 kHz
ONSET_RMS = 350.0               # int16 RMS to count a frame as speech (~ -39 dBFS)
SILENCE_TAIL_MS = 500           # stop after this much trailing quiet
MAX_SPEECH_MS = 1500            # hard cap on one utterance
ONSET_TIMEOUT_S = 5.0           # give up waiting for speech this long
PREROLL_FRAMES = 8              # ~160 ms kept before onset


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def _capture_one(stream) -> np.ndarray | None:
    """Wait for speech, record until trailing silence, return centered 1.5 s int16."""
    from collections import deque
    preroll: deque[np.ndarray] = deque(maxlen=PREROLL_FRAMES)
    recorded: list[np.ndarray] = []
    silence_tail = SILENCE_TAIL_MS // 20
    max_frames = MAX_SPEECH_MS // 20
    onset_timeout = int(ONSET_TIMEOUT_S * 1000 / 20)

    started = False
    silent = 0
    waited = 0
    peak = 0.0
    while True:
        frame, _ = stream.read(FRAME)
        pcm = frame[:, 0]
        r = _rms(pcm)
        peak = max(peak, float(np.max(np.abs(pcm))))
        if not started:
            preroll.append(pcm)
            if r >= ONSET_RMS:
                started = True
                recorded.extend(preroll)
            else:
                waited += 1
                if waited >= onset_timeout:
                    return None                       # nothing said
        else:
            recorded.append(pcm)
            silent = 0 if r >= ONSET_RMS else silent + 1
            if silent >= silence_tail or len(recorded) >= max_frames + PREROLL_FRAMES:
                break

    audio = np.concatenate(recorded)
    if peak >= 32000:                                 # clipped — too loud
        print("  ! too loud (clipped) — move back or lower gain; redoing.")
        return None
    return fit_window(audio, WINDOW_SAMPLES).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=40, help="how many clips to record")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    OUT.mkdir(parents=True, exist_ok=True)
    existing = sorted(OUT.glob("real_*.wav"))
    start_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
    if existing:
        print(f"{len(existing)} recordings already in {OUT}; appending.")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=FRAME, device=args.device)
    stream.start()
    print(f"\nRecording {args.takes} takes of \"Atlas\". "
          f"Mic: {sd.query_devices(args.device, 'input')['name']}")
    print("Press Enter, then say 'Atlas'. ('r'=redo last, 'q'=quit)\n")

    saved = 0
    last_path = None
    try:
        i = start_idx
        while saved < args.takes:
            cmd = input(f"[{saved + 1}/{args.takes}] Enter to record> ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r" and last_path is not None:
                Path(last_path).unlink(missing_ok=True)
                saved -= 1
                print("  (removed last take — re-record it now)")
                continue
            print("  listening... say 'Atlas'")
            # Drain any buffered audio from before the prompt.
            try:
                stream.read(stream.read_available or FRAME)
            except Exception:
                pass
            clip = _capture_one(stream)
            if clip is None:
                print("  (no clear speech — try again)")
                continue
            path = OUT / f"real_{i:04d}.wav"
            sf.write(str(path), clip, SAMPLE_RATE)
            last_path = str(path)
            saved += 1
            i += 1
            dur = np.mean(np.abs(clip) > ONSET_RMS)
            print(f"  saved {path.name}  (voiced {dur*100:.0f}% of window)")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()

    total = len(list(OUT.glob("real_*.wav")))
    print(f"\nDone. {saved} new clip(s); {total} total in {OUT}.")
    print("Now retrain:  wake_pytorch\\.venv\\Scripts\\python wake_pytorch\\train.py")


if __name__ == "__main__":
    main()
