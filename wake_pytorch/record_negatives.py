r"""Record YOUR voice saying things that are NOT "Atlas" -> hard negatives.

The strongest negative for a wake word is the *same speaker* saying other words:
it forces the model to key on the word "Atlas", not just "this person's voice".
MUSAN speech only covers other people, so adding your own non-wake speech (and
especially phonetically similar words) sharply cuts false wakes on your chatter.

For each take you read a short prompt; it records continuous speech, then slices
it into overlapping 1.5 s windows (only the voiced ones) saved to
data/negative_real/ in the same format as every other training clip.

    wake_pytorch\.venv\Scripts\python wake_pytorch\record_negatives.py            # ~20 prompts
    ... record_negatives.py --takes 30 --device 1

IMPORTANT: never actually say "Atlas" during these takes. 'r'=redo last, 'q'=quit.
Then retrain:  python wake_pytorch/train.py
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from features import SAMPLE_RATE, WINDOW_SAMPLES

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "negative_real"

FRAME = 320                     # 20 ms @ 16 kHz
ONSET_RMS = 350.0               # int16 RMS to count a frame as speech
SILENCE_TAIL_MS = 1200          # stop a take after this much trailing quiet
MAX_TAKE_MS = 15000             # hard cap on one take
ONSET_TIMEOUT_S = 6.0
HOP_SAMPLES = WINDOW_SAMPLES // 2      # 0.75 s hop between sliced windows
MIN_VOICED_FRAC = 0.15                 # drop near-silent windows

# The HARDEST negatives: words/phrases that sound like "Atlas" (/ˈæt.ləs/) but
# aren't the wake word. Reading these in YOUR voice teaches the model the precise
# boundary of the wake word. NEVER say the actual word "Atlas". Shown a few per
# take so you can read them as a short list.
CONFUSABLES = [
    "at last", "a class", "at us", "add less", "hatless", "artless",
    "cutlass", "callous", "Dallas", "Wallace", "palace", "malice",
    "chalice", "solace", "atlantic", "atlanta", "actress", "cactus",
    "access", "practice", "flatlands", "matches", "lattice", "notice",
]

# Ordinary conversational speech — general non-wake negatives.
SENTENCES = [
    "the athlete practiced every single morning",
    "what time is the meeting scheduled for tomorrow",
    "turn the volume up a little bit please",
    "one two three four five six seven eight nine ten",
    "the weather today is cloudy with a chance of rain",
    "open the browser and search for a recipe",
    "my favorite color is somewhere between blue and green",
    "remind me to buy groceries after work",
    "the last bus leaves the station at midnight",
    "he grabbed his jacket and headed out the door",
    "can you play some music while I cook dinner",
    "the mountains looked massive against the sky",
]


def _build_prompts() -> list[tuple[str, str]]:
    """(label, text) takes: sound-alike word groups first, then sentences."""
    takes: list[tuple[str, str]] = []
    for i in range(0, len(CONFUSABLES), 4):       # 4 soundalikes per take
        group = CONFUSABLES[i:i + 4]
        takes.append(("SOUND-ALIKE words (say each clearly)",
                      "   /   ".join(group)))
    for s in SENTENCES:
        takes.append(("sentence", s))
    return takes


PROMPTS = _build_prompts()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def _capture_take(stream) -> np.ndarray | None:
    """Record continuous speech until trailing silence; return raw int16 audio."""
    preroll: deque[np.ndarray] = deque(maxlen=8)
    recorded: list[np.ndarray] = []
    silence_tail = SILENCE_TAIL_MS // 20
    max_frames = MAX_TAKE_MS // 20
    onset_timeout = int(ONSET_TIMEOUT_S * 1000 / 20)

    started = False
    silent = waited = 0
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
                    return None
        else:
            recorded.append(pcm)
            silent = 0 if r >= ONSET_RMS else silent + 1
            if silent >= silence_tail or len(recorded) >= max_frames:
                break
    if peak >= 32000:
        print("  ! too loud (clipped) — move back or lower gain; redoing.")
        return None
    return np.concatenate(recorded)


def _slice_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Slice raw audio into overlapping 1.5 s windows; keep the voiced ones."""
    windows = []
    if audio.shape[0] < WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - audio.shape[0]
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    for start in range(0, audio.shape[0] - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        w = audio[start:start + WINDOW_SAMPLES].astype(np.int16)
        if np.mean(np.abs(w) > ONSET_RMS) >= MIN_VOICED_FRAC:
            windows.append(w)
    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=len(PROMPTS),
                    help=f"number of takes (default {len(PROMPTS)}: "
                         f"{ (len(CONFUSABLES)+3)//4 } sound-alike + {len(SENTENCES)} sentences)")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    OUT.mkdir(parents=True, exist_ok=True)
    existing = sorted(OUT.glob("neg_*.wav"))
    idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
    if existing:
        print(f"{len(existing)} negative windows already in {OUT}; appending.")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=FRAME, device=args.device)
    stream.start()
    print(f"\nRecording ~{args.takes} spoken takes (NOT the word 'Atlas'). "
          f"Mic: {sd.query_devices(args.device, 'input')['name']}")
    print("Press Enter, read the prompt aloud, pause when done. "
          "('r'=redo last, 'q'=quit)\n")

    saved_windows = 0
    last_files: list[str] = []
    try:
        for t in range(args.takes):
            label, text = PROMPTS[t % len(PROMPTS)]
            print(f'[{t + 1}/{args.takes}] {label}:')
            print(f'      {text}')
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r" and last_files:
                for f in last_files:
                    Path(f).unlink(missing_ok=True)
                saved_windows -= len(last_files)
                last_files = []
                print("  (removed last take — re-record it now)")
            print("  listening... read the sentence")
            try:
                stream.read(stream.read_available or FRAME)     # drain pre-prompt audio
            except Exception:
                pass
            audio = _capture_take(stream)
            if audio is None:
                print("  (no clear speech — try again)")
                continue
            windows = _slice_windows(audio)
            last_files = []
            for w in windows:
                path = OUT / f"neg_{idx:05d}.wav"
                sf.write(str(path), w, SAMPLE_RATE)
                last_files.append(str(path))
                idx += 1
            saved_windows += len(windows)
            print(f"  +{len(windows)} windows ({len(audio)/SAMPLE_RATE:.1f}s speech)")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()

    total = len(list(OUT.glob("neg_*.wav")))
    print(f"\nDone. +{saved_windows} new windows; {total} total in {OUT}.")
    print("Now retrain:  wake_pytorch\\.venv\\Scripts\\python wake_pytorch\\train.py")


if __name__ == "__main__":
    main()
