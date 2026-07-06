r"""Live-mic wake-word tester — verify models/atlas.onnx in the REAL runtime path.

Streams your mic through the exact detector the app uses (WakeDetector), in the
same 80 ms chunks / rolling 1.5 s window, and applies the same threshold +
consecutive-frame gating as main.py. Use it to confirm the trained model behaves
before (or instead of) launching the full assistant — no auth, no follow-up mode.

    wake_pytorch\.venv\Scripts\python wake_pytorch\live_mic_test.py
    ... live_mic_test.py --device 2 --threshold 0.75 --consecutive 3

Say "Atlas" -> you should see a WAKE. Say "hello", "yellow", chatter -> it should
stay quiet. A live bar shows P("Atlas") each frame. Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import WakeDetector
from features import SAMPLE_RATE

CHUNK = 1280   # 80 ms @ 16 kHz — same cadence as the app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent
                                           / "models" / "atlas.onnx"))
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    det = WakeDetector(args.model)
    print(f"model: {args.model}")
    print(f"mic:   {sd.query_devices(args.device, 'input')['name']}")
    print(f"gate:  threshold={args.threshold}  consecutive={args.consecutive}\n")
    print("Say 'Atlas' (should WAKE); say 'hello'/'yellow'/chatter (should stay "
          "quiet). Ctrl+C to stop.\n")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=CHUNK, device=args.device)
    stream.start()
    over = 0
    peak = 0.0
    wakes = 0
    try:
        while True:
            frame, _ = stream.read(CHUNK)
            p = det.predict(frame[:, 0])
            peak = max(peak, p)
            over = over + 1 if p >= args.threshold else 0
            bar = "#" * int(p * 40)
            hot = "  <== WAKE" if over >= args.consecutive else ""
            if hot:
                wakes += 1
                over = 0                    # re-arm after a detection
                det.reset()
            sys.stdout.write(f"\rP={p:0.3f} |{bar:<40}| peak={peak:0.2f} "
                             f"wakes={wakes}{hot}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Detected {wakes} wake(s); session peak P={peak:.3f}.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
