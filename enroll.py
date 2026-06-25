"""One-time enrollment: record a few clips of your voice and save a voiceprint.

Run this once before turning on the speaker gate:
    python enroll.py

This builds your reference voiceprint by averaging several short clips. It is
PERSONALIZATION, not security — a recording of your voice would also pass.
"""

from config import Config
import audio_input
from speaker_id import SpeakerVerifier

NUM_SAMPLES = 5
SECONDS_EACH = 4.0

PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "I would like to check the weather and my schedule today.",
    "Please set a timer for ten minutes from now.",
    "Tell me something interesting about the solar system.",
    "Atlas, this is my voice for verification.",
]


def main() -> None:
    cfg = Config()

    print("Loading speaker model (first run downloads it)...")
    verifier = SpeakerVerifier(cfg)

    try:
        stream = audio_input.open_stream(cfg)
    except Exception as e:
        raise SystemExit(f"Could not open the microphone: {e}")

    print(f"\nEnrollment: you'll read {NUM_SAMPLES} short phrases "
          f"({SECONDS_EACH:.0f}s each).\n")
    samples = []
    try:
        for i in range(NUM_SAMPLES):
            phrase = PHRASES[i % len(PHRASES)]
            input(f"[{i + 1}/{NUM_SAMPLES}] Press Enter, then say: \"{phrase}\"")
            print("  recording...")
            clip = audio_input.record_fixed(stream, SECONDS_EACH, cfg)
            samples.append(clip)
            print("  done.\n")
    finally:
        stream.stop()
        stream.close()

    print("Building voiceprint...")
    voiceprint = verifier.build_voiceprint(samples)
    verifier.save_voiceprint(voiceprint)
    print(f"Saved to {cfg.voiceprint_path}. You're enrolled.")
    print("Now run `python speaker_id.py` to test accept/reject scoring.")


if __name__ == "__main__":
    main()
