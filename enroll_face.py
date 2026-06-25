"""Enroll a face from the webcam (the visual parallel to enroll.py).

    python enroll_face.py <name> [num_shots]

Captures a few frames a moment apart (so small movements add variety) and saves
their embeddings under <name> in the face database (config.FaceConfig.db_path).
Run again with the same name to add more samples, or a new name to add a person.
"""

import sys
import time

from config import FaceConfig
from face_id import FaceRecognizer
from vision import capture_camera


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else input("Name to enroll: ").strip()
    if not name:
        raise SystemExit("No name given.")
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    fr = FaceRecognizer(FaceConfig())
    if not fr.available:
        raise SystemExit(f"Face recognition unavailable: {fr.reason}")

    print(f"Enrolling '{name}': taking {shots} shots — look at the camera "
          "(vary your angle slightly)...")
    saved = 0
    for i in range(shots):
        msg = fr.enroll(capture_camera(0), name)
        print(f"  [{i + 1}/{shots}] {msg}")
        if msg.startswith("Saved"):
            saved += 1
        time.sleep(0.7)

    print(f"Done. {saved}/{shots} shots saved. Enrolled people: {fr.names()}")


if __name__ == "__main__":
    main()
