"""Live face-recognition window: the webcam feed with boxes + names drawn on it.

The camera window appears immediately; the face model (InsightFace, CPU) loads on
a background thread and starts drawing boxes/names once ready (a few seconds), so
there's no long blank wait. Known people (enrolled via enroll_face.py) are boxed
green with their name + match score; unrecognized faces are boxed red "unknown".

Run standalone:
    python face_window.py            # default camera
    python face_window.py 1          # camera index 1
Press 'q' (or close the window) to quit.

Atlas can also open it by voice via the open_face_window tool (it launches this
as a subprocess so the assistant keeps running).
"""

from __future__ import annotations

import sys
import threading
import time

_WINDOW = "Atlas - Face Recognition (press q to close)"


def run(camera: int = 0) -> None:
    import cv2
    import numpy as np  # noqa: F401  (cv2 needs numpy importable)
    from PIL import Image

    backend = getattr(cv2, "CAP_DSHOW", 0)
    cap = cv2.VideoCapture(camera, backend) if backend else cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Couldn't open camera {camera}.")
        return

    state = {"frame": None, "results": [], "status": "loading face model..."}
    lock = threading.Lock()
    stop = threading.Event()

    def recognize_loop() -> None:
        # Load the (slow) model here so the window can show the camera instantly.
        from config import FaceConfig
        from face_id import FaceRecognizer
        fr = FaceRecognizer(FaceConfig())
        with lock:
            state["status"] = "" if fr.available else f"faces off: {fr.reason}"
        if not fr.available:
            return
        while not stop.is_set():
            with lock:
                frame = state["frame"]
            if frame is None:
                time.sleep(0.03)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = fr.identify(Image.fromarray(rgb))
            except Exception:
                res = []
            with lock:
                state["results"] = res
            time.sleep(0.05)

    threading.Thread(target=recognize_loop, daemon=True).start()

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    try:                                  # try to bring it to the front
        cv2.setWindowProperty(_WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    print("Face window open. Press 'q' to close.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with lock:
                state["frame"] = frame
                results = list(state["results"])
                status = state["status"]

            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                known = r["name"] != "unknown"
                color = (0, 200, 0) if known else (0, 0, 255)   # BGR
                label = f"{r['name']} {r['score']:.2f}" if known else "unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, max(y1 - 22, 0)),
                              (x1 + 12 + 9 * len(label), max(y1, 22)), color, -1)
                cv2.putText(frame, label, (x1 + 5, max(y1 - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if status:
                cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 220), 2)

            cv2.imshow(_WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
