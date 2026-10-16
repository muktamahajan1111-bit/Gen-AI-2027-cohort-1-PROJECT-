"""Video input: a webcam index (``0``) or a recorded file. Frames are resampled
to the requested FPS and stamped with session-relative seconds."""
from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np


def iter_frames(source: str | int, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    import cv2

    is_cam = isinstance(source, int) or str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    start, next_t, step = time.monotonic(), 0.0, 1.0 / fps
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (time.monotonic() - start) if is_cam else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 1e-6 < next_t:
                continue
            next_t = t + step
            yield t, frame
    finally:
        cap.release()
