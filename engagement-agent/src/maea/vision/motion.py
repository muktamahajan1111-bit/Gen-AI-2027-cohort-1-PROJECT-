"""Per-person movement estimate from face-box displacement."""
from __future__ import annotations

import math

from maea.vision.tracker import BBox


class MotionEstimator:
    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, float]] = {}

    def update(self, track_id: int, bbox: BBox) -> float:
        """Displacement of the face centre since the last frame, in face-widths."""
        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        prev = self._prev.get(track_id)
        self._prev[track_id] = (cx, cy)
        if prev is None or bbox[2] <= 0:
            return 0.0
        return math.hypot(cx - prev[0], cy - prev[1]) / bbox[2]
