"""Head pose (yaw / pitch / roll) from six facial landmarks using solvePnP."""
from __future__ import annotations

import numpy as np

# Generic 3D face model (mm) matching MediaPipe landmarks 1, 152, 33, 263, 61, 291
_MODEL = np.array([
    (0.0, 0.0, 0.0),            # nose tip
    (0.0, -330.0, -65.0),       # chin
    (-225.0, 170.0, -135.0),    # eye outer corner (image-left)
    (225.0, 170.0, -135.0),     # eye outer corner (image-right)
    (-150.0, -150.0, -125.0),   # mouth corner (image-left)
    (150.0, -150.0, -125.0),    # mouth corner (image-right)
], dtype=np.float64)
LANDMARK_IDS = (1, 152, 33, 263, 61, 291)


def estimate_pose(points_px: np.ndarray, width: int, height: int) -> tuple[float, float, float] | None:
    """``points_px``: (6, 2) pixel coordinates in LANDMARK_IDS order."""
    import cv2

    focal = float(width)
    cam = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL, points_px.astype(np.float64), cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    # RQDecomp reports pitch near +/-180 for this model; fold into [-90, 90].
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return float(yaw), float(pitch), float(roll)
