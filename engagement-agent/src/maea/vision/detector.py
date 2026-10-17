"""Face detectors behind one interface. MediaPipe gives head pose; Haar is a
dependency-light fallback. Add a YOLO-face detector by implementing ``detect``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    frontal: bool = True


class HaarDetector:
    name = "haar"

    def __init__(self) -> None:
        import cv2
        self._cv2 = cv2
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        # A frontal cascade only fires on forward-facing heads -> use as attention cue.
        return [Detection(bbox=tuple(map(float, f)), frontal=True) for f in faces]


class MediaPipeDetector:
    name = "mediapipe"

    def __init__(self, max_faces: int = 60) -> None:
        import cv2
        import mediapipe as mp
        from maea.vision.head_pose import LANDMARK_IDS
        self._cv2 = cv2
        self._ids = LANDMARK_IDS
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=max_faces,
            refine_landmarks=False, min_detection_confidence=0.5)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        from maea.vision.head_pose import estimate_pose
        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB))
        out: list[Detection] = []
        for face in res.multi_face_landmarks or []:
            xs = np.array([p.x for p in face.landmark]) * w
            ys = np.array([p.y for p in face.landmark]) * h
            bbox = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min()))
            pts = np.array([[xs[i], ys[i]] for i in self._ids])
            pose = estimate_pose(pts, w, h)
            if pose:
                out.append(Detection(bbox, yaw=pose[0], pitch=pose[1], roll=pose[2]))
            else:
                out.append(Detection(bbox))
        return out


def create_detector(name: str = "auto"):
    """``auto`` prefers MediaPipe and falls back to Haar if it is not installed."""
    if name in ("auto", "mediapipe"):
        try:
            return MediaPipeDetector()
        except Exception:
            if name == "mediapipe":
                raise
    return HaarDetector()
