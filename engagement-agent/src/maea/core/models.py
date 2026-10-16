"""Plain data objects shared by every layer. No logic, no dependencies."""
from __future__ import annotations

from dataclasses import dataclass


def student_label(track_id: int) -> str:
    return f"Student_{track_id:02d}"


@dataclass(frozen=True)
class Observation:
    """What the vision layer saw for one person in one frame."""
    track_id: int
    t: float                      # seconds since session start
    motion: float                 # normalised displacement since last frame
    yaw: float | None = None      # degrees; None if head pose unavailable
    pitch: float | None = None
    frontal: bool = True          # fallback attention cue when pose is unavailable


@dataclass(frozen=True)
class AudioFrame:
    t: float
    rms_db: float
    is_speech: bool


@dataclass
class AudioWindow:
    t_start: float
    t_end: float
    speech_ratio: float
    energy: float                 # 0..1
    turns: int
    mode: str                     # lecture | discussion | silence | mixed | unknown


@dataclass
class StudentWindow:
    student_id: str
    t_start: float
    t_end: float
    attention: float
    movement: float
    inactivity: float
    score: float


@dataclass
class ClassWindow:
    t_start: float
    t_end: float
    score: float
    smoothed: float
    present: int
    attention: float
    inactivity: float
    audio_mode: str
    speech_ratio: float
    turns: int


@dataclass
class TrendEvent:
    kind: str                     # drop | peak
    t: float
    magnitude: float


@dataclass
class Suggestion:
    t: float
    severity: str                 # info | warning | positive
    rule: str
    title: str
    detail: str


@dataclass
class AttendanceEvent:
    student_id: str
    kind: str                     # PRESENT | LEFT | RETURNED
    t: float
