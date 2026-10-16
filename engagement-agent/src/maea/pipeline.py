"""The orchestrator. It knows *nothing* about cameras or microphones: it consumes
Observations and AudioFrames, so live capture, recorded files and the synthetic
demo all drive exactly the same logic."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from maea.agent.rules import RuleEngine
from maea.attendance.presence import PresenceTracker
from maea.audio.aggregator import AudioAggregator
from maea.config import Settings
from maea.core.models import (AttendanceEvent, AudioFrame, ClassWindow, Observation,
                              StudentWindow, Suggestion, TrendEvent, student_label)
from maea.fusion.engagement import ClassFuser, EngagementAggregator
from maea.fusion.trends import TrendDetector
from maea.storage.db import Database


@dataclass
class WindowResult:
    window: ClassWindow
    students: list[StudentWindow]
    event: TrendEvent | None
    suggestions: list[Suggestion] = field(default_factory=list)


class Pipeline:
    def __init__(self, settings: Settings, db: Database, name: str, source: str = "live",
                 consent: bool = True, on_window: Callable[[WindowResult], None] | None = None):
        self.s, self.db, self.on_window = settings, db, on_window
        self.win_sec = settings.engagement.window_sec
        self.session_id = db.create_session(name, source, consent, self.win_sec)
        self._presence: dict[int, PresenceTracker] = {}
        self._agg = EngagementAggregator(settings.engagement)
        self._audio = AudioAggregator(settings.audio)
        self._fuser = ClassFuser(settings.engagement)
        self._trend = TrendDetector(settings.engagement)
        self._rules = RuleEngine(settings.agent, self.win_sec)
        self._windows: list[ClassWindow] = []
        self._events: list[TrendEvent] = []
        self._win_start, self._t = 0.0, 0.0

    # -- inputs ----------------------------------------------------------------
    def on_audio(self, frames: list[AudioFrame]) -> None:
        self._audio.add(frames)

    def on_frame(self, t: float, observations: list[Observation]) -> None:
        self._t = t
        seen = {o.track_id for o in observations}
        for o in observations:
            self._presence.setdefault(o.track_id, PresenceTracker(self.s.presence))
            self._agg.add(o)
        for tid, tracker in self._presence.items():
            for kind, et in tracker.update(tid in seen, t):
                self.db.add_attendance_event(self.session_id,
                                             AttendanceEvent(student_label(tid), kind, et))
        while t >= self._win_start + self.win_sec:
            self._flush(self._win_start, self._win_start + self.win_sec)
            self._win_start += self.win_sec

    # -- window / session end ----------------------------------------------------
    def _flush(self, t0: float, t1: float) -> None:
        marked = {tid for tid, p in self._presence.items() if p.marked}
        students = self._agg.flush(t0, t1, marked)
        audio = self._audio.summarize(t0, t1)
        cw = self._fuser.fuse(students, audio, t0, t1)
        if cw is None:
            return
        self._windows.append(cw)
        event = self._trend.update(cw)
        if event:
            self._events.append(event)
            self.db.add_trend_event(self.session_id, event)
        self.db.add_windows(self.session_id, cw, students)
        sugg = self._rules.evaluate(self._windows, self._events)
        for sg in sugg:
            self.db.add_suggestion(self.session_id, sg)
        if self.on_window:
            self.on_window(WindowResult(cw, students, event, sugg))

    def finish(self, t_end: float | None = None) -> int:
        t_end = self._t if t_end is None else t_end
        if t_end - self._win_start >= 15:           # keep a meaningful partial window
            self._flush(self._win_start, t_end)
        late_after = self.s.presence.late_after_min * 60
        summary, rejected = [], 0
        for tid, p in self._presence.items():
            if not p.marked:
                rejected += 1                       # flickers that never became "sustained"
                continue
            pct = min(100.0, 100.0 * p.total_present_sec / t_end) if t_end else 0.0
            status = "Late" if (p.first_seen or 0) > late_after else "Present"
            summary.append((student_label(tid), p.first_seen or 0.0, p.total_present_sec,
                            round(pct, 1), status))
        self.db.finalize_session(self.session_id, t_end, rejected, sorted(summary))
        return self.session_id
