"""Streaming detection of attention drops and interaction peaks."""
from __future__ import annotations

from maea.config import EngagementConfig
from maea.core.models import ClassWindow, TrendEvent


class TrendDetector:
    def __init__(self, cfg: EngagementConfig):
        self.cfg = cfg
        self._hist: list[float] = []
        self._state = "normal"

    def update(self, w: ClassWindow) -> TrendEvent | None:
        c, event = self.cfg, None
        if len(self._hist) >= 3:
            base = sum(self._hist[-c.baseline_windows:]) / len(self._hist[-c.baseline_windows:])
            change = (w.smoothed - base) / base if base else 0.0
            if change <= -c.drop_pct:
                if self._state != "drop":
                    event = TrendEvent("drop", w.t_end, round(change, 3))
                self._state = "drop"
            elif change >= c.peak_pct:
                if self._state != "peak":
                    event = TrendEvent("peak", w.t_end, round(change, 3))
                self._state = "peak"
            else:
                self._state = "normal"
        self._hist.append(w.smoothed)
        return event
