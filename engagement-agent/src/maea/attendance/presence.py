"""Sustained-presence logic: a person is marked present only after being
consistently visible, never from a single frame."""
from __future__ import annotations

from collections import deque

from maea.config import PresenceConfig


class PresenceTracker:
    """One instance per tracked person. Feed it a yes/no every frame."""

    def __init__(self, cfg: PresenceConfig):
        self.cfg = cfg
        self._samples: deque[tuple[float, bool]] = deque()
        self._start: float | None = None
        self.first_seen: float | None = None
        self.last_seen: float | None = None
        self.marked = False
        self.away = False
        self.total_present_sec = 0.0

    def update(self, seen: bool, t: float) -> list[tuple[str, float]]:
        """Returns zero or more (event_kind, event_time) tuples."""
        events: list[tuple[str, float]] = []
        c = self.cfg
        if self._start is None:
            self._start = t
        self._samples.append((t, seen))
        while self._samples and t - self._samples[0][0] > c.window_sec:
            self._samples.popleft()

        if seen:
            if self.first_seen is None:
                self.first_seen = t
            if self.last_seen is not None and t - self.last_seen <= c.gap_sec:
                self.total_present_sec += t - self.last_seen
            self.last_seen = t
            if self.marked and self.away:
                self.away = False
                events.append(("RETURNED", t))

        if not self.marked and self.first_seen is not None:
            ratio = sum(s for _, s in self._samples) / len(self._samples)
            if t - self._start >= c.confirm_sec and ratio >= c.min_ratio:
                self.marked = True
                events.append(("PRESENT", self.first_seen))  # log true arrival time

        if (self.marked and not self.away and not seen
                and self.last_seen is not None and t - self.last_seen > c.gap_sec):
            self.away = True
            events.append(("LEFT", self.last_seen))
        return events
