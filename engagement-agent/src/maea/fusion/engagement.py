"""Multimodal fusion: per-student visual signals + class-level audio context
-> engagement scores."""
from __future__ import annotations

from collections import defaultdict
from statistics import median

from maea.config import EngagementConfig
from maea.core.models import (AudioWindow, ClassWindow, Observation,
                              StudentWindow, student_label)

_MIN_SAMPLES = 3


class EngagementAggregator:
    """Buffers observations; ``flush`` turns one window into StudentWindows."""

    def __init__(self, cfg: EngagementConfig):
        self.cfg = cfg
        self._buf: dict[int, list[Observation]] = defaultdict(list)

    def add(self, obs: Observation) -> None:
        self._buf[obs.track_id].append(obs)

    def flush(self, t_start: float, t_end: float, allowed: set[int]) -> list[StudentWindow]:
        out = []
        for tid, obs in self._buf.items():
            if tid in allowed and len(obs) >= _MIN_SAMPLES:
                out.append(self._student(tid, obs, t_start, t_end))
        self._buf.clear()
        return sorted(out, key=lambda s: s.student_id)

    # -- per-student signals -------------------------------------------------
    def _student(self, tid: int, obs: list[Observation], t0: float, t1: float) -> StudentWindow:
        c = self.cfg
        obs = sorted(obs, key=lambda o: o.t)
        attn = sum(self._attentive(o) for o in obs) / len(obs)
        movement = min(1.0, (sum(o.motion for o in obs) / len(obs)) / c.motion_target)
        inactivity = self._inactivity(obs)
        score = (c.w_attention * attn + c.w_movement * movement
                 + c.w_activity * (1.0 - inactivity))
        return StudentWindow(student_label(tid), t0, t1, round(attn, 4), round(movement, 4),
                             round(inactivity, 4), round(min(1.0, max(0.0, score)), 4))

    def _attentive(self, o: Observation) -> bool:
        if o.yaw is None or o.pitch is None:
            return o.frontal
        return abs(o.yaw) < self.cfg.yaw_limit and abs(o.pitch) < self.cfg.pitch_limit

    def _inactivity(self, obs: list[Observation]) -> float:
        """Share of the window spent in still stretches longer than ``inactive_min_sec``."""
        c = self.cfg
        if len(obs) < 2:
            return 0.0
        dt = median(b.t - a.t for a, b in zip(obs, obs[1:])) or 1.0
        total, run_start, run_end = 0.0, None, None
        for o in obs + [None]:  # sentinel closes the last run
            still = o is not None and o.motion < c.motion_eps
            if still:
                run_start = o.t if run_start is None else run_start
                run_end = o.t
            elif run_start is not None:
                length = run_end - run_start + dt
                if length >= c.inactive_min_sec:
                    total += length
                run_start = run_end = None
        return min(1.0, total / (len(obs) * dt))


class ClassFuser:
    """Combines student scores with audio context and smooths over time."""

    def __init__(self, cfg: EngagementConfig):
        self.cfg = cfg
        self._ema: float | None = None

    def fuse(self, students: list[StudentWindow], audio: AudioWindow | None,
             t0: float, t1: float) -> ClassWindow | None:
        if not students:
            return None
        n = len(students)
        score = sum(s.score for s in students) / n
        attn = sum(s.attention for s in students) / n
        inact = sum(s.inactivity for s in students) / n
        mode = audio.mode if audio else "unknown"
        if audio:
            if mode == "discussion":
                score += 0.05            # interaction bonus
            elif mode == "silence" and attn < 0.5:
                score -= 0.08            # quiet *and* looking away
        score = min(1.0, max(0.0, score))
        a = self.cfg.ema_alpha
        self._ema = score if self._ema is None else a * score + (1 - a) * self._ema
        return ClassWindow(t0, t1, round(score, 4), round(self._ema, 4), n, round(attn, 4),
                           round(inact, 4), mode, round(audio.speech_ratio, 4) if audio else 0.0,
                           audio.turns if audio else 0)
