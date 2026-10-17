"""Turns a stream of VAD frames into per-window audio context."""
from __future__ import annotations

from maea.config import AudioConfig
from maea.core.models import AudioFrame, AudioWindow


class AudioAggregator:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._frames: list[AudioFrame] = []
        self._last_speech_t: float | None = None

    def add(self, frames: list[AudioFrame]) -> None:
        self._frames.extend(frames)

    def summarize(self, t_start: float, t_end: float) -> AudioWindow | None:
        win = [f for f in self._frames if t_start <= f.t < t_end]
        self._frames = [f for f in self._frames if f.t >= t_end]
        if not win:
            return None
        speech = [f for f in win if f.is_speech]
        ratio = len(speech) / len(win)

        # An "interaction burst" starts when speech resumes after a long pause.
        turns, last = 0, self._last_speech_t
        for f in speech:
            if last is None or f.t - last >= self.cfg.pause_turn_sec:
                turns += 1
            last = f.t
        self._last_speech_t = last

        mean_db = sum(f.rms_db for f in speech) / len(speech) if speech else -60.0
        energy = min(1.0, max(0.0, (mean_db + 60.0) / 60.0))
        minutes = max((t_end - t_start) / 60.0, 1e-6)
        return AudioWindow(t_start, t_end, ratio, energy, turns,
                           self._mode(ratio, turns / minutes))

    def _mode(self, ratio: float, turns_per_min: float) -> str:
        if ratio < 0.10:
            return "silence"
        if ratio >= 0.20 and turns_per_min >= self.cfg.discussion_turns_per_min:
            return "discussion"
        if ratio >= 0.50:
            return "lecture"
        return "mixed"
