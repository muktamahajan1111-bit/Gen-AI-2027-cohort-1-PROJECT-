"""Energy-based voice activity detection with an adaptive noise floor.
Pure numpy, so it runs anywhere and fully offline."""
from __future__ import annotations

import numpy as np

from maea.config import AudioConfig
from maea.core.models import AudioFrame

_ABS_MIN_DB = -45.0     # never call anything quieter than this "speech"
_HANGOVER_FRAMES = 5    # keep speech on briefly after the level drops


class EnergyVAD:
    def __init__(self, cfg: AudioConfig, sample_rate: int | None = None):
        self.cfg = cfg
        self.sr = sample_rate or cfg.sample_rate
        self._n = int(self.sr * cfg.frame_ms / 1000)
        self._buf = np.zeros(0, dtype=np.float32)
        self._floor = -50.0
        self._hang = 0
        self._t = 0.0
        self._started = False

    def process(self, samples: np.ndarray, t0: float) -> list[AudioFrame]:
        if not self._started:
            self._t, self._started = t0, True
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        frames: list[AudioFrame] = []
        while len(self._buf) >= self._n:
            chunk, self._buf = self._buf[:self._n], self._buf[self._n:]
            db = 20 * np.log10(float(np.sqrt(np.mean(chunk ** 2))) + 1e-9)
            # noise floor: drops instantly, rises slowly
            self._floor = db if db < self._floor else self._floor + 0.002 * (db - self._floor)
            loud = db > max(self._floor + self.cfg.margin_db, _ABS_MIN_DB)
            self._hang = _HANGOVER_FRAMES if loud else max(0, self._hang - 1)
            frames.append(AudioFrame(self._t, db, loud or self._hang > 0))
            self._t += self.cfg.frame_ms / 1000
        return frames
