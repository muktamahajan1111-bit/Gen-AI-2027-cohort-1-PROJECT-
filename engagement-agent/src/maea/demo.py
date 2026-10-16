"""Synthetic classroom generator. It feeds the *real* pipeline with believable
signals so the whole system (and dashboard) can be explored with no camera."""
from __future__ import annotations

import numpy as np

from maea.config import Settings
from maea.core.models import AudioFrame, Observation
from maea.pipeline import Pipeline, WindowResult
from maea.storage.db import Database

# (minute, base engagement): warm-up, long lecture, a sharp dip, a poll, then discussion
_PROFILE = [(0, .78), (5, .80), (15, .66), (21, .60), (23, .36), (30.5, .34),
            (32.5, .62), (34, .84), (40, .86), (42, .66), (45, .55)]
# (start minute, mode) for the audio track
_AUDIO = [(0, "lecture"), (23, "silence"), (28, "lecture"), (31, "discussion"), (40, "lecture")]


def _audio_mode(minute: float) -> str:
    mode = _AUDIO[0][1]
    for start, m in _AUDIO:
        if minute >= start:
            mode = m
    return mode


def _audio_frames(rng, minutes: float, step: float = 0.1) -> list[AudioFrame]:
    frames, t, end = [], 0.0, minutes * 60
    while t < end:
        mode = _audio_mode(t / 60)
        if mode == "silence":
            seg, pause = rng.uniform(0.5, 1.5), rng.uniform(15, 30)
        elif mode == "lecture":
            seg, pause = rng.uniform(15, 40), rng.uniform(0.4, 1.8)
        else:
            seg, pause = rng.uniform(2, 6), rng.uniform(2.2, 5.0)
        for kind, length in (("speech", seg), ("pause", pause)):
            n = int(length / step)
            for i in range(n):
                tt = t + i * step
                if tt >= end:
                    break
                sp = kind == "speech"
                frames.append(AudioFrame(tt, rng.normal(-24 if sp else -52, 3), sp))
            t += length
    return frames


def run_demo(settings: Settings, db: Database, students: int = 18, minutes: float = 45,
             seed: int = 7, fps: float = 2.0, on_window=None) -> int:
    rng = np.random.default_rng(seed)
    pipe = Pipeline(settings, db, f"Demo lecture ({students} students)", source="synthetic",
                    consent=True, on_window=on_window)
    pipe.on_audio(_audio_frames(rng, minutes))

    profile_t = np.array([p[0] for p in _PROFILE]) * 60
    profile_v = np.array([p[1] for p in _PROFILE])
    arrive = rng.uniform(0, 60, students)
    arrive[-1], arrive[-2] = 11 * 60, 14 * 60                    # two late arrivals
    leave = {3: (20 * 60, 26 * 60)}                               # one student steps out
    offset = rng.normal(0, 0.08, students)
    still_until = np.zeros(students)
    still = np.zeros(students, dtype=bool)

    dt, t = 1.0 / fps, 0.0
    while t < minutes * 60:
        base = float(np.interp(t, profile_t, profile_v))
        obs = []
        for i in range(students):
            gone = i in leave and leave[i][0] <= t < leave[i][1]
            if t < arrive[i] or gone:
                continue
            lvl = min(1.0, max(0.05, base + offset[i]))
            if t >= still_until[i]:
                still[i] = rng.random() < 0.10 + 0.75 * (1 - lvl)
                still_until[i] = t + (rng.uniform(8, 40) if still[i] else rng.uniform(3, 15))
            attentive = rng.random() < lvl
            yaw = rng.normal(0, 8) if attentive else rng.choice([-1, 1]) * rng.normal(40, 8)
            pitch = rng.normal(0, 7) if attentive else rng.normal(28, 6)
            motion = abs(rng.normal(0.002, 0.001)) if still[i] else abs(rng.normal(0.03, 0.012))
            obs.append(Observation(i + 1, t, float(motion), float(yaw), float(pitch)))
        if 12 * 60 <= t < 12 * 60 + 5:                            # a 5-second flicker (false track)
            obs.append(Observation(101, t, 0.01, 0.0, 0.0))
        pipe.on_frame(t, obs)
        t += dt
    return pipe.finish(minutes * 60)
