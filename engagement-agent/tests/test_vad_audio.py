import numpy as np

from maea.audio.aggregator import AudioAggregator
from maea.audio.vad import EnergyVAD
from maea.config import AudioConfig
from maea.core.models import AudioFrame


def test_vad_separates_speech_from_silence():
    sr = 16000
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.0005, sr * 2)
    loud = rng.normal(0, 0.1, sr * 2)
    frames = EnergyVAD(AudioConfig(), sr).process(np.concatenate([quiet, loud, quiet]), 0.0)
    flags = np.array([f.is_speech for f in frames])
    third = len(flags) // 3
    assert flags[:third - 10].mean() < 0.05
    assert flags[third + 10:2 * third - 10].mean() > 0.95


def _frames(pattern):  # list of (is_speech, seconds)
    out, t = [], 0.0
    for sp, sec in pattern:
        for _ in range(int(sec * 10)):
            out.append(AudioFrame(t, -25 if sp else -55, sp))
            t += 0.1
    return out


def test_modes():
    cfg = AudioConfig()
    lecture = AudioAggregator(cfg); lecture.add(_frames([(True, 25), (False, 1), (True, 30), (False, 4)]))
    assert lecture.summarize(0, 60).mode == "lecture"
    talk = AudioAggregator(cfg); talk.add(_frames([(True, 4), (False, 4)] * 7 + [(False, 4)]))
    assert talk.summarize(0, 60).mode == "discussion"
    quiet = AudioAggregator(cfg); quiet.add(_frames([(False, 60)]))
    assert quiet.summarize(0, 60).mode == "silence"
