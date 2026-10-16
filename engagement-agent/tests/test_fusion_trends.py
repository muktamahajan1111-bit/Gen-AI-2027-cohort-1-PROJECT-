from maea.config import EngagementConfig
from maea.core.models import AudioWindow, ClassWindow, Observation
from maea.fusion.engagement import ClassFuser, EngagementAggregator
from maea.fusion.trends import TrendDetector

CFG = EngagementConfig()


def obs(tid, n, motion, yaw=0.0, pitch=0.0, dt=0.5):
    return [Observation(tid, i * dt, motion, yaw, pitch) for i in range(n)]


def score_for(**kw):
    agg = EngagementAggregator(CFG)
    for o in obs(1, 120, **kw):
        agg.add(o)
    return agg.flush(0, 60, {1})[0]


def test_attentive_and_moving_scores_high():
    assert score_for(motion=0.03).score > 0.9


def test_looking_away_and_still_scores_low():
    s = score_for(motion=0.001, yaw=50.0, pitch=30.0)
    assert s.attention == 0 and s.inactivity > 0.9 and s.score < 0.1


def test_unconfirmed_tracks_are_excluded():
    agg = EngagementAggregator(CFG)
    for o in obs(7, 50, 0.02):
        agg.add(o)
    assert agg.flush(0, 60, allowed=set()) == []


def _cw(i, v):
    return ClassWindow(i * 60, (i + 1) * 60, v, v, 10, v, 0.1, "lecture", 0.6, 1)


def test_drop_and_peak_detection():
    td = TrendDetector(CFG)
    events = [td.update(_cw(i, v)) for i, v in enumerate([.7] * 6 + [.4, .4, .4, .85])]
    kinds = [e.kind for e in events if e]
    assert kinds == ["drop", "peak"]


def test_audio_context_adjusts_class_score():
    from maea.core.models import StudentWindow
    s = [StudentWindow("Student_01", 0, 60, .7, .5, .1, .6)]
    base = ClassFuser(CFG).fuse(s, None, 0, 60).score
    talk = ClassFuser(CFG).fuse(s, AudioWindow(0, 60, .5, .5, 8, "discussion"), 0, 60).score
    assert talk > base
