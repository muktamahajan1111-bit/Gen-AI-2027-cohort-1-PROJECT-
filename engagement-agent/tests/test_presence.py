from maea.attendance.presence import PresenceTracker
from maea.config import PresenceConfig


def feed(tr, pattern, fps=2):
    ev = []
    for i, seen in enumerate(pattern):
        ev += tr.update(seen, i / fps)
    return ev


def test_single_frame_is_not_attendance():
    tr = PresenceTracker(PresenceConfig())
    assert feed(tr, [True] + [False] * 120) == []
    assert not tr.marked


def test_sustained_presence_is_marked_with_true_arrival_time():
    tr = PresenceTracker(PresenceConfig(confirm_sec=30))
    ev = feed(tr, [True] * 100)
    assert ev == [("PRESENT", 0.0)]


def test_leave_and_return_are_logged():
    tr = PresenceTracker(PresenceConfig(confirm_sec=10, gap_sec=5))
    kinds = [k for k, _ in feed(tr, [True] * 40 + [False] * 40 + [True] * 10)]
    assert kinds == ["PRESENT", "LEFT", "RETURNED"]
    assert tr.total_present_sec > 0
