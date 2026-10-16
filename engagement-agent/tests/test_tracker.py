from maea.vision.tracker import IoUTracker


def test_ids_are_stable_and_unique():
    tr = IoUTracker()
    a = tr.update([(0, 0, 50, 50), (200, 0, 50, 50)], 0.0)
    b = tr.update([(3, 2, 50, 50), (202, 1, 50, 50)], 0.5)
    assert a == b and len(set(a)) == 2


def test_track_expires_after_missing_window():
    tr = IoUTracker(max_missing_sec=1.0)
    first = tr.update([(0, 0, 50, 50)], 0.0)
    tr.update([], 5.0)
    assert tr.update([(0, 0, 50, 50)], 5.5) != first
