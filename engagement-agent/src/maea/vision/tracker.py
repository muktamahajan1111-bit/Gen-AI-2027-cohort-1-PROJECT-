"""Lightweight IoU tracker that gives each face a stable ID across frames."""
from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]  # x, y, w, h


def iou(a: BBox, b: BBox) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    id: int
    bbox: BBox
    last_t: float


class IoUTracker:
    def __init__(self, max_missing_sec: float = 3.0, iou_threshold: float = 0.15):
        self.max_missing_sec = max_missing_sec
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: list[BBox], t: float) -> list[int]:
        """Returns one track id per input box (same order)."""
        pairs = sorted(
            ((iou(tr.bbox, b), ti, bi)
             for ti, tr in enumerate(self._tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t: set[int] = set()
        ids: dict[int, int] = {}
        for score, ti, bi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or bi in ids:
                continue
            used_t.add(ti)
            tr = self._tracks[ti]
            tr.bbox, tr.last_t = boxes[bi], t
            ids[bi] = tr.id
        for bi, b in enumerate(boxes):
            if bi not in ids:
                self._tracks.append(_Track(self._next_id, b, t))
                ids[bi] = self._next_id
                self._next_id += 1
        self._tracks = [tr for tr in self._tracks if t - tr.last_t <= self.max_missing_sec]
        return [ids[i] for i in range(len(boxes))]
