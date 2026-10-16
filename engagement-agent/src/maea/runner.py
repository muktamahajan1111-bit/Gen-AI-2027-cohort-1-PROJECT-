"""Glue for real inputs: camera/video (+ optional audio) -> Pipeline."""
from __future__ import annotations

from maea.agent.advisor import fmt_time
from maea.audio.vad import EnergyVAD
from maea.capture.audio_source import MicSource, WavSource
from maea.capture.video_source import iter_frames
from maea.config import Settings
from maea.core.models import Observation
from maea.pipeline import Pipeline, WindowResult
from maea.storage.db import Database
from maea.vision.detector import create_detector
from maea.vision.motion import MotionEstimator
from maea.vision.tracker import IoUTracker


def _print_window(r: WindowResult) -> None:
    w = r.window
    print(f"[{fmt_time(w.t_end)}] present={w.present:<3} engagement={w.smoothed:.0%} "
          f"audio={w.audio_mode}" + (f"  << {r.event.kind.upper()}" if r.event else ""))
    for s in r.suggestions:
        print(f"           suggestion: {s.title}")


def run_live(settings: Settings, db: Database, video: str, audio: str | None = None,
             use_mic: bool = False, name: str = "Live session", show: bool = False,
             max_minutes: float | None = None) -> int:
    detector = create_detector(settings.vision.detector)
    print(f"Detector: {detector.name}  |  press Ctrl+C{' or Q' if show else ''} to stop")
    tracker = IoUTracker(settings.vision.max_missing_sec)
    motion = MotionEstimator()
    pipe = Pipeline(settings, db, name, source=str(video), on_window=_print_window)

    audio_src = WavSource(audio) if audio else (MicSource(settings.audio.sample_rate) if use_mic else None)
    vad = EnergyVAD(settings.audio, getattr(audio_src, "sample_rate", None)) if audio_src else None
    cv2 = None
    if show:
        import cv2  # noqa: F811
    last_t = 0.0
    try:
        for t, frame in iter_frames(video, settings.vision.fps):
            if max_minutes and t > max_minutes * 60:
                break
            if audio_src:
                for ct, chunk in audio_src.poll(t):
                    pipe.on_audio(vad.process(chunk, ct))
            dets = detector.detect(frame)
            ids = tracker.update([d.bbox for d in dets], t)
            obs = [Observation(i, t, motion.update(i, d.bbox), d.yaw, d.pitch, d.frontal)
                   for i, d in zip(ids, dets)]
            pipe.on_frame(t, obs)
            last_t = t
            if show:
                for i, d in zip(ids, dets):
                    x, y, w, h = map(int, d.bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 120), 2)
                    cv2.putText(frame, f"Student_{i:02d}", (x, max(12, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1)
                cv2.imshow("MAEA (frames are never saved)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if show:
            cv2.destroyAllWindows()
        if hasattr(audio_src, "close"):
            audio_src.close()
    return pipe.finish(last_t)
