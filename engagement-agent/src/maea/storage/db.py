"""SQLite persistence. Local file only; no raw frames or audio are ever stored."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from maea.core.models import (AttendanceEvent, ClassWindow, StudentWindow,
                              Suggestion, TrendEvent)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, started_at REAL, duration_sec REAL DEFAULT 0,
  source TEXT, consent INTEGER, window_sec REAL, rejected_tracks INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS attendance_events(
  session_id INTEGER, student_id TEXT, kind TEXT, t REAL);
CREATE TABLE IF NOT EXISTS attendance_summary(
  session_id INTEGER, student_id TEXT, arrival_t REAL, present_sec REAL, attendance_pct REAL, status TEXT);
CREATE TABLE IF NOT EXISTS student_windows(
  session_id INTEGER, student_id TEXT, t_start REAL, t_end REAL,
  attention REAL, movement REAL, inactivity REAL, score REAL);
CREATE TABLE IF NOT EXISTS class_windows(
  session_id INTEGER, t_start REAL, t_end REAL, score REAL, smoothed REAL, present INTEGER,
  attention REAL, inactivity REAL, audio_mode TEXT, speech_ratio REAL, turns INTEGER);
CREATE TABLE IF NOT EXISTS trend_events(session_id INTEGER, kind TEXT, t REAL, magnitude REAL);
CREATE TABLE IF NOT EXISTS suggestions(
  session_id INTEGER, t REAL, severity TEXT, rule TEXT, title TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_sw ON student_windows(session_id);
CREATE INDEX IF NOT EXISTS ix_cw ON class_windows(session_id);
"""

_CHILD_TABLES = ("attendance_events", "attendance_summary", "student_windows",
                 "class_windows", "trend_events", "suggestions")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self):
        with self._conn:
            yield self._conn

    # -- writes ----------------------------------------------------------------
    def create_session(self, name: str, source: str, consent: bool, window_sec: float) -> int:
        with self._tx() as c:
            cur = c.execute("INSERT INTO sessions(name, started_at, source, consent, window_sec) "
                            "VALUES(?,?,?,?,?)", (name, time.time(), source, int(consent), window_sec))
            return int(cur.lastrowid)

    def add_attendance_event(self, sid: int, e: AttendanceEvent) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO attendance_events VALUES(?,?,?,?)", (sid, e.student_id, e.kind, e.t))

    def add_windows(self, sid: int, cw: ClassWindow, students: list[StudentWindow]) -> None:
        with self._tx() as c:
            d = asdict(cw)
            c.execute("INSERT INTO class_windows VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (sid, d["t_start"], d["t_end"], d["score"], d["smoothed"], d["present"],
                       d["attention"], d["inactivity"], d["audio_mode"], d["speech_ratio"], d["turns"]))
            c.executemany("INSERT INTO student_windows VALUES(?,?,?,?,?,?,?,?)",
                          [(sid, s.student_id, s.t_start, s.t_end, s.attention, s.movement,
                            s.inactivity, s.score) for s in students])

    def add_trend_event(self, sid: int, e: TrendEvent) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO trend_events VALUES(?,?,?,?)", (sid, e.kind, e.t, e.magnitude))

    def add_suggestion(self, sid: int, s: Suggestion) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO suggestions VALUES(?,?,?,?,?,?)",
                      (sid, s.t, s.severity, s.rule, s.title, s.detail))

    def finalize_session(self, sid: int, duration: float, rejected: int,
                         summary: list[tuple[str, float, float, float, str]]) -> None:
        with self._tx() as c:
            c.execute("UPDATE sessions SET duration_sec=?, rejected_tracks=? WHERE id=?",
                      (duration, rejected, sid))
            c.executemany("INSERT INTO attendance_summary VALUES(?,?,?,?,?,?)",
                          [(sid, *row[:3], row[3], row[4]) for row in summary])

    # -- reads -----------------------------------------------------------------
    def _all(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def sessions(self) -> list[dict]:
        return self._all("SELECT * FROM sessions ORDER BY id DESC")

    def session(self, sid: int) -> dict | None:
        rows = self._all("SELECT * FROM sessions WHERE id=?", sid)
        return rows[0] if rows else None

    def table(self, name: str, sid: int, order: str = "t") -> list[dict]:
        if name not in _CHILD_TABLES:
            raise ValueError(name)
        cols = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({name})")}
        order_sql = f" ORDER BY {order}" if order in cols else ""
        return self._all(f"SELECT * FROM {name} WHERE session_id=?{order_sql}", sid)

    # -- privacy ---------------------------------------------------------------
    def delete_session(self, sid: int) -> None:
        with self._tx() as c:
            for t in _CHILD_TABLES:
                c.execute(f"DELETE FROM {t} WHERE session_id=?", (sid,))
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))

    def purge_older_than(self, days: float) -> int:
        cutoff = time.time() - days * 86400
        old = [r["id"] for r in self._all("SELECT id FROM sessions WHERE started_at<?", cutoff)]
        for sid in old:
            self.delete_session(sid)
        return len(old)
