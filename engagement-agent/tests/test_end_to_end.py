from maea.config import Settings
from maea.demo import run_demo
from maea.reports import build_html_report, export_csv
from maea.storage.db import Database


def test_demo_runs_through_real_pipeline(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    sid = run_demo(Settings(), db, students=10, minutes=30, fps=1.0)
    summary = db.table("attendance_summary", sid, "student_id")
    assert len(summary) == 10
    assert db.session(sid)["rejected_tracks"] == 1          # the 5-second flicker
    assert any(r["status"] == "Late" for r in summary)
    assert db.table("class_windows", sid) and db.table("suggestions", sid)
    assert "Demo lecture" in build_html_report(db, sid)
    assert export_csv(db, sid, tmp_path / "out")


def test_privacy_purge_and_delete(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    sid = run_demo(Settings(), db, students=5, minutes=5, fps=1.0)
    assert db.purge_older_than(30) == 0
    db.delete_session(sid)
    assert db.session(sid) is None and db.table("class_windows", sid) == []
