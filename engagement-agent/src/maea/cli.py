"""Command-line entry point:  python -m maea <command>"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from maea.config import load_settings
from maea.storage.db import Database


def _db(settings) -> Database:
    return Database(settings.storage.db_path)


def cmd_demo(a, settings) -> None:
    from maea.agent.advisor import fmt_time
    from maea.demo import run_demo
    db = _db(settings)
    print(f"Simulating a {a.minutes:.0f}-minute class with {a.students} students...")
    sid = run_demo(settings, db, a.students, a.minutes, a.seed)
    for s in db.table("suggestions", sid):
        print(f"  [{fmt_time(s['t'])}] {s['title']}")
    print(f"\nDone. Session #{sid} saved to {settings.storage.db_path}")
    print("Open the dashboard with:  python -m maea dashboard")


def cmd_run(a, settings) -> None:
    from maea.runner import run_live
    if settings.privacy.consent_required and not a.consent:
        sys.exit("Consent is required. Confirm participants have been informed, then re-run with --consent.")
    sid = run_live(settings, _db(settings), a.video, a.audio, a.mic, a.name, a.show, a.max_minutes)
    print(f"Session #{sid} saved. View it with:  python -m maea dashboard")


def cmd_dashboard(a, settings) -> None:
    app = Path(__file__).parent / "dashboard" / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                              "--server.headless", "true" if a.headless else "false",
                              "--browser.gatherUsageStats", "false"]))


def cmd_sessions(a, settings) -> None:
    for s in _db(settings).sessions():
        print(f"#{s['id']:<3} {s['name']:<40} {s['duration_sec']/60:5.1f} min  source={s['source']}")


def cmd_report(a, settings) -> None:
    from maea.reports import build_html_report, export_csv
    db = _db(settings)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"session{a.session}_report.html").write_text(build_html_report(db, a.session, settings.agent))
    export_csv(db, a.session, out)
    print(f"Report and CSV files written to {out}/")


def cmd_purge(a, settings) -> None:
    days = a.days if a.days is not None else settings.privacy.retention_days
    print(f"Deleted {_db(settings).purge_older_than(days)} session(s) older than {days} days.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="maea", description="Multimodal Attendance & Engagement Agent")
    p.add_argument("--config", help="path to a TOML config (default: configs/default.toml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="simulate a class session (no camera needed)")
    d.add_argument("--students", type=int, default=18)
    d.add_argument("--minutes", type=float, default=45)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="analyse a live camera or recorded video")
    r.add_argument("--video", required=True, help="camera index (0) or video file")
    r.add_argument("--audio", help="WAV file to analyse alongside the video")
    r.add_argument("--mic", action="store_true", help="use the live microphone")
    r.add_argument("--name", default="Classroom session")
    r.add_argument("--consent", action="store_true", help="confirm participants are informed")
    r.add_argument("--show", action="store_true", help="preview window (frames are not saved)")
    r.add_argument("--max-minutes", type=float)
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("dashboard", help="launch the web dashboard")
    b.add_argument("--headless", action="store_true")
    b.set_defaults(fn=cmd_dashboard)

    sub.add_parser("sessions", help="list saved sessions").set_defaults(fn=cmd_sessions)

    rp = sub.add_parser("report", help="export HTML report + CSV files")
    rp.add_argument("--session", type=int, required=True)
    rp.add_argument("--out", default="data/exports")
    rp.set_defaults(fn=cmd_report)

    pg = sub.add_parser("purge", help="delete sessions older than the retention period")
    pg.add_argument("--days", type=float)
    pg.set_defaults(fn=cmd_purge)

    a = p.parse_args(argv)
    a.fn(a, load_settings(a.config))
