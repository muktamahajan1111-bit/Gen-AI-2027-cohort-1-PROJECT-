"""CSV and self-contained HTML exports (no internet, no extra dependencies)."""
from __future__ import annotations

import csv
import html
from pathlib import Path

from maea.agent.advisor import Advisor, fmt_time
from maea.config import AgentConfig
from maea.storage.db import Database


def session_facts(db: Database, sid: int) -> dict:
    cw = db.table("class_windows", sid, "t_start")
    ev = db.table("trend_events", sid)
    summ = db.table("attendance_summary", sid, "student_id")
    facts = {"present": len(summ), "avg_engagement": 0.0, "drops": 0, "peaks": 0}
    if cw:
        facts["avg_engagement"] = sum(w["score"] for w in cw) / len(cw)
        facts["lowest"] = min(cw, key=lambda w: w["smoothed"])["t_end"]
        facts["highest"] = max(cw, key=lambda w: w["smoothed"])["t_end"]
    facts["drops"] = sum(e["kind"] == "drop" for e in ev)
    facts["peaks"] = sum(e["kind"] == "peak" for e in ev)
    return facts


def export_csv(db: Database, sid: int, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for table in ("attendance_summary", "attendance_events", "class_windows",
                  "student_windows", "trend_events", "suggestions"):
        rows = db.table(table, sid)
        if not rows:
            continue
        p = out / f"session{sid}_{table}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        paths.append(p)
    return paths


def _svg_line(values: list[float], width=720, height=180) -> str:
    if len(values) < 2:
        return ""
    step = (width - 40) / (len(values) - 1)
    pts = " ".join(f"{20 + i * step:.1f},{height - 20 - v * (height - 40):.1f}" for i, v in enumerate(values))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%">'
            f'<line x1="20" y1="{height-20}" x2="{width-20}" y2="{height-20}" stroke="#cbd5e1"/>'
            f'<polyline points="{pts}" fill="none" stroke="#4f46e5" stroke-width="3" '
            f'stroke-linejoin="round"/></svg>')


def build_html_report(db: Database, sid: int, agent_cfg: AgentConfig | None = None) -> str:
    sess = db.session(sid)
    if sess is None:
        raise ValueError(f"No such session: {sid}")
    facts = session_facts(db, sid)
    sugg = db.table("suggestions", sid)
    cw = db.table("class_windows", sid, "t_start")
    summ = db.table("attendance_summary", sid, "student_id")
    narrative = Advisor(agent_cfg or AgentConfig()).summarize(facts, sugg)
    rows = "".join(
        f"<tr><td>{html.escape(r['student_id'])}</td><td>{fmt_time(r['arrival_t'])}</td>"
        f"<td>{r['present_sec']/60:.1f} min</td><td>{r['attendance_pct']:.0f}%</td>"
        f"<td>{r['status']}</td></tr>" for r in summ)
    cards = "".join(f"<li><b>{html.escape(s['title'])}</b> <small>(minute {int(s['t']//60)})</small>"
                    f"<br>{html.escape(s['detail'])}</li>" for s in sugg) or "<li>No suggestions.</li>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Session report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:820px;margin:32px auto;color:#1e293b;padding:0 16px}}
h1{{margin-bottom:0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 10px;border-bottom:1px solid #e2e8f0;text-align:left}}
.kpi{{display:flex;gap:12px}}.kpi div{{flex:1;background:#f1f5f9;border-radius:10px;padding:12px}}
.kpi b{{font-size:1.6em;display:block}}li{{margin-bottom:10px}}</style></head><body>
<h1>{html.escape(sess['name'])}</h1><p>Duration {fmt_time(sess['duration_sec'])} &middot; generated locally</p>
<div class="kpi"><div><b>{facts['present']}</b>students present</div>
<div><b>{facts['avg_engagement']:.0%}</b>avg engagement</div>
<div><b>{facts['drops']}</b>attention drops</div><div><b>{facts['peaks']}</b>interaction peaks</div></div>
<h2>Summary</h2><p>{html.escape(narrative)}</p>
<h2>Engagement over time</h2>{_svg_line([w['smoothed'] for w in cw])}
<h2>Teaching suggestions</h2><ul>{cards}</ul>
<h2>Attendance</h2><table><tr><th>Student</th><th>Arrived</th><th>Time present</th><th>Attendance</th><th>Status</th></tr>{rows}</table>
</body></html>"""
