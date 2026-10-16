"""Streamlit dashboard. Reads only from the local SQLite database."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run without installing

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from maea.agent.advisor import Advisor, fmt_time
from maea.config import load_settings
from maea.demo import run_demo
from maea.reports import build_html_report, session_facts
from maea.storage.db import Database

INDIGO, SLATE, GREEN, AMBER, RED = "#4f46e5", "#64748b", "#16a34a", "#d97706", "#dc2626"
MODE_COLOR = {"lecture": "rgba(79,70,229,.07)", "discussion": "rgba(22,163,74,.14)",
              "silence": "rgba(100,116,139,.16)", "mixed": "rgba(217,119,6,.08)", "unknown": "rgba(0,0,0,0)"}

st.set_page_config(page_title="Engagement Agent", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.kpi{background:var(--secondary-background-color);border-radius:14px;padding:16px 18px;
     border:1px solid rgba(128,128,128,.18)}
.kpi .v{font-size:2rem;font-weight:700;line-height:1.1}
.kpi .l{opacity:.7;font-size:.85rem;margin-top:2px}
.sg{border-left:5px solid;border-radius:10px;padding:12px 16px;margin-bottom:10px;
    background:var(--secondary-background-color)}
.sg small{opacity:.65}
.badge{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.78rem;
       background:rgba(22,163,74,.15);color:#16a34a;margin-right:6px}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_settings():
    return load_settings(os.environ.get("MAEA_CONFIG"))


settings = get_settings()
db = Database(settings.storage.db_path)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 Engagement Agent")
    st.caption("Multimodal Attendance & Engagement")
    st.markdown('<span class="badge">Offline</span><span class="badge">Local storage</span>'
                '<span class="badge">No video stored</span>', unsafe_allow_html=True)
    st.divider()
    sessions = db.sessions()
    if st.button("▶ Generate demo session", width="stretch", type="primary"):
        with st.spinner("Simulating a 45-minute class..."):
            run_demo(settings, db)
        st.rerun()
    if not sessions:
        st.info("No sessions yet. Click **Generate demo session** to explore the dashboard.")
        st.stop()
    labels = {f"#{s['id']} · {s['name']}": s["id"] for s in sessions}
    sid = labels[st.selectbox("Session", list(labels))]

sess = db.session(sid)
cw = pd.DataFrame(db.table("class_windows", sid, "t_start"))
sw = pd.DataFrame(db.table("student_windows", sid, "t_start"))
att = pd.DataFrame(db.table("attendance_summary", sid, "student_id"))
events = pd.DataFrame(db.table("attendance_events", sid))
trend = pd.DataFrame(db.table("trend_events", sid))
sugg = db.table("suggestions", sid)
facts = session_facts(db, sid)

st.title(sess["name"])
st.caption(f"Duration {fmt_time(sess['duration_sec'])} · source: {sess['source']} · "
           f"{'consent recorded' if sess['consent'] else 'no consent flag'}")

c = st.columns(5)
n_late = int((att["status"] == "Late").sum()) if not att.empty else 0
kpi(c[0], facts["present"], "students marked present")
kpi(c[1], f"{att['attendance_pct'].mean():.0f}%" if not att.empty else "–", "average attendance")
kpi(c[2], f"{facts['avg_engagement']:.0%}", "average engagement")
kpi(c[3], facts["drops"], "attention drops")
kpi(c[4], facts["peaks"], "interaction peaks")
st.write("")

tab_over, tab_att, tab_eng, tab_sug, tab_priv = st.tabs(
    ["📈 Overview", "🧾 Attendance", "🔥 Engagement", "💡 Suggestions", "🔒 Privacy & Export"])

# ----------------------------------------------------------------------------- overview
with tab_over:
    if cw.empty:
        st.info("No engagement windows recorded for this session.")
    else:
        fig = go.Figure()
        for _, r in cw.iterrows():
            fig.add_vrect(x0=r.t_start / 60, x1=r.t_end / 60, fillcolor=MODE_COLOR[r.audio_mode],
                          line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=cw.t_end / 60, y=cw.score, name="Raw", mode="lines",
                                 line=dict(color="rgba(79,70,229,.3)", width=1.5)))
        fig.add_trace(go.Scatter(x=cw.t_end / 60, y=cw.smoothed, name="Smoothed", mode="lines",
                                 line=dict(color=INDIGO, width=4, shape="spline")))
        if not trend.empty:
            for kind, color, sym in (("drop", RED, "triangle-down"), ("peak", GREEN, "triangle-up")):
                e = trend[trend.kind == kind]
                y = [cw.loc[(cw.t_end - t).abs().idxmin(), "smoothed"] for t in e.t]
                fig.add_trace(go.Scatter(x=e.t / 60, y=y, mode="markers", name=kind.title(),
                                         marker=dict(color=color, size=14, symbol=sym,
                                                     line=dict(color="white", width=1))))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                          xaxis_title="Minutes into session", yaxis=dict(range=[0, 1], tickformat=".0%",
                          title="Engagement"), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("Background shading = audio context: indigo lecture · green discussion · grey silence.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Session summary")
        st.write(Advisor(settings.agent).summarize(facts, sugg))
    with right:
        st.subheader("Top suggestions")
        if not sugg:
            st.write("Nothing to flag.")
        for s in sugg[:3]:
            st.markdown(f"**{fmt_time(s['t'])}** · {s['title']}")

# ----------------------------------------------------------------------------- attendance
with tab_att:
    if att.empty:
        st.info("No attendance recorded.")
    else:
        view = pd.DataFrame({
            "Student": att.student_id, "Arrived": att.arrival_t.map(fmt_time),
            "Minutes present": (att.present_sec / 60).round(1),
            "Attendance": att.attendance_pct, "Status": att.status})
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "Attendance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
        st.caption(f"{sess['rejected_tracks']} brief appearance(s) were ignored because they were not "
                   "sustained (false-positive protection).")

        fig = go.Figure()
        end = sess["duration_sec"]
        for sid_, g in events.groupby("student_id"):
            g, start = g.sort_values("t"), None
            for _, e in g.iterrows():
                if e.kind in ("PRESENT", "RETURNED"):
                    start = e.t
                elif e.kind == "LEFT" and start is not None:
                    fig.add_trace(go.Bar(y=[sid_], x=[(e.t - start) / 60], base=[start / 60],
                                         orientation="h", marker_color=INDIGO, showlegend=False,
                                         hovertemplate=f"{sid_}<br>{fmt_time(start)}–{fmt_time(e.t)}<extra></extra>"))
                    start = None
            if start is not None:
                fig.add_trace(go.Bar(y=[sid_], x=[(end - start) / 60], base=[start / 60], orientation="h",
                                     marker_color=INDIGO, showlegend=False,
                                     hovertemplate=f"{sid_}<br>{fmt_time(start)}–{fmt_time(end)}<extra></extra>"))
        fig.update_layout(barmode="overlay", height=max(300, 22 * len(att) + 80),
                          margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Minutes into session",
                          yaxis=dict(autorange="reversed"))
        st.subheader("Presence timeline")
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- engagement
with tab_eng:
    if sw.empty:
        st.info("No engagement data.")
    else:
        st.subheader("Engagement heatmap")
        pivot = sw.pivot_table(index="student_id", columns="t_end", values="score")
        fig = go.Figure(go.Heatmap(z=pivot.values, x=(pivot.columns / 60).round(1), y=pivot.index,
                                   colorscale="RdYlGn", zmin=0, zmax=1, colorbar=dict(title="Score")))
        fig.update_layout(height=max(320, 22 * len(pivot) + 80), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Minutes into session", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Individual scores are for pattern-spotting, not grading. Turn on anonymization in "
                   "the config to keep students as Student_NN.")

        a, b = st.columns(2)
        with a:
            st.subheader("What drives the score")
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=cw.t_end / 60, y=cw.attention, name="Attention"))
            f2.add_trace(go.Scatter(x=cw.t_end / 60, y=1 - cw.inactivity, name="Activity (1 − inactivity)"))
            f2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1], tickformat=".0%"),
                             xaxis_title="Minutes", legend=dict(orientation="h", y=1.15))
            st.plotly_chart(f2, width="stretch")
        with b:
            st.subheader("Audio activity")
            f3 = go.Figure(go.Bar(x=cw.t_end / 60, y=cw.speech_ratio,
                                  marker_color=[GREEN if m == "discussion" else INDIGO if m == "lecture" else SLATE
                                                for m in cw.audio_mode],
                                  customdata=cw.audio_mode, hovertemplate="%{customdata}<br>%{y:.0%}<extra></extra>"))
            f3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1], tickformat=".0%",
                             title="Speech share"), xaxis_title="Minutes")
            st.plotly_chart(f3, width="stretch")

# ----------------------------------------------------------------------------- suggestions
with tab_sug:
    st.caption("Advisory only. You know your class best; these are prompts, not instructions.")
    color = {"warning": AMBER, "info": INDIGO, "positive": GREEN}
    if not sugg:
        st.success("No issues detected in this session.")
    for s in sugg:
        st.markdown(f'<div class="sg" style="border-color:{color[s["severity"]]}"><b>{s["title"]}</b> '
                    f'<small>· minute {int(s["t"] // 60)}</small><br>{s["detail"]}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- privacy
with tab_priv:
    p = settings.privacy
    st.subheader("Privacy by design")
    st.markdown(f"""
- **Fully offline** – no cloud calls; the optional language model runs locally (Ollama).
- **No raw video or audio stored** – only numbers (scores, timestamps) are saved. Frame storage: `{p.store_frames}`.
- **Anonymous IDs** – students appear as `Student_NN`. Anonymization: `{p.anonymize}`.
- **Consent gate** – live analysis requires an explicit `--consent` flag (`{p.consent_required}`).
- **Retention** – sessions older than `{p.retention_days}` days can be purged automatically.
""")
    st.subheader("Export")
    st.download_button("⬇ HTML report", build_html_report(db, sid, settings.agent),
                       file_name=f"session{sid}_report.html", mime="text/html")
    if not att.empty:
        st.download_button("⬇ Attendance CSV", att.to_csv(index=False), file_name=f"session{sid}_attendance.csv")
    if not cw.empty:
        st.download_button("⬇ Engagement CSV", cw.to_csv(index=False), file_name=f"session{sid}_engagement.csv")
    st.subheader("Data control")
    if st.button("🗑 Delete this session", type="secondary"):
        db.delete_session(sid)
        st.rerun()
