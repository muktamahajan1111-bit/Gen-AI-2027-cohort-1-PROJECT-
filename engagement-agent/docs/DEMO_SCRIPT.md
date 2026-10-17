# Demo video outline (5-6 minutes)

1. **Problem (30 s)**: manual roll calls, no insight into engagement.
2. **Architecture (60 s)**: show the Mermaid diagram in README; explain that one `Pipeline` serves camera, video and demo.
3. **Code tour (90 s)**: `presence.py` (sustained presence), `fusion/engagement.py` (scoring), `agent/rules.py` (suggestions), `configs/default.toml`.
4. **Live dashboard (2 min)**: run `python -m maea demo` then `python -m maea dashboard`. Walk through KPIs, the engagement chart (drop around minute 24, peak around minute 32), the attendance timeline (late arrivals, one student stepping out), the heatmap, and the suggestions.
5. **Privacy (30 s)**: Privacy tab, `--consent` gate, no frames stored.
6. **Tests + close (30 s)**: `pytest -q`, limitations and roadmap.
