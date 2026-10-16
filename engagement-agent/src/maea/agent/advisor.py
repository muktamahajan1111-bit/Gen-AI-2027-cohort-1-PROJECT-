"""Turns metrics + suggestions into a readable session summary.
Works without any model; optionally polishes the text with a *local* Ollama LLM."""
from __future__ import annotations

import json
import urllib.request

from maea.config import AgentConfig


def fmt_time(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class Advisor:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def summarize(self, facts: dict, suggestions: list[dict]) -> str:
        base = self._template(facts, suggestions)
        if self.cfg.use_llm:
            polished = self._ask_llm(base)
            if polished:
                return polished
        return base

    @staticmethod
    def _template(f: dict, suggestions: list[dict]) -> str:
        lines = [f"{f['present']} students were marked present; average engagement was "
                 f"{f['avg_engagement']:.0%}."]
        if f.get("lowest"):
            lines.append(f"Engagement was lowest around minute {int(f['lowest'] // 60)} "
                         f"and highest around minute {int(f['highest'] // 60)}.")
        if f.get("drops"):
            lines.append(f"{f['drops']} attention drop(s) and {f['peaks']} interaction peak(s) were detected.")
        if suggestions:
            lines.append("Suggestions to consider: " + "; ".join(s["title"] for s in suggestions[:3]) + ".")
        lines.append("These are advisory observations, not judgements about individual students.")
        return " ".join(lines)

    def _ask_llm(self, text: str) -> str | None:
        prompt = ("Rewrite this classroom analytics summary as three short, friendly, advisory "
                  "sentences for a teacher. Do not invent facts.\n\n" + text)
        body = json.dumps({"model": self.cfg.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(self.cfg.ollama_url.rstrip("/") + "/api/generate", body,
                                     {"Content-Type": "application/json"})
        try:  # localhost only: nothing leaves the machine
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["response"].strip() or None
        except Exception:
            return None
