"""Rule engine: detects classroom patterns and proposes advisory suggestions.
Rules are small, named functions, so adding one is a five-line change."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from maea.config import AgentConfig
from maea.core.models import ClassWindow, Suggestion, TrendEvent


def minute(t: float) -> int:
    return int(t // 60)


@dataclass
class Context:
    windows: list[ClassWindow]
    events: list[TrendEvent]
    cfg: AgentConfig
    window_sec: float

    def last(self, minutes: float) -> list[ClassWindow]:
        n = max(1, math.ceil(minutes * 60 / self.window_sec))
        return self.windows[-n:]

    def enough(self, minutes: float) -> bool:
        return len(self.windows) >= max(1, math.ceil(minutes * 60 / self.window_sec))


Rule = Callable[[Context], Suggestion | None]


def sustained_low_engagement(ctx: Context) -> Suggestion | None:
    mins = ctx.cfg.low_engagement_minutes
    if not ctx.enough(mins):
        return None
    win = ctx.last(mins)
    if not all(w.smoothed < ctx.cfg.low_engagement for w in win):
        return None
    t = win[-1].t_end
    lecturing = sum(w.audio_mode == "lecture" for w in win) > len(win) / 2
    if lecturing:
        return Suggestion(t, "warning", "sustained_low_engagement",
                          "Engagement has stayed low during continuous lecturing",
                          f"Class engagement has been below {ctx.cfg.low_engagement:.0%} for about "
                          f"{mins:.0f} minutes of uninterrupted lecture. Consider a quick poll, "
                          "a think-pair-share, or a short worked example.")
    return Suggestion(t, "warning", "sustained_low_engagement",
                      "Engagement has stayed low",
                      f"Engagement has been below {ctx.cfg.low_engagement:.0%} for about {mins:.0f} "
                      "minutes. A short break or a change of activity may help.")


def attention_drop(ctx: Context) -> Suggestion | None:
    if not ctx.events or not ctx.windows:
        return None
    ev = ctx.events[-1]
    if ev.kind != "drop" or ev.t != ctx.windows[-1].t_end:
        return None
    return Suggestion(ev.t, "warning", "attention_drop",
                      f"Attention dropped around minute {minute(ev.t)}",
                      f"Engagement fell {abs(ev.magnitude):.0%} below the recent baseline. "
                      "Asking a question or changing the pace could bring focus back.")


def silent_and_disengaged(ctx: Context) -> Suggestion | None:
    if not ctx.enough(3):
        return None
    win = ctx.last(3)
    if all(w.audio_mode == "silence" and w.attention < 0.5 for w in win):
        return Suggestion(win[-1].t_end, "info", "silent_and_disengaged",
                          "The room is quiet and attention is low",
                          "Students may be stuck or have lost the thread. Check for understanding "
                          "or invite questions.")
    return None


def no_interaction(ctx: Context) -> Suggestion | None:
    mins = ctx.cfg.no_interaction_minutes
    if not ctx.enough(mins):
        return None
    win = ctx.last(mins)
    if any(w.audio_mode == "discussion" for w in win):
        return None
    return Suggestion(win[-1].t_end, "info", "no_interaction",
                      f"No interaction for {mins:.0f} minutes",
                      "The session has been one-directional for a while. A discussion prompt or "
                      "a small group activity could raise participation.")


def restless_or_inactive(ctx: Context) -> Suggestion | None:
    if not ctx.enough(3):
        return None
    win = ctx.last(3)
    if sum(w.inactivity for w in win) / 3 > 0.6 and win[-1].inactivity > 0.5:
        return Suggestion(win[-1].t_end, "info", "inactive_class",
                          "Many students have been still for a long time",
                          "Prolonged stillness often signals passive listening. "
                          "A brief stretch or hands-on task may re-energise the room.")
    return None


def positive_momentum(ctx: Context) -> Suggestion | None:
    if not ctx.enough(3):
        return None
    win = ctx.last(3)
    if all(w.smoothed >= 0.70 and w.audio_mode == "discussion" for w in win):
        return Suggestion(win[-1].t_end, "positive", "positive_momentum",
                          "Discussion is going well",
                          "Engagement is high and students are interacting. Consider letting this "
                          "activity run a little longer.")
    return None


DEFAULT_RULES: list[Rule] = [sustained_low_engagement, attention_drop, silent_and_disengaged,
                             no_interaction, restless_or_inactive, positive_momentum]


class RuleEngine:
    """Evaluates rules and suppresses repeats with a per-rule cooldown."""

    def __init__(self, cfg: AgentConfig, window_sec: float, rules: list[Rule] | None = None):
        self.cfg, self.window_sec = cfg, window_sec
        self.rules = rules or DEFAULT_RULES
        self._last_fired: dict[str, float] = {}

    def evaluate(self, windows: list[ClassWindow], events: list[TrendEvent]) -> list[Suggestion]:
        ctx = Context(windows, events, self.cfg, self.window_sec)
        out = []
        for rule in self.rules:
            s = rule(ctx)
            if s is None:
                continue
            last = self._last_fired.get(s.rule)
            if last is not None and s.t - last < self.cfg.cooldown_min * 60:
                continue
            self._last_fired[s.rule] = s.t
            out.append(s)
        return out
