from maea.agent.rules import RuleEngine
from maea.config import AgentConfig
from maea.core.models import ClassWindow


def cw(i, v, mode="lecture", attn=0.7):
    return ClassWindow(i * 60, (i + 1) * 60, v, v, 10, attn, 0.1, mode, 0.6, 1)


def test_sustained_low_engagement_during_lecture():
    eng = RuleEngine(AgentConfig(), 60)
    wins = [cw(i, 0.4) for i in range(6)]
    out = eng.evaluate(wins, [])
    assert any(s.rule == "sustained_low_engagement" and "lecturing" in s.title.lower() for s in out)


def test_cooldown_suppresses_repeats():
    eng = RuleEngine(AgentConfig(cooldown_min=10), 60)
    wins = [cw(i, 0.4) for i in range(6)]
    assert eng.evaluate(wins, [])
    wins.append(cw(6, 0.4))
    assert not any(s.rule == "sustained_low_engagement" for s in eng.evaluate(wins, []))


def test_positive_feedback():
    eng = RuleEngine(AgentConfig(), 60)
    out = eng.evaluate([cw(i, 0.85, "discussion") for i in range(3)], [])
    assert any(s.severity == "positive" for s in out)
