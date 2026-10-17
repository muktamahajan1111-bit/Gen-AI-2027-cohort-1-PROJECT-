"""Typed settings loaded from a TOML file (falls back to sane defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


@dataclass
class PresenceConfig:
    confirm_sec: float = 30.0
    window_sec: float = 40.0
    min_ratio: float = 0.7
    gap_sec: float = 10.0
    late_after_min: float = 10.0


@dataclass
class VisionConfig:
    fps: float = 5.0
    detector: str = "auto"
    max_missing_sec: float = 3.0


@dataclass
class EngagementConfig:
    window_sec: float = 60.0
    yaw_limit: float = 25.0
    pitch_limit: float = 20.0
    motion_eps: float = 0.006
    motion_target: float = 0.03
    inactive_min_sec: float = 20.0
    w_attention: float = 0.45
    w_movement: float = 0.25
    w_activity: float = 0.30
    ema_alpha: float = 0.4
    drop_pct: float = 0.20
    peak_pct: float = 0.15
    baseline_windows: int = 5


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    margin_db: float = 8.0
    pause_turn_sec: float = 2.0
    discussion_turns_per_min: float = 6.0


@dataclass
class AgentConfig:
    use_llm: bool = False
    ollama_url: str = "http://localhost:11434"
    model: str = "llama3"
    cooldown_min: float = 10.0
    low_engagement: float = 0.5
    low_engagement_minutes: float = 5.0
    no_interaction_minutes: float = 15.0


@dataclass
class PrivacyConfig:
    store_frames: bool = False
    anonymize: bool = True
    consent_required: bool = True
    retention_days: int = 30


@dataclass
class StorageConfig:
    db_path: str = "data/maea.sqlite3"


@dataclass
class Settings:
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def _merge(obj: Any, data: dict) -> None:
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        if is_dataclass(cur) and isinstance(data[f.name], dict):
            _merge(cur, data[f.name])
        else:
            setattr(obj, f.name, data[f.name])


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from ``path`` (TOML). Missing keys keep their defaults."""
    settings = Settings()
    if path is None:
        default = Path("configs/default.toml")
        path = default if default.exists() else None
    if path is not None and tomllib is not None and Path(path).exists():
        with open(path, "rb") as fh:
            _merge(settings, tomllib.load(fh))
    return settings
