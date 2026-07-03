"""Load and validate Murrow configuration.

Config pins *intent*: which outlets, which events, which models, tuning knobs.
The authoritative captured *reality* lives in committed ``data/`` after discovery.

All config is TOML under ``pipeline/config/``. Paths are resolved relative to the
pipeline package root so the CLI works from any cwd.
"""

from __future__ import annotations

import os
import tomllib
from functools import cache
from pathlib import Path

from pydantic import BaseModel, Field

from .models import EventSpec, Outlet

# pipeline/src/murrow/config.py -> pipeline/
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PIPELINE_ROOT / "config"
DATA_DIR = PIPELINE_ROOT / "data"


class EloConfig(BaseModel):
    k_factor: int = 24
    provisional_k: int = 40
    provisional_until: int = 10
    epochs: int = 20
    seed: int = 1729
    provisional_threshold: int = 20  # n_battles below which an outlet is "provisional"
    bootstrap_samples: int = 500


class ModelConfig(BaseModel):
    """Which model does what. reference_model builds the shared baseline yardstick."""

    reference_model: str = "claude-opus-4-8"
    default_benchmark_model: str = "openai/gpt-oss-120b"
    max_output_tokens: int = 4096


class Thresholds(BaseModel):
    min_outlets: int = 6  # usable non-baseline articles required, else quarantine
    thin_chars: int = 500  # extracted body shorter than this => "thin"


class PromptVersions(BaseModel):
    baseline: str = "v1"
    extract: str = "v1"
    judge: str = "v1"


class PipelineConfig(BaseModel):
    elo: EloConfig = Field(default_factory=EloConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    prompt_versions: PromptVersions = Field(default_factory=PromptVersions)
    user_agent: str = "MurrowBot/0.1 (media-bias research; +https://github.com/nqrwhal/Murrow)"


def _load_toml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


@cache
def load_outlets() -> list[Outlet]:
    data = _load_toml(CONFIG_DIR / "outlets.toml")
    return [Outlet(**o) for o in data.get("outlet", [])]


@cache
def load_events() -> list[EventSpec]:
    data = _load_toml(CONFIG_DIR / "events.toml")
    return [EventSpec(**e) for e in data.get("event", [])]


@cache
def load_pipeline_config() -> PipelineConfig:
    path = CONFIG_DIR / "pipeline.toml"
    raw = _load_toml(path) if path.exists() else {}
    return PipelineConfig(**raw)


# ------- derived lookups -------
@cache
def domain_to_outlet() -> dict[str, str]:
    """Map every configured domain -> outlet_id."""
    mapping: dict[str, str] = {}
    for outlet in load_outlets():
        for d in outlet.domains:
            mapping[d.lower()] = outlet.id
    return mapping


@cache
def outlet_by_id() -> dict[str, Outlet]:
    return {o.id: o for o in load_outlets()}


@cache
def baseline_outlet_ids() -> list[str]:
    return [o.id for o in load_outlets() if o.is_baseline]


def require_env(name: str) -> str:
    """Fetch a required env var (loading pipeline/.env if present)."""
    _load_dotenv()
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required environment variable {name} is not set (see .env.example)")
    return val


@cache
def _load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dep). Does not override real env vars."""
    env_path = PIPELINE_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
