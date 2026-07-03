"""Content-addressed, stage-separated cache — the idempotency layer.

Every expensive operation (a scrape, an LLM call) is keyed by a hash of its full
input (content + prompt version + model + schema version). If the key is present on
disk, the operation is skipped. This is what makes re-runs free, lets us delete the
transient fulltext, and lets a fresh clone rebuild the whole site from committed
derived artifacts with zero network/LLM spend.

Layout under pipeline/data/ (see .gitignore for what's committed vs transient):
    raw/events/<event_id>/gdelt.json        committed
    raw/events/<event_id>/articles.json     committed
    raw/events/<event_id>/baseline.json     committed
    fulltext/<content_key>.txt               GITIGNORED (transient)
    runs/<model_slug>/metrics/<event>/<outlet>.json   committed
    runs/<model_slug>/battles/<event>/<a>__<b>.json   committed
    runs/<model_slug>/llm_raw/<key>.json     committed (audit; no article bodies)
    manifest.json                            committed (statuses, hashes)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import DATA_DIR


def content_key(*parts: Any) -> str:
    """Stable sha256 over the given parts (order matters). Returns hex digest."""
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x1f")  # unit separator to avoid ambiguous concatenation
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model id, e.g. 'openai/gpt-oss-120b' -> 'openai__gpt-oss-120b'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model.replace("/", "__"))


# ------- path helpers -------
def raw_event_dir(event_id: str) -> Path:
    return DATA_DIR / "raw" / "events" / event_id


def fulltext_path(key: str) -> Path:
    return DATA_DIR / "fulltext" / f"{key}.txt"


def run_dir(model: str) -> Path:
    return DATA_DIR / "runs" / model_slug(model)


def metrics_path(model: str, event_id: str, outlet_id: str) -> Path:
    return run_dir(model) / "metrics" / event_id / f"{outlet_id}.json"


def battle_path(model: str, event_id: str, a: str, b: str) -> Path:
    lo, hi = sorted([a, b])  # order-normalized so (a,b) and (b,a) collide
    return run_dir(model) / "battles" / event_id / f"{lo}__{hi}.json"


def llm_raw_path(model: str, key: str) -> Path:
    return run_dir(model) / "llm_raw" / f"{key}.json"


def manifest_path() -> Path:
    return DATA_DIR / "manifest.json"


# ------- read/write helpers -------
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    """Write JSON (pydantic model, dict, or list) with stable formatting."""
    ensure_parent(path)
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False, default=str)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_model(path: Path, model_cls: type[BaseModel]) -> BaseModel:
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def load_manifest() -> dict:
    p = manifest_path()
    if exists(p):
        return read_json(p)
    return {"fetch": {}, "events": {}}


def save_manifest(manifest: dict) -> None:
    write_json(manifest_path(), manifest)
