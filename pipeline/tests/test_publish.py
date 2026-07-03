"""publish.py contract tests.

Two real bugs found via live integration testing are pinned here:
1. _write_json's json.dumps fallback stringified each element of a list of
   BaseModel instances instead of dict-ifying them (events.json shipped as a
   list of repr() strings, not JSON objects).
2. _committed_events() originally only considered events listed in
   config/events.toml, so a manually- or discover-built event not yet added
   to that file was silently excluded from publish even with full committed
   data on disk.
"""

from __future__ import annotations

import json

import murrow.cache as cache
from murrow.models import CorpusEvent, PublishedEventIndexEntry
from murrow.stages.publish import _committed_events, _to_jsonable, _write_json


def test_to_jsonable_converts_list_of_models_to_dicts():
    entries = [
        PublishedEventIndexEntry(
            id="e1", title="T", slug="e1", date="2026-01-01",
            category="general", baseline_outlet="ap", n_outlets=3,
        )
    ]
    result = _to_jsonable(entries)
    assert isinstance(result[0], dict)
    assert result[0]["id"] == "e1"


def test_write_json_produces_valid_object_list(tmp_path):
    entries = [
        PublishedEventIndexEntry(
            id="e1", title="T", slug="e1", date="2026-01-01",
            category="general", baseline_outlet="ap", n_outlets=3,
        )
    ]
    out_path = tmp_path / "events.json"
    _write_json(out_path, entries)

    loaded = json.loads(out_path.read_text())
    assert isinstance(loaded, list)
    assert isinstance(loaded[0], dict), "must be a JSON object, not a stringified repr"
    assert loaded[0]["id"] == "e1"


def test_committed_events_found_without_matching_events_toml_entry(tmp_path, monkeypatch):
    """An event built directly on disk (e.g. via a manual fixture, or discover with
    an id not yet added to events.toml) must still be picked up by publish -- the
    corpus's source of truth is data/raw/, not the config file."""
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    import murrow.stages.publish as publish_mod

    monkeypatch.setattr(publish_mod, "DATA_DIR", tmp_path)

    event = CorpusEvent(
        id="not-in-toml", title="T", slug="not-in-toml", date="2026-01-01",
        category="general", articles=[], baseline_present=True, quarantined=False,
    )
    cache.write_json(cache.raw_event_dir("not-in-toml") / "articles.json", event)

    events = _committed_events()
    assert [e.id for e in events] == ["not-in-toml"]


def test_committed_events_excludes_quarantined(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    import murrow.stages.publish as publish_mod

    monkeypatch.setattr(publish_mod, "DATA_DIR", tmp_path)

    event = CorpusEvent(
        id="bad-event", title="T", slug="bad-event", date="2026-01-01",
        category="general", articles=[], baseline_present=False,
        quarantined=True, quarantine_reason="no baseline",
    )
    cache.write_json(cache.raw_event_dir("bad-event") / "articles.json", event)

    assert _committed_events() == []
