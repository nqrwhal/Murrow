"""discover.py contract tests — the one-way snapshot guarantee is safety-critical.

GDELT's rolling 3-month window means a second call for the same event could
silently return different (or empty) results. Once a snapshot exists on disk,
discover.run() must never call GDELT again for that event_id.
"""

from __future__ import annotations

import murrow.cache as cache
import murrow.stages.discover as discover


def test_discover_writes_wrapped_articles_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(discover.gdelt, "search", lambda query, start, end, **kw: [{"domain": "apnews.com"}])

    discover.run(query="q", event_id="ev1", start="20260101000000", end="20260102000000")

    path = cache.raw_event_dir("ev1") / "gdelt.json"
    assert cache.exists(path)
    data = cache.read_json(path)
    assert data == {"articles": [{"domain": "apnews.com"}]}


def test_discover_is_one_way_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        discover.gdelt, "search", lambda query, start, end, **kw: calls.append(1) or [{"domain": "x"}]
    )

    discover.run(query="q", event_id="ev1", start="20260101000000", end="20260102000000")
    assert len(calls) == 1

    discover.run(query="q", event_id="ev1", start="20260101000000", end="20260102000000")
    assert len(calls) == 1, "second call must not re-hit GDELT"


def test_discover_propagates_gdelt_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)

    def _raise(*a, **kw):
        raise discover.gdelt.GdeltError("boom")

    monkeypatch.setattr(discover.gdelt, "search", _raise)

    try:
        discover.run(query="q", event_id="ev1", start="20260101000000", end="20260102000000")
        raise AssertionError("expected GdeltError")
    except discover.gdelt.GdeltError:
        pass
