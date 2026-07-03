"""collect.py contract tests.

The most important thing this suite guards against: discover.py's real output shape
is {"articles": [...]}, not a bare list. An earlier version of collect.py assumed a
bare list and silently dropped every article on real GDELT output (0 outlets
resolved despite matching domains in the raw snapshot). These tests pin that
contract so it can't regress silently again.
"""

from __future__ import annotations

import murrow.cache as cache
import murrow.stages.collect as collect
from murrow.models import CorpusEvent


def _gdelt_article(domain: str, title: str = "A Headline", seendate: str = "20260615T120000Z") -> dict:
    return {
        "url": f"https://{domain}/article/{title.lower().replace(' ', '-')}",
        "title": title,
        "seendate": seendate,
        "domain": domain,
        "language": "English",
        "sourcecountry": "United States",
    }


def test_collect_resolves_outlets_from_wrapped_articles_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)

    event_id = "test-event"
    gdelt_snapshot = {
        "articles": [
            _gdelt_article("apnews.com"),
            _gdelt_article("foxnews.com"),
            _gdelt_article("nytimes.com"),
        ]
    }
    gdelt_path = cache.raw_event_dir(event_id) / "gdelt.json"
    cache.write_json(gdelt_path, gdelt_snapshot)

    collect.run(event_id=event_id)

    articles_path = cache.raw_event_dir(event_id) / "articles.json"
    assert cache.exists(articles_path)
    event = CorpusEvent.model_validate_json(articles_path.read_text())
    resolved = {a.outlet_id for a in event.articles}
    assert resolved == {"ap", "foxnews", "nytimes"}


def test_collect_keeps_only_first_article_per_outlet(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    event_id = "dup-event"
    gdelt_snapshot = {
        "articles": [
            _gdelt_article("apnews.com", title="First AP Story"),
            _gdelt_article("apnews.com", title="Second AP Story"),
        ]
    }
    cache.write_json(cache.raw_event_dir(event_id) / "gdelt.json", gdelt_snapshot)

    collect.run(event_id=event_id)

    event = CorpusEvent.model_validate_json(
        (cache.raw_event_dir(event_id) / "articles.json").read_text()
    )
    assert len(event.articles) == 1
    assert event.articles[0].headline == "First AP Story"


def test_collect_quarantines_when_no_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    event_id = "no-baseline-event"
    gdelt_snapshot = {"articles": [_gdelt_article("foxnews.com")]}
    cache.write_json(cache.raw_event_dir(event_id) / "gdelt.json", gdelt_snapshot)

    collect.run(event_id=event_id)

    event = CorpusEvent.model_validate_json(
        (cache.raw_event_dir(event_id) / "articles.json").read_text()
    )
    assert event.quarantined is True
    assert "baseline" in event.quarantine_reason.lower()


def test_collect_raises_if_gdelt_snapshot_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    try:
        collect.run(event_id="never-discovered")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "discover" in str(exc)


def test_collect_skips_unconfigured_domains(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    event_id = "unknown-domain-event"
    gdelt_snapshot = {
        "articles": [_gdelt_article("some-random-blog.example.com"), _gdelt_article("apnews.com")]
    }
    cache.write_json(cache.raw_event_dir(event_id) / "gdelt.json", gdelt_snapshot)

    collect.run(event_id=event_id)

    event = CorpusEvent.model_validate_json(
        (cache.raw_event_dir(event_id) / "articles.json").read_text()
    )
    assert {a.outlet_id for a in event.articles} == {"ap"}
