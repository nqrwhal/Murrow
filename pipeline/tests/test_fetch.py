"""fetch.py contract tests — cached fulltext must never trigger a re-fetch."""

from __future__ import annotations

from datetime import UTC, datetime

import murrow.cache as cache
import murrow.stages.fetch as fetch
from murrow.models import ArticleRef, CorpusEvent
from murrow.scrape import ArticleFetchResult


def _write_event(event_id: str, articles: list[ArticleRef]) -> None:
    event = CorpusEvent(
        id=event_id,
        title="t",
        slug=event_id,
        date="2026-01-01",
        category="general",
        articles=articles,
        baseline_present=True,
    )
    cache.write_json(cache.raw_event_dir(event_id) / "articles.json", event)


def test_fetch_skips_network_call_when_fulltext_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    article = ArticleRef(
        outlet_id="ap", url="https://apnews.com/x", headline="H", domain="apnews.com",
        published_at=datetime.now(UTC),
    )
    _write_event("ev1", [article])

    key = cache.content_key(article.url, article.headline)
    cache.ensure_parent(cache.fulltext_path(key))
    cache.fulltext_path(key).write_text("already cached body text")

    calls = []
    monkeypatch.setattr(
        fetch.scrape, "fetch_article", lambda *a, **kw: calls.append(1) or ArticleFetchResult(status="ok")
    )

    fetch.run(event_id="ev1")
    assert calls == [], "cached fulltext must not trigger a network fetch"

    manifest = cache.load_manifest()
    assert manifest["fetch"][key]["status"] == "ok"


def test_fetch_records_all_statuses_and_never_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    articles = [
        ArticleRef(outlet_id="ap", url="https://a.com/1", headline="H1", domain="a.com"),
        ArticleRef(outlet_id="foxnews", url="https://b.com/2", headline="H2", domain="b.com"),
    ]
    _write_event("ev2", articles)

    results = iter([ArticleFetchResult(status="ok", text="x" * 600), ArticleFetchResult(status="paywalled")])
    monkeypatch.setattr(fetch.scrape, "fetch_article", lambda *a, **kw: next(results))

    fetch.run(event_id="ev2")

    manifest = cache.load_manifest()
    statuses = {v["status"] for v in manifest["fetch"].values()}
    assert statuses == {"ok", "paywalled"}


def test_fetch_raises_if_articles_json_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path)
    try:
        fetch.run(event_id="never-collected")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "collect" in str(exc)
