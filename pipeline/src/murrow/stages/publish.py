"""Publish a benchmark run's derived data as the static site's JSON artifacts.

This is the last stage before Astro builds, and it's the enforcement point for the
legal firewall: every artifact written here is validated against the
derived-data-only Pydantic models before anything touches disk. The Snippet type
already caps quote length, but this stage additionally guards against total
snippet volume creeping up per article — a hard backstop against ever accidentally
reconstructing something close to a full article body from many small pieces.

Layout mirrors the plan's handoff design: small, aggregate JSON goes to
site/src/data/ (imported at Astro build time), heavy per-event detail goes to
site/public/data/events/<id>.json (fetched client-side by an island so the initial
page load stays light).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..cache import metrics_path, raw_event_dir, run_dir
from ..config import DATA_DIR, PIPELINE_ROOT, load_outlets, load_pipeline_config
from ..models import (
    ArticleMetrics,
    Baseline,
    Battle,
    CorpusEvent,
    PublishedArticle,
    PublishedBattle,
    PublishedEventDetail,
    PublishedEventIndexEntry,
    RunMeta,
    Standings,
)

SITE_ROOT = PIPELINE_ROOT.parent / "site"
SITE_DATA_DIR = SITE_ROOT / "src" / "data"
SITE_PUBLIC_EVENTS_DIR = SITE_ROOT / "public" / "data" / "events"

# A hard backstop on top of the per-field Snippet cap: bounds how many receipt-sized
# fields a single published article can carry in total, so many small "compliant"
# snippets can never be stitched into something resembling a full article body.
MAX_SNIPPET_FIELDS_PER_ARTICLE = 40


class PublishValidationError(RuntimeError):
    """Raised when a derived artifact would violate a legal/schema invariant."""


def _snippet_field_count(article_metrics: ArticleMetrics) -> int:
    count = len(article_metrics.loaded_language.examples) * 2  # phrase + neutral_alt
    count += len(article_metrics.selection_omission.added_unverified)
    count += len(article_metrics.word_swaps) * 3  # baseline_term + outlet_term + snippet
    count += 1 if article_metrics.headline_fidelity.headline_snippet else 0
    return count


def _committed_events() -> list[CorpusEvent]:
    """Every non-quarantined event with a committed articles.json.

    Scans data/raw/events/ directly rather than config/events.toml: per the
    corpus design (see config.py's module docstring), events.toml pins *intent*
    but data/raw/ is the authoritative captured *reality* after discovery — an
    event can exist on disk without a matching (or any longer matching) TOML
    entry, and publish should reflect what was actually built, not what was
    originally queried for.
    """
    events_root = DATA_DIR / "raw" / "events"
    if not events_root.exists():
        return []
    results = []
    for event_dir in sorted(events_root.iterdir()):
        path = event_dir / "articles.json"
        if not path.exists():
            continue
        event = CorpusEvent.model_validate_json(path.read_text(encoding="utf-8"))
        if not event.quarantined:
            results.append(event)
    return results


def _load_baseline(event_id: str) -> Baseline | None:
    path = raw_event_dir(event_id) / "baseline.json"
    if not path.exists():
        return None
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


def _load_metrics_for_event(model: str, event: CorpusEvent) -> dict[str, ArticleMetrics]:
    out = {}
    for article in event.articles:
        path = metrics_path(model, event.id, article.outlet_id)
        if path.exists():
            out[article.outlet_id] = ArticleMetrics.model_validate_json(path.read_text(encoding="utf-8"))
    return out


def _load_battles_for_event(model: str, event_id: str) -> list[Battle]:
    battles_dir = run_dir(model) / "battles" / event_id
    if not battles_dir.exists():
        return []
    return [Battle.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(battles_dir.glob("*.json"))]


def _to_jsonable(value: object) -> object:
    """Recursively convert BaseModel instances (including inside lists) to dicts.

    json.dumps' `default=` hook only fires for values it can't serialize at all,
    which for a list means EACH element gets stringified independently rather than
    dict-ified -- silently producing a list of repr() strings instead of objects.
    Pre-converting avoids that trap for the plain list/list-of-model payloads this
    stage writes (outlets.json, events.json).
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump_json"):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def run(*, model: str) -> None:
    """Validate and publish `model`'s committed run data as static site JSON.

    Raises:
        PublishValidationError: if any article would exceed the snippet-volume
            backstop, or if a quarantined/unbaselined event slips through.
    """
    cfg = load_pipeline_config()
    outlets = load_outlets()
    events = _committed_events()

    index_entries: list[PublishedEventIndexEntry] = []
    total_articles = 0
    total_battles = 0

    for event in events:
        baseline = _load_baseline(event.id)
        if baseline is None:
            # Corpus and baseline stages can be run independently; an event without
            # a committed baseline yet simply isn't ready to publish. Skip, don't fail.
            continue

        metrics_by_outlet = _load_metrics_for_event(model, event)
        battles = _load_battles_for_event(model, event.id)

        published_articles: list[PublishedArticle] = []
        for article in event.articles:
            am = metrics_by_outlet.get(article.outlet_id)
            if am is None:
                continue  # baseline outlet itself, or metrics not yet extracted for `model`

            snippet_count = _snippet_field_count(am)
            if snippet_count > MAX_SNIPPET_FIELDS_PER_ARTICLE:
                raise PublishValidationError(
                    f"event={event.id} outlet={article.outlet_id}: {snippet_count} snippet "
                    f"fields exceeds the {MAX_SNIPPET_FIELDS_PER_ARTICLE}-field backstop"
                )

            published_articles.append(
                PublishedArticle(
                    outlet_id=article.outlet_id,
                    headline=article.headline,
                    url=article.url,
                    published_at=article.published_at,
                    metrics=am,
                    closeness_score=am.closeness_score,
                )
            )

        published_battles = [
            PublishedBattle(
                outlet_a=b.outlet_a,
                outlet_b=b.outlet_b,
                winner_outlet=b.winner_outlet,
                order_swap_agreed=b.order_swap_agreed,
                receipt_a=b.receipt_a,
                receipt_b=b.receipt_b,
                reasoning=b.reasoning,
            )
            for b in battles
        ]

        detail = PublishedEventDetail(
            id=event.id,
            title=event.title,
            date=event.date,
            baseline=baseline,
            articles=published_articles,
            battles=published_battles,
        )
        _write_json(SITE_PUBLIC_EVENTS_DIR / f"{event.id}.json", detail)

        index_entries.append(
            PublishedEventIndexEntry(
                id=event.id,
                title=event.title,
                slug=event.slug,
                date=event.date,
                category=event.category,
                baseline_outlet=baseline.outlet_id,
                n_outlets=len(published_articles),
            )
        )
        total_articles += len(published_articles)
        total_battles += len(published_battles)

    _write_json(SITE_DATA_DIR / "outlets.json", [o.model_dump() for o in outlets])
    _write_json(SITE_DATA_DIR / "events.json", index_entries)

    standings_path = run_dir(model) / "standings.json"
    if standings_path.exists():
        standings = Standings.model_validate_json(standings_path.read_text(encoding="utf-8"))
        _write_json(SITE_DATA_DIR / "standings.json", standings)
    else:
        print(f"publish: no standings.json for {model} yet — run 'murrow aggregate --model {model}' first")

    meta = RunMeta(
        model=model,
        reference_model=cfg.models.reference_model,
        built_at="",  # stamped by the caller/CI, not computed here (workflow scripts can't call Date.now())
        prompt_versions=cfg.prompt_versions.model_dump(),
        n_events=len(index_entries),
        n_articles=total_articles,
        n_battles=total_battles,
    )
    _write_json(SITE_DATA_DIR / "meta.json", meta)

    print(f"publish: {len(index_entries)} events, {total_articles} articles, {total_battles} battles for {model}")
