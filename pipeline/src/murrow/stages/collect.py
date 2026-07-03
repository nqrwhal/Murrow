"""Resolve GDELT article coverage to configured outlets.

This stage reads the GDELT snapshot from discover.run(), maps article domains to
configured outlet IDs (keeping only the first/best match per outlet), and builds
a CorpusEvent model with baseline-presence and quarantine metadata. The work is
idempotent and cheap — always recompute to pick up outlet config changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .. import cache, config, models


def run(*, event_id: str) -> None:
    """Resolve article domains to outlets; build and persist CorpusEvent.

    Reads pipeline/data/raw/events/<event_id>/gdelt.json (created by discover.py).
    Writes pipeline/data/raw/events/<event_id>/articles.json.

    Args:
        event_id: Event identifier, must have a prior discover.run() call.

    Raises:
        RuntimeError: If gdelt.json is missing (no prior discover.run()).
    """
    event_dir = cache.raw_event_dir(event_id)
    gdelt_path = event_dir / "gdelt.json"

    if not gdelt_path.exists():
        raise RuntimeError(
            f"No GDELT data found for event '{event_id}' at {gdelt_path}.\n"
            f"Run: murrow discover --event-id {event_id} --query ... --start ... --end ..."
        )

    # Load configuration and GDELT results
    cfg = config.load_pipeline_config()
    domain_lookup = config.domain_to_outlet()
    baseline_ids = config.baseline_outlet_ids()
    events = config.load_events()

    gdelt_snapshot = cache.read_json(gdelt_path)
    gdelt_articles = gdelt_snapshot.get("articles", []) if isinstance(gdelt_snapshot, dict) else []

    # Find EventSpec for this event, fallback to event_id
    event_spec: models.EventSpec | None = None
    for e in events:
        if e.id == event_id:
            event_spec = e
            break

    title = event_spec.title if event_spec else event_id
    category = event_spec.category if event_spec else "general"

    # Parse date: use EventSpec.start first 8 chars (YYYYMMDD) formatted as YYYY-MM-DD,
    # fallback to today
    date_str: str
    if event_spec and len(event_spec.start) >= 8:
        date_part = event_spec.start[:8]
        try:
            dt = datetime.strptime(date_part, "%Y%m%d")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    # Walk GDELT articles in order (already sorted by relevance)
    # Keep only the first (highest-relevance) article per outlet
    resolved_outlets: set[str] = set()
    articles: list[models.ArticleRef] = []

    for article_dict in gdelt_articles:
        domain = article_dict.get("domain", "").lower()
        if not domain:
            continue

        # Look up outlet_id
        outlet_id = domain_lookup.get(domain)
        if not outlet_id:
            continue

        # Skip if we've already resolved an article for this outlet
        if outlet_id in resolved_outlets:
            continue

        # Parse published_at from seendate field (format YYYYMMDDTHHMMSSZ, UTC-aware)
        published_at: datetime | None = None
        seendate = article_dict.get("seendate", "")
        if seendate:
            try:
                published_at = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=UTC
                )
            except ValueError:
                pass

        # Build ArticleRef (headline stripped of whitespace)
        headline = article_dict.get("title", "").strip()
        url = article_dict.get("url", "")

        article_ref = models.ArticleRef(
            outlet_id=outlet_id,
            url=url,
            headline=headline,
            domain=domain,
            published_at=published_at,
            gdelt_relevance=None,
        )

        articles.append(article_ref)
        resolved_outlets.add(outlet_id)

    # Determine baseline_present
    baseline_present = any(oid in baseline_ids for oid in resolved_outlets)

    # Determine quarantine status
    quarantined = False
    quarantine_reason = ""

    if not baseline_present:
        quarantined = True
        quarantine_reason = "no baseline outlet (AP/Reuters) present"
    else:
        # Count non-baseline articles
        non_baseline_count = sum(
            1 for article in articles if article.outlet_id not in baseline_ids
        )
        if non_baseline_count < cfg.thresholds.min_outlets:
            quarantined = True
            quarantine_reason = (
                f"{non_baseline_count} non-baseline outlets (need {cfg.thresholds.min_outlets})"
            )

    # Build CorpusEvent
    corpus_event = models.CorpusEvent(
        id=event_id,
        title=title,
        slug=event_id,
        date=date_str,
        category=category,
        articles=articles,
        baseline_present=baseline_present,
        quarantined=quarantined,
        quarantine_reason=quarantine_reason,
    )

    # Write to articles.json
    articles_path = event_dir / "articles.json"
    cache.write_json(articles_path, corpus_event)

    # Print summary
    baseline_status = "baseline present" if baseline_present else "NO BASELINE"
    quarantine_status = "QUARANTINED" if quarantined else "OK"
    print(
        f"{len(articles)} outlets resolved, {baseline_status}, {quarantine_status}"
        + (f" ({quarantine_reason})" if quarantine_reason else "")
    )
