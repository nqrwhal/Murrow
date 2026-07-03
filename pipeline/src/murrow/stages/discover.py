"""Discover stage: fetch GDELT articles for an event query.

This stage hits the GDELT DOC 2.0 API once per event, takes a stable snapshot
of matching articles, and caches the result. The ONE-WAY SNAPSHOT PRINCIPLE
ensures that once discovered, an event's article list is never re-fetched
(GDELT's rolling 3-month window means later calls could return different/empty
results). Idempotency is the foundation of the entire pipeline.
"""

from __future__ import annotations

from pathlib import Path

from murrow import cache, gdelt


def run(*, query: str, event_id: str, start: str, end: str) -> None:
    """Discover articles for an event via GDELT API.

    One-way snapshot: if the output already exists, return immediately without
    calling GDELT. Otherwise, fetch the articles, write the result, and report
    what was found.

    Args:
        query: GDELT query string (e.g., '"apple" sourcelang:eng')
        event_id: Unique event identifier
        start: Start datetime in GDELT format YYYYMMDDHHMMSS
        end: End datetime in GDELT format YYYYMMDDHHMMSS

    Raises:
        gdelt.GdeltError: If the GDELT API call fails after retries.
    """
    target_path: Path = cache.raw_event_dir(event_id) / "gdelt.json"

    # ONE-WAY SNAPSHOT PRINCIPLE: if already discovered, skip GDELT.
    if cache.exists(target_path):
        print(f"event {event_id} already discovered, skipping GDELT call")
        return

    # Fetch articles from GDELT.
    try:
        articles = gdelt.search(query, start, end)
    except gdelt.GdeltError as exc:
        print(
            f"GDELT discovery failed for event {event_id}, query={query}, "
            f"start={start}, end={end}: {exc}"
        )
        raise

    # Write the stable snapshot.
    output: dict = {"articles": articles}
    cache.write_json(target_path, output)

    # Report results.
    count: int = len(articles)
    print(f"discovered {count} articles for event {event_id}")
