"""Fetch stage: transiently scrape full article text.

This stage downloads the full body text for every article discovered in the
collect stage, storing it in the gitignored fulltext/ cache. It is idempotent:
articles whose text has already been fetched are skipped. All fetch outcomes
(success, paywall, too-thin, failure) are recorded in the manifest for
observability and to detect partial runs.
"""

from __future__ import annotations

from collections import Counter

from murrow import cache, config, models, scrape


def run(*, event_id: str) -> None:
    """Fetch full article text for an event's articles.

    Reads articles.json, scrapes each article's text to the fulltext cache,
    and records all fetch outcomes in the manifest. Handles failures gracefully
    (one bad article doesn't stop the loop) and skips articles whose text has
    already been cached.

    Args:
        event_id: Unique event identifier

    Raises:
        RuntimeError: If the articles.json (output from collect stage) is missing.
    """
    # Load the event's articles from collect stage output.
    articles_path = cache.raw_event_dir(event_id) / "articles.json"
    if not cache.exists(articles_path):
        raise RuntimeError(
            f"event {event_id} has no articles.json. Run 'murrow collect --event-id {event_id}' first."
        )

    event = cache.read_model(articles_path, models.CorpusEvent)

    # Load config and manifest.
    cfg = config.load_pipeline_config()
    thin_chars = cfg.thresholds.thin_chars
    manifest = cache.load_manifest()

    # Ensure fetch key exists in manifest.
    if "fetch" not in manifest:
        manifest["fetch"] = {}

    # Fetch each article's text.
    status_counts: Counter = Counter()
    for article in event.articles:
        key = cache.content_key(article.url, article.headline)
        path = cache.fulltext_path(key)

        # IDEMPOTENCY: if text is already cached, skip the network call but backfill
        # the manifest if missing.
        if cache.exists(path):
            status = "ok"
            status_counts[status] += 1
            if key not in manifest["fetch"]:
                manifest["fetch"][key] = {
                    "event_id": event_id,
                    "outlet_id": article.outlet_id,
                    "url": article.url,
                    "status": status,
                }
            print(f"  {article.outlet_id:10} {status:10} (cached) {article.url}")
            continue

        # Fetch the article text.
        try:
            result = scrape.fetch_article(article.url, thin_chars=thin_chars)
            status = result.status

            # Write the text if successful.
            if result.status == "ok":
                cache.ensure_parent(path)
                path.write_text(result.text, encoding="utf-8")

            # Record in manifest regardless of status.
            manifest["fetch"][key] = {
                "event_id": event_id,
                "outlet_id": article.outlet_id,
                "url": article.url,
                "status": status,
            }
            status_counts[status] += 1
            print(f"  {article.outlet_id:10} {status:10} {article.url}")

        except Exception as e:  # noqa: BLE001
            # Unexpected exception: record as failed and continue.
            status = "failed"
            manifest["fetch"][key] = {
                "event_id": event_id,
                "outlet_id": article.outlet_id,
                "url": article.url,
                "status": status,
            }
            status_counts[status] += 1
            print(f"  {article.outlet_id:10} {status:10} (error: {e}) {article.url}")

    # Save the manifest.
    cache.save_manifest(manifest)

    # Report summary.
    total = len(event.articles)
    summary_parts = [f"{count} {status}" for status, count in sorted(status_counts.items())]
    print(f"fetched {total} articles for event {event_id}: {', '.join(summary_parts)}")
