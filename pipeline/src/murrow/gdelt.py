"""GDELT DOC 2.0 API client.

This module provides a thin, pure HTTP client for the GDELT Project's Document
API v2. It performs no I/O beyond HTTP requests — caching and persistence are
handled by a separate stage module.

https://www.gdeltproject.org/data/documentation/v2/API_USAGE.html
"""

from __future__ import annotations

import json

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import load_pipeline_config


class GdeltError(Exception):
    """Raised when GDELT API calls fail after retries, or return unparseable data."""


def _should_retry(exc: BaseException) -> bool:
    """Determine if an exception is retryable (429, 5xx, network errors)."""
    if isinstance(exc, httpx.HTTPStatusError):
        # Retry on 429 (rate limit) and 5xx (server errors)
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    # Retry on network and timeout errors
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=30),
    retry=retry_if_exception(_should_retry),
)
def _call_gdelt(url: str, params: dict, user_agent: str) -> dict:
    """Make a single HTTP request to GDELT with retryable error handling.

    The retry decorator wraps this function, so this code path will be retried
    on 429, 5xx, and network errors. On final exhaustion, the exception bubbles
    up to be caught and re-raised as GdeltError by the caller.
    """
    headers = {"User-Agent": user_agent}
    with httpx.Client(timeout=20.0) as client:
        response = client.get(url, params=params, headers=headers)
        # raise_for_status() will raise httpx.HTTPStatusError on non-2xx.
        response.raise_for_status()
        return response.json()


def search(
    query: str,
    start: str,
    end: str,
    *,
    max_records: int = 250,
) -> list[dict]:
    """Query GDELT DOC 2.0 for articles matching a topic and date range.

    Args:
        query: GDELT query string (e.g., '"apple" sourcelang:eng')
        start: Start datetime in GDELT format YYYYMMDDHHMMSS
        end: End datetime in GDELT format YYYYMMDDHHMMSS
        max_records: Maximum number of article results to return (default 250)

    Returns:
        List of article dicts, empty if no results found (GDELT returns {} with
        no "articles" key when a query has zero matches, which is NOT an error).

    Raises:
        GdeltError: If the HTTP response is non-2xx after retries exhausted, or
            if the response body fails to parse as JSON.
    """
    cfg = load_pipeline_config()
    user_agent = cfg.user_agent

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": max_records,
        "startdatetime": start,
        "enddatetime": end,
    }

    try:
        response = _call_gdelt(url, params, user_agent)
    except httpx.HTTPStatusError as exc:
        raise GdeltError(
            f"GDELT API returned {exc.response.status_code} after retries exhausted"
        ) from exc
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise GdeltError(f"Network error calling GDELT: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GdeltError(f"GDELT response did not parse as JSON: {exc}") from exc

    # GDELT returns {} (empty dict) when no articles match, not {"articles": []}.
    # This is not an error condition.
    return response.get("articles", [])
