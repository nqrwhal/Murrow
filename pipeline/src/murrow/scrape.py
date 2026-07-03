"""Pure article fetcher — fail-soft content extraction without caching.

This module fetches and extracts article text from URLs using trafilatura (primary)
and selectolax (fallback). It detects paywalls heuristically and classifies extraction
outcomes without raising exceptions — all failure modes are data, not errors.

Caching is handled by a separate fetch.py stage; this module is purely functional.
"""

from __future__ import annotations

import logging

import httpx
import trafilatura
from pydantic import BaseModel
from selectolax.parser import HTMLParser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import load_pipeline_config
from .models import FetchStatus

logger = logging.getLogger(__name__)


class ArticleFetchResult(BaseModel):
    """Outcome of a single article fetch and extraction."""

    status: FetchStatus
    text: str | None = None
    title: str | None = None


def _detect_paywall(html: str) -> bool:
    """Heuristically detect clear paywall markers in HTML.

    Checks for:
    - JSON-LD "isAccessibleForFree": false
    - Strong paywall markers (metered-content, subscription-wall, meter-wall, article-locked)

    Note: We avoid false positives from paywalls that *could* fire but don't always block
    (e.g., Piano metering is common but not always blocking on first fetch). This detector
    only fires on explicit markers that clearly indicate a hard paywall.
    """
    if not html:
        return False

    lower_html = html.lower()

    # JSON-LD marker
    if '"isaccessibleforfree": false' in lower_html or '"isaccessibleforfree":false' in lower_html:
        return True

    # Only match strong paywall class/id patterns (avoid false positives from piano/meter configs)
    strong_paywall_patterns = [
        "subscription-wall",
        "meter-wall",
        "article-locked",
        "paywall-container",
        "gated-content",
    ]
    for pattern in strong_paywall_patterns:
        if pattern in lower_html:
            return True

    return False


def _extract_title_from_selectolax(html: str) -> str | None:
    """Attempt to extract title using selectolax (fallback)."""
    try:
        parser = HTMLParser(html)
        title_tag = parser.css_first("title")
        if title_tag:
            title_text = title_tag.text()
            if title_text:
                return title_text.strip()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"selectolax title extraction failed: {e}")
    return None


def _extract_paragraphs_from_selectolax(html: str) -> str | None:
    """Extract visible paragraph text using selectolax (fallback)."""
    try:
        parser = HTMLParser(html)
        # Remove script and style tags
        for tag in parser.css("script"):
            tag.decompose()
        for tag in parser.css("style"):
            tag.decompose()
        paragraphs = parser.css("p")
        texts = [p.text() for p in paragraphs]
        combined = "\n".join(t for t in texts if t.strip())
        return combined if combined else None
    except Exception as e:  # noqa: BLE001
        logger.debug(f"selectolax paragraph extraction failed: {e}")
    return None


class _TransientServerError(Exception):
    """Internal signal that a 5xx response should be retried."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=5),
    retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException, _TransientServerError)),
    reraise=True,
)
def _fetch_with_backoff(url: str, user_agent: str, timeout: float) -> httpx.Response:
    """Fetch URL with exponential backoff for transient network/5xx errors."""
    with httpx.Client(http2=True) as client:
        response = client.get(url, headers={"User-Agent": user_agent}, timeout=timeout, follow_redirects=True)
    if response.status_code >= 500:
        raise _TransientServerError(f"{response.status_code} from {url}")
    return response


def fetch_article(url: str, *, thin_chars: int = 500) -> ArticleFetchResult:
    """Fetch and extract article content from a URL.

    Args:
        url: Article URL to fetch
        thin_chars: Minimum text length threshold; content shorter than this is marked "thin"

    Returns:
        ArticleFetchResult with status, extracted text, and title. Never raises; all
        failures are classified as status="failed", "thin", or "paywalled".
    """
    try:
        cfg = load_pipeline_config()
        user_agent = cfg.user_agent
        timeout = 15.0

        # Fetch HTML with retries for transient network/5xx errors (4xx is terminal).
        try:
            response = _fetch_with_backoff(url, user_agent, timeout)
        except (httpx.NetworkError, httpx.TimeoutException, _TransientServerError) as e:
            logger.debug(f"Fetch failed after retries for {url}: {e}")
            return ArticleFetchResult(status="failed")

        if response.status_code >= 400:
            # 4xx client error (403/paywall, 404, etc) — terminal, don't retry.
            logger.debug(f"Client error {response.status_code} for {url}")
            return ArticleFetchResult(status="failed")

        html = response.text

        # Detect paywall before attempting extraction.
        if _detect_paywall(html):
            logger.debug(f"Paywall detected for {url}")
            return ArticleFetchResult(status="paywalled")

        # Extract with trafilatura (primary extraction).
        text = trafilatura.extract(html, include_comments=False, favor_precision=True)

        # If trafilatura yields nothing or very little, fall back to selectolax.
        if not text or len(text) < thin_chars // 2:
            logger.debug(f"trafilatura yielded minimal content for {url}, falling back to selectolax")
            alt_text = _extract_paragraphs_from_selectolax(html)
            if alt_text:
                text = alt_text

        # Extract title (prefer trafilatura's internal title if available, else try selectolax).
        title = _extract_title_from_selectolax(html)

        # Classify by content length.
        if not text or len(text) < thin_chars:
            logger.debug(f"Extracted text too short ({len(text) if text else 0} chars) for {url}")
            return ArticleFetchResult(status="thin")

        return ArticleFetchResult(status="ok", text=text, title=title)

    except Exception as e:  # noqa: BLE001
        # Catch any unexpected runtime error (programming bugs should surface during testing).
        logger.exception(f"Unexpected error fetching {url}: {e}")
        return ArticleFetchResult(status="failed")
