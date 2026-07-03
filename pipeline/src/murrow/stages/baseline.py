"""Extract neutral key-fact baseline from a wire-service article.

This stage produces a shared factual yardstick for all benchmark models by
extracting key facts from a baseline outlet (AP/Reuters) using a pinned reference
model. The baseline is built once per event and is model-independent — every
measurement is judged against the same set of facts.
"""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from ..cache import content_key, exists, fulltext_path, llm_raw_path, raw_event_dir, write_json
from ..config import baseline_outlet_ids, load_pipeline_config
from ..llm import client, prompts
from ..models import Baseline, CorpusEvent, KeyFact

logger = logging.getLogger(__name__)

# Local pydantic model for the wire format of the LLM response
LLMKeyFact = Annotated[str, StringConstraints(max_length=240, strip_whitespace=True)]


class BaselineExtractionResult(BaseModel):
    """Schema for baseline LLM response: a flat list of extracted facts."""

    key_facts: list[LLMKeyFact] = Field(
        description="Extracted neutral, paraphrased facts from the baseline article"
    )


def run(*, event_id: str) -> None:
    """Extract key facts from a baseline article for an event.

    Reads the corpus event metadata, identifies the baseline article (first one
    from a baseline outlet in baseline priority order), fetches its fulltext,
    calls the reference model to extract key facts, and writes the Baseline object.

    Raises:
        RuntimeError: If articles.json doesn't exist, no baseline article found,
                     fulltext doesn't exist, or LLM call fails.
    """
    event_dir = raw_event_dir(event_id)

    # Load event metadata
    articles_path = event_dir / "articles.json"
    if not articles_path.exists():
        raise RuntimeError(
            f"Event {event_id} has no articles.json. Run 'murrow collect --event-id {event_id}' first."
        )

    event = CorpusEvent.model_validate_json(articles_path.read_text(encoding="utf-8"))

    # Check idempotency
    baseline_path = event_dir / "baseline.json"
    if exists(baseline_path):
        print(f"Baseline already extracted for event {event_id}. Skipping.")
        return

    # Find baseline article (first article from a baseline outlet in baseline priority order)
    baseline_outlet_id_list = baseline_outlet_ids()
    baseline_article = None
    for bid in baseline_outlet_id_list:
        for article in event.articles:
            if article.outlet_id == bid:
                baseline_article = article
                break
        if baseline_article:
            break

    if not baseline_article:
        raise RuntimeError(
            f"Event {event_id} has no articles from baseline outlets {baseline_outlet_id_list}. "
            f"This event should have been quarantined by collect.py."
        )

    # Locate fulltext
    key = content_key(baseline_article.url, baseline_article.headline)
    fulltext = fulltext_path(key)
    if not exists(fulltext):
        raise RuntimeError(
            f"Fulltext for baseline article not found at {fulltext}. "
            f"Run 'murrow fetch --event-id {event_id}' first."
        )

    wire_text = fulltext.read_text(encoding="utf-8")

    # Load config
    cfg = load_pipeline_config()

    # Call LLM
    parsed, raw_response = client.structured_call(
        model=cfg.models.reference_model,
        system=prompts.BASELINE_EXTRACT_SYSTEM,
        user=prompts.baseline_extract_user(baseline_article.headline, wire_text),
        schema_model=BaselineExtractionResult,
        max_tokens=cfg.models.max_output_tokens,
    )

    # Convert LLM facts to KeyFact objects (1-indexed ids: f1, f2, ...)
    key_facts = [
        KeyFact(id=f"f{i+1}", text=fact)
        for i, fact in enumerate(parsed.key_facts)
    ]

    # Build Baseline object
    baseline = Baseline(
        event_id=event_id,
        outlet_id=baseline_article.outlet_id,
        headline=baseline_article.headline,
        url=baseline_article.url,
        key_facts=key_facts,
        reference_model=cfg.models.reference_model,
        prompt_version=cfg.prompt_versions.baseline,
    )

    # Write outputs
    write_json(baseline_path, baseline)
    write_json(llm_raw_path(cfg.models.reference_model, f"baseline_{event_id}"), raw_response)

    print(f"Extracted {len(key_facts)} key facts for event {event_id}.")
