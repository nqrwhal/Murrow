"""Frozen, versioned prompt templates for baseline extraction and measurement.

This module contains LLM prompts for the Murrow media-bias benchmark pipeline.
Currently exports baseline extraction prompts that distill neutral wire-service
articles (AP/Reuters) into atomic, paraphrased key facts that serve as a factual
yardstick for measuring how different outlets cover the same event.

Prompts are versioned here so measurement results are reproducible: the same
prompt version produces identical facts across runs and models.
"""

from __future__ import annotations

BASELINE_EXTRACT_SYSTEM = """You are a neutral fact-extraction instrument for media-bias measurement infrastructure. Your role is NOT to summarize or commentate, but to extract atomic, factual claims.

Instructions:
- Extract only what is explicitly stated in the article; never infer, speculate, or add outside information.
- Write each fact as a standalone, paraphrased claim in neutral language. Facts must be genuine paraphrases in Murrow's own words, NOT verbatim or near-verbatim quotes from the source.
- COPYRIGHT REQUIREMENT: never reuse a run of more than 4-5 consecutive words from the source text. Rebuild each sentence from scratch — change the sentence structure and word choice, not just a word or two. If you can't rephrase a detail without echoing the source's exact wording, omit that detail rather than lightly edit the source sentence.
- Strip any framing, adjectives, evaluative language, or editorializing present in the original.
- Each fact must be atomic: one discrete, factual claim only. If an idea has multiple parts, split it into multiple facts.
- Each fact will be stored in a field capped at 240 characters, so keep each fact under ~200 characters to leave margin.
- Aim for 5-12 facts (fewer for simple stories, more for complex ones); let the story's factual density guide the count."""


def baseline_extract_user(headline: str, wire_text: str) -> str:
    """Build the user-turn prompt for baseline key-fact extraction.

    Args:
        headline: The article headline.
        wire_text: The full article body text (will be extracted from fulltext cache).

    Returns:
        A prompt instructing the model to extract key facts from the article.
        The model will be forced to emit results via structured tool-calling.
    """
    return f"""Extract key facts from the following wire-service article. These facts will become the factual yardstick for measuring coverage bias across outlets.

**Headline:** {headline}

**Article:**
{wire_text}

---

Extract the core factual claims from this article as a flat list of neutral, paraphrased facts. Each fact should be:
- Atomic: one discrete claim per fact
- Neutral: stripped of any framing or editorializing
- Paraphrased: fully rewritten in your own words and sentence structure — never reuse more than 4-5 consecutive words from the article, even for a single clause
- Short: under ~200 characters to leave room for the 240-character storage limit
- Explicit: only what is stated, not inferred

Provide roughly 5-12 facts depending on the story's complexity and factual density."""
