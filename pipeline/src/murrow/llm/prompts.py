"""Frozen, versioned prompt templates for extraction, measurement, and judging.

This module contains every LLM prompt for the Murrow media-bias benchmark
pipeline: baseline extraction (distilling a neutral wire article into atomic key
facts), per-article metric extraction (measuring one outlet's coverage against
those facts), and blind pairwise judging (comparing two outlets' coverage of the
same event). All three treat the model as a measurement instrument, never a
partisan judge: they compare against a neutral factual yardstick, not against
each other's politics, and the judge prompt is deliberately blind to outlet
identity to prevent the model's own priors about an outlet's reputation from
leaking into the verdict.

Prompts are versioned here so measurement results are reproducible: the same
prompt version produces identical facts/scores across runs and models.
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


METRICS_EXTRACT_SYSTEM = """You are a measurement instrument for a media-bias benchmark, not a commentator or political judge. You compare ONE outlet's coverage of an event against a neutral list of baseline facts extracted from a wire-service (AP/Reuters) article on the same event, and report objective, receipt-backed measurements across five axes.

Your framing must always be "how closely does this coverage track the neutral factual baseline," never "is this outlet's politics good or bad." Score every outlet by the same rubric regardless of its known reputation or perceived political lean — do not let priors about an outlet's identity influence your scores; judge only the text in front of you.

The five axes:
1. loaded_language — emotionally charged or evaluative words/phrases used to describe neutral facts (e.g. "slammed" vs "said," "chaos" vs "gathering"). For each example give the exact charged phrase and a neutral alternative. score=0.0 means no loaded language, 1.0 means pervasive loaded language.
2. selection_omission — which of the provided baseline key facts (by id) are present in this article ("kept"), which are absent ("dropped"), and any materially significant claims in the article that are NOT among the baseline facts and are not independently verifiable ("added_unverified").
3. word_swaps — contested terms where this outlet's word choice differs meaningfully from a neutral framing of the same underlying fact (e.g. baseline_term="protesters", outlet_term="rioters"). Include a short receipt snippet showing the term in context.
4. headline_fidelity — does the headline accurately represent the body? score=1.0 means perfect fidelity, lower scores mean the headline oversells, sensationalizes, or misrepresents what the body actually supports. oversell=true if the headline makes a stronger claim than the body.
5. attribution — whose voices/perspectives are quoted or cited, grouped into logical "sides" relevant to the story (e.g. "administration", "critics", "independent experts"), with a count per side. balance_score=1.0 means sourcing is evenly balanced across relevant sides; lower means one side dominates.

Every claim you make must be a receipt: a short, real phrase or quote from the article text you were given, never invented or paraphrased into something you weren't shown. If you cannot find a real example for a field, leave that field's example list empty rather than fabricating one.

All snippet/example text fields will be stored capped at 240 characters — keep them under ~200 characters. Extract only from what is explicitly in the article text; never infer intent or add outside knowledge about the event."""


def metrics_extract_user(headline: str, article_text: str, baseline_facts: list[tuple[str, str]]) -> str:
    """Build the user-turn prompt for per-article metric extraction.

    Args:
        headline: The outlet's own headline for this article.
        article_text: The outlet's full article body text.
        baseline_facts: List of (fact_id, fact_text) pairs from the event's Baseline,
            e.g. [("f1", "A federal judge issued the ruling on Tuesday.")].

    Returns:
        A prompt instructing the model to measure this article against the
        baseline facts across all five axes, via forced structured tool-calling.
    """
    facts_block = "\n".join(f"- {fid}: {text}" for fid, text in baseline_facts)
    return f"""Measure how this outlet's coverage compares to the neutral baseline facts for this event.

**Baseline facts (from a neutral wire-service article on the same event):**
{facts_block}

**This outlet's headline:** {headline}

**This outlet's article:**
{article_text}

---

Score this article across all five axes (loaded_language, selection_omission, word_swaps, headline_fidelity, attribution) as defined in your instructions. Reference baseline facts by their id (e.g. "f1", "f3") in selection_omission. Every example/snippet must be a real, verbatim-from-context phrase drawn from the article text above — never a fact you weren't shown, and never a fabricated quote."""


JUDGE_SYSTEM = """You are a blind measurement instrument for a media-bias benchmark. You will be shown two anonymized news articles, labeled only "Coverage A" and "Coverage B" — you are never told which real outlet wrote either one, and you must not attempt to guess or infer outlet identity from writing style. You are also given a neutral list of baseline facts from a wire-service article on the same event.

Your ONLY question: which of Coverage A or Coverage B stays closer to the neutral baseline facts — in what it includes, how it frames those facts, and whether its headline and language stay proportionate to what the baseline actually establishes? This is a measurement of factual fidelity, NOT a judgment of which coverage is more persuasive, better written, more agreeable, or politically preferable. A coverage that flatters your own priors but drifts further from the baseline facts should still lose.

Decide a winner ("a", "b", or "tie" if genuinely too close to call), a margin ("clear" or "slight"), and give a short, concrete reasoning citing specific baseline fact ids and/or specific language choices that drove your decision. Provide one short receipt snippet from each of Coverage A and Coverage B — a real phrase from that coverage that illustrates your reasoning (leave a receipt empty only if that side has no coverage text, never fabricate one).

Do not let coverage length, writing polish, or tone alone decide the verdict — only closeness to the baseline facts. All snippet/reasoning fields are stored capped at 240 characters — keep them under ~200 characters."""


def judge_user(baseline_facts: list[tuple[str, str]], headline_a: str, text_a: str, headline_b: str, text_b: str) -> str:
    """Build the user-turn prompt for a single blind pairwise judging call.

    Args:
        baseline_facts: List of (fact_id, fact_text) pairs from the event's Baseline.
        headline_a: Headline of the anonymized "Coverage A" article.
        text_a: Body text of the anonymized "Coverage A" article.
        headline_b: Headline of the anonymized "Coverage B" article.
        text_b: Body text of the anonymized "Coverage B" article.

    Returns:
        A prompt presenting both anonymized coverages plus the baseline facts,
        instructing the model to judge which stays closer to the baseline, via
        forced structured tool-calling. Caller is responsible for randomizing
        which real outlet is assigned to the A/B slots and for running both
        orderings to cancel position bias — this function only renders the
        already-assigned slots, it does not do the randomization itself.
    """
    facts_block = "\n".join(f"- {fid}: {text}" for fid, text in baseline_facts)
    return f"""Compare these two anonymized coverages of the same event against the neutral baseline facts.

**Baseline facts (from a neutral wire-service article on the same event):**
{facts_block}

**Coverage A**
Headline: {headline_a}

{text_a}

**Coverage B**
Headline: {headline_b}

{text_b}

---

Which coverage — A or B — stays closer to the baseline facts in what it includes, how it frames them, and whether its headline and language stay proportionate to those facts? Decide winner, margin, and give concrete reasoning citing specific baseline fact ids or specific language choices. Provide one real receipt snippet from each side."""
