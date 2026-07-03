"""Blind pairwise judging: which outlet's coverage stays closer to the baseline.

This is the credibility-critical stage of the whole project. A media-bias tool is
only trustworthy if its own judging is demonstrably harder to bias than the thing
it measures, so every battle is:

  - BLIND: the judge sees only "Coverage A" / "Coverage B", never real outlet names
    or any hint of outlet identity, so the model's own priors about an outlet's
    reputation cannot leak into the verdict.
  - POSITION-DEBIASED: each pair is judged in BOTH orderings (which real outlet
    sits in the A slot is swapped between the two calls), and which real outlet
    starts in the A slot is chosen by a seed derived from the event/pair/model/
    prompt version — so the assignment is reproducible, not run-order-dependent.
    If the two orderings disagree, the pair is recorded as a tie rather than
    picking one arbitrarily; this cancels any residual position bias by
    construction rather than by hoping the model has none.

Battles never cross events — coverage is only ever compared to other coverage of
the SAME event, since the baseline facts (the yardstick) are event-specific.
"""

from __future__ import annotations

import random
from itertools import combinations

from pydantic import BaseModel

from ..cache import (
    battle_path,
    content_key,
    exists,
    fulltext_path,
    llm_raw_path,
    raw_event_dir,
    write_json,
)
from ..config import baseline_outlet_ids, load_pipeline_config
from ..llm import client, prompts
from ..models import SNIPPET_MAX_LEN, Baseline, Battle, CorpusEvent, Margin, Winner


def _clip(text: str) -> str:
    """Defensively enforce the Snippet cap: models don't always respect the prompt's
    character-count guidance exactly, so truncate a borderline-over-length (but
    otherwise perfectly good) verdict rather than lose it to a validation error."""
    text = text.strip()
    return text if len(text) <= SNIPPET_MAX_LEN else text[: SNIPPET_MAX_LEN - 1].rstrip() + "…"


class _JudgeVerdict(BaseModel):
    """Wire format for the LLM's structured judge response, in blinded A/B space."""

    winner: Winner
    margin: Margin
    reasoning: str
    receipt_a: str = ""
    receipt_b: str = ""


def _pair_seed(event_id: str, model: str, outlet_lo: str, outlet_hi: str, prompt_version: str) -> int:
    """Deterministic seed for which real outlet starts in the 'A' slot for this pair."""
    digest = content_key(event_id, model, outlet_lo, outlet_hi, prompt_version)
    return int(digest[:8], 16)


def _judge_once(
    model: str,
    max_tokens: int,
    baseline_fact_pairs: list[tuple[str, str]],
    headline_a: str,
    text_a: str,
    headline_b: str,
    text_b: str,
) -> _JudgeVerdict:
    parsed, _raw = client.structured_call(
        model=model,
        system=prompts.JUDGE_SYSTEM,
        user=prompts.judge_user(baseline_fact_pairs, headline_a, text_a, headline_b, text_b),
        schema_model=_JudgeVerdict,
        max_tokens=max_tokens,
    )
    return parsed


def _resolve_real_winner(verdict: _JudgeVerdict, a_outlet: str, b_outlet: str) -> str | None:
    """Map a blinded verdict ("a"/"b"/"tie") back to a real outlet id, or None for tie."""
    if verdict.winner == "a":
        return a_outlet
    if verdict.winner == "b":
        return b_outlet
    return None


def run(*, event_id: str, model: str) -> None:
    """Judge every unordered outlet pair for an event, for `model`.

    Reads articles.json + baseline.json and each subject outlet's fulltext.
    Writes one Battle per pair to data/runs/<model>/battles/<event_id>/<a>__<b>.json,
    order-normalized so re-running is idempotent regardless of pair order.

    Raises:
        RuntimeError: if articles.json or baseline.json is missing for this event.
    """
    event_dir = raw_event_dir(event_id)
    articles_path = event_dir / "articles.json"
    baseline_path = event_dir / "baseline.json"

    if not exists(articles_path):
        raise RuntimeError(
            f"Event {event_id} has no articles.json. Run 'murrow collect --event-id {event_id}' first."
        )
    if not exists(baseline_path):
        raise RuntimeError(
            f"Event {event_id} has no baseline.json. Run 'murrow baseline --event-id {event_id}' first."
        )

    event = CorpusEvent.model_validate_json(articles_path.read_text(encoding="utf-8"))
    baseline = Baseline.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    baseline_fact_pairs = [(f.id, f.text) for f in baseline.key_facts]
    baseline_ids = set(baseline_outlet_ids())

    cfg = load_pipeline_config()
    prompt_version = cfg.prompt_versions.judge

    # Only outlets with fetched fulltext can be judged; skip anything unfetched/failed.
    subjects = []
    for article in event.articles:
        if article.outlet_id in baseline_ids:
            continue
        key = content_key(article.url, article.headline)
        text_path = fulltext_path(key)
        if exists(text_path):
            subjects.append((article, text_path.read_text(encoding="utf-8")))

    if len(subjects) < 2:
        print(f"battles: fewer than 2 judgeable outlets for event {event_id}, nothing to do")
        return

    judged = 0
    for (article_1, text_1), (article_2, text_2) in combinations(subjects, 2):
        outlet_lo, outlet_hi = sorted([article_1.outlet_id, article_2.outlet_id])
        out_path = battle_path(model, event_id, outlet_lo, outlet_hi)
        if exists(out_path):
            print(f"  {outlet_lo} vs {outlet_hi}  already judged, skipping")
            continue

        by_id = {article_1.outlet_id: (article_1, text_1), article_2.outlet_id: (article_2, text_2)}
        seed = _pair_seed(event_id, model, outlet_lo, outlet_hi, prompt_version)
        rng = random.Random(seed)
        first_a_id, first_b_id = (outlet_lo, outlet_hi) if rng.random() < 0.5 else (outlet_hi, outlet_lo)

        article_a, text_a = by_id[first_a_id]
        article_b, text_b = by_id[first_b_id]

        try:
            verdict_1 = _judge_once(
                model, cfg.models.max_output_tokens, baseline_fact_pairs,
                article_a.headline, text_a, article_b.headline, text_b,
            )
            # Swapped ordering: same two real outlets, opposite A/B slot assignment.
            verdict_2 = _judge_once(
                model, cfg.models.max_output_tokens, baseline_fact_pairs,
                article_b.headline, text_b, article_a.headline, text_a,
            )
        except client.LLMCallError as exc:
            print(f"  {outlet_lo} vs {outlet_hi}  FAILED judging: {exc}")
            continue

        winner_1 = _resolve_real_winner(verdict_1, first_a_id, first_b_id)
        # verdict_2 was judged with slots swapped, so map its "a"/"b" back accordingly.
        winner_2 = _resolve_real_winner(verdict_2, first_b_id, first_a_id)

        order_swap_agreed = winner_1 == winner_2
        winner_outlet = winner_1 if order_swap_agreed else None

        battle = Battle(
            event_id=event_id,
            model=model,
            outlet_a=outlet_lo,
            outlet_b=outlet_hi,
            prompt_version=prompt_version,
            winner_outlet=winner_outlet,
            order_swap_agreed=order_swap_agreed,
            receipt_a=_clip(verdict_1.receipt_a or verdict_2.receipt_b),
            receipt_b=_clip(verdict_1.receipt_b or verdict_2.receipt_a),
            reasoning=_clip(verdict_1.reasoning),
        )
        write_json(out_path, battle)
        write_json(
            llm_raw_path(model, f"battle_{event_id}_{outlet_lo}__{outlet_hi}"),
            {"verdict_1": verdict_1.model_dump(), "verdict_2": verdict_2.model_dump()},
        )
        judged += 1
        outcome = winner_outlet or ("tie" if order_swap_agreed else "tie (order disagreement)")
        print(f"  {outlet_lo} vs {outlet_hi}  -> {outcome}")

    print(f"battles: judged {judged} new pair(s) for event {event_id} with model {model}")
