"""Per-article metric extraction: measure one outlet's coverage vs the event baseline.

This is where a benchmark model does its core measurement work. Every article in
an event (excluding the baseline outlet itself, which is the yardstick, not a
subject) is scored across five axes against the same shared list of baseline key
facts, by the SAME model whose benchmark run this is — so results are comparable
model-to-model but never comparable to a different model's scores for the same
article without accounting for which model produced them (that's what `model` on
every artifact records).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..cache import (
    content_key,
    exists,
    fulltext_path,
    llm_raw_path,
    metrics_path,
    raw_event_dir,
    write_json,
)
from ..config import baseline_outlet_ids, load_pipeline_config
from ..llm import client, prompts
from ..models import (
    ArticleMetrics,
    Attribution,
    Baseline,
    CorpusEvent,
    HeadlineFidelity,
    LoadedLanguage,
    SelectionOmission,
    WordSwap,
)


class _MetricsExtraction(BaseModel):
    """Wire format for the LLM's structured metrics response."""

    loaded_language: LoadedLanguage
    selection_omission: SelectionOmission
    word_swaps: list[WordSwap] = Field(default_factory=list)
    headline_fidelity: HeadlineFidelity
    attribution: Attribution


def _closeness_score(m: _MetricsExtraction, n_baseline_facts: int) -> float:
    """Derive a single 0..1 closeness-to-baseline score from the five raw axes.

    This is a simple, transparent, equally-weighted blend — not itself an LLM
    judgment — so it's fully reproducible from the stored per-axis numbers alone.
    Selection fidelity (kept vs dropped facts) is weighted by how many baseline
    facts exist so a short baseline doesn't get free credit for having little to omit.
    """
    selection_score = (
        len(m.selection_omission.kept) / n_baseline_facts if n_baseline_facts > 0 else 1.0
    )
    unverified_penalty = min(len(m.selection_omission.added_unverified) * 0.1, 0.3)
    selection_score = max(0.0, selection_score - unverified_penalty)

    scores = [
        1.0 - m.loaded_language.score,
        selection_score,
        m.headline_fidelity.score,
        m.attribution.balance_score,
    ]
    return sum(scores) / len(scores)


def run(*, event_id: str, model: str) -> None:
    """Extract metrics for every non-baseline outlet article in an event, for `model`.

    Reads articles.json + baseline.json (both produced earlier, model-independent),
    and each article's fulltext. Writes one ArticleMetrics per outlet to
    data/runs/<model>/metrics/<event_id>/<outlet_id>.json, skipping outlets whose
    metrics already exist (idempotent, cache-guarded on (article content, baseline
    facts, prompt version, model)).

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
    subjects = [a for a in event.articles if a.outlet_id not in baseline_ids]

    scored = 0
    for article in subjects:
        out_path = metrics_path(model, event_id, article.outlet_id)
        if exists(out_path):
            print(f"  {article.outlet_id:10} metrics already extracted, skipping")
            continue

        key = content_key(article.url, article.headline)
        text_path = fulltext_path(key)
        if not exists(text_path):
            print(f"  {article.outlet_id:10} SKIP no fulltext cached (fetch status was not 'ok')")
            continue

        article_text = text_path.read_text(encoding="utf-8")

        try:
            parsed, raw_response = client.structured_call(
                model=model,
                system=prompts.METRICS_EXTRACT_SYSTEM,
                user=prompts.metrics_extract_user(article.headline, article_text, baseline_fact_pairs),
                schema_model=_MetricsExtraction,
                max_tokens=cfg.models.max_output_tokens,
            )
        except client.LLMCallError as exc:
            print(f"  {article.outlet_id:10} FAILED metrics extraction: {exc}")
            continue

        metrics = ArticleMetrics(
            event_id=event_id,
            outlet_id=article.outlet_id,
            model=model,
            prompt_version=cfg.prompt_versions.extract,
            loaded_language=parsed.loaded_language,
            selection_omission=parsed.selection_omission,
            word_swaps=parsed.word_swaps,
            headline_fidelity=parsed.headline_fidelity,
            attribution=parsed.attribution,
            closeness_score=_closeness_score(parsed, len(baseline_fact_pairs)),
        )
        write_json(out_path, metrics)
        write_json(llm_raw_path(model, f"metrics_{event_id}_{article.outlet_id}"), raw_response)
        scored += 1
        print(f"  {article.outlet_id:10} closeness={metrics.closeness_score:.2f}")

    print(f"metrics: scored {scored} new article(s) for event {event_id} with model {model}")
