"""Aggregate a benchmark model's committed metrics + battles into standings.

Pure compute, no network/LLM calls — this stage only reads what's already on disk
for one model's run and derives ELO ratings, a Bradley-Terry cross-check, bootstrap
confidence intervals, and the per-axis transparency table. It's cheap enough to
always recompute rather than cache, so adding one new event's metrics/battles and
re-running immediately reflects in the standings.
"""

from __future__ import annotations

from ..cache import run_dir
from ..config import load_outlets, load_pipeline_config
from ..elo import BattleResult, bootstrap_ci, bradley_terry, compute_ratings
from ..models import ArticleMetrics, AxisScores, Battle, EloMethod, EloStanding, Standings


def _load_all_metrics(model: str) -> list[ArticleMetrics]:
    metrics_dir = run_dir(model) / "metrics"
    if not metrics_dir.exists():
        return []
    results = []
    for path in sorted(metrics_dir.glob("*/*.json")):
        results.append(ArticleMetrics.model_validate_json(path.read_text(encoding="utf-8")))
    return results


def _load_all_battles(model: str) -> list[Battle]:
    battles_dir = run_dir(model) / "battles"
    if not battles_dir.exists():
        return []
    results = []
    for path in sorted(battles_dir.glob("*/*.json")):
        results.append(Battle.model_validate_json(path.read_text(encoding="utf-8")))
    return results


def _to_battle_results(battles: list[Battle]) -> list[BattleResult]:
    """Convert committed Battle records into the score_a form elo.py consumes.

    A tie (winner_outlet is None, whether from a genuine tie verdict or an
    order-swap disagreement) contributes 0.5/0.5 — this is what "cancels residual
    position bias by construction" in battles.py actually means numerically.
    """
    results = []
    for b in battles:
        if b.winner_outlet is None:
            score_a = 0.5
        elif b.winner_outlet == b.outlet_a:
            score_a = 1.0
        else:
            score_a = 0.0
        results.append(BattleResult(outlet_a=b.outlet_a, outlet_b=b.outlet_b, score_a=score_a))
    return results


def _axis_scores_for_outlet(outlet_id: str, metrics: list[ArticleMetrics]) -> AxisScores:
    """Average this outlet's per-axis scores across every article it produced.

    Each sub-score is normalized to "higher is closer to baseline" (1.0 = best),
    matching closeness_score's convention, so the published table reads
    consistently across axes without a legend explaining which direction is good.
    """
    own = [m for m in metrics if m.outlet_id == outlet_id]
    n = len(own)
    if n == 0:
        return AxisScores(
            outlet_id=outlet_id, loaded_language=0.0, selection=0.0,
            word_choice=0.0, headline_fidelity=0.0, attribution=0.0,
        )

    loaded = sum(1.0 - m.loaded_language.score for m in own) / n
    headline = sum(m.headline_fidelity.score for m in own) / n
    attribution = sum(m.attribution.balance_score for m in own) / n

    selection_vals = []
    for m in own:
        total = len(m.selection_omission.kept) + len(m.selection_omission.dropped)
        selection_vals.append(len(m.selection_omission.kept) / total if total > 0 else 1.0)
    selection = sum(selection_vals) / n

    # word_choice: fraction of articles with zero flagged contested-term swaps.
    word_choice = sum(1.0 if not m.word_swaps else 0.0 for m in own) / n

    return AxisScores(
        outlet_id=outlet_id,
        loaded_language=loaded,
        selection=selection,
        word_choice=word_choice,
        headline_fidelity=headline,
        attribution=attribution,
    )


def compute_standings(model: str) -> Standings:
    """Build the full Standings object for `model` from its committed run data."""
    cfg = load_pipeline_config()
    all_outlets = load_outlets()
    baseline_ids = {o.id for o in all_outlets if o.is_baseline}
    competing_outlet_ids = [o.id for o in all_outlets if o.id not in baseline_ids]

    battles = _load_all_battles(model)
    battle_results = _to_battle_results(battles)
    metrics = _load_all_metrics(model)

    elo_cfg = cfg.elo
    ratings, n_battles = compute_ratings(
        battle_results, competing_outlet_ids,
        k_factor=elo_cfg.k_factor, provisional_k=elo_cfg.provisional_k,
        provisional_until=elo_cfg.provisional_until, epochs=elo_cfg.epochs, seed=elo_cfg.seed,
    )
    ci = bootstrap_ci(
        battle_results, competing_outlet_ids,
        k_factor=elo_cfg.k_factor, provisional_k=elo_cfg.provisional_k,
        provisional_until=elo_cfg.provisional_until, epochs=elo_cfg.epochs, seed=elo_cfg.seed,
        n_samples=elo_cfg.bootstrap_samples,
    )
    # Bradley-Terry is computed as a cross-check available to callers/tests but is
    # not itself published — see elo.py's module docstring for why.
    bradley_terry(battle_results, competing_outlet_ids)

    ranked = sorted(competing_outlet_ids, key=lambda oid: ratings[oid], reverse=True)
    elo_standings = [
        EloStanding(
            outlet_id=oid,
            rating=round(ratings[oid], 1),
            rank=i + 1,
            n_battles=n_battles.get(oid, 0),
            ci95=(round(ci[oid][0], 1), round(ci[oid][1], 1)),
            provisional=n_battles.get(oid, 0) < elo_cfg.provisional_threshold,
        )
        for i, oid in enumerate(ranked)
    ]

    axes = [_axis_scores_for_outlet(oid, metrics) for oid in competing_outlet_ids]

    return Standings(
        model=model,
        elo=elo_standings,
        axes=axes,
        method=EloMethod(
            k_factor=elo_cfg.k_factor,
            provisional_k=elo_cfg.provisional_k,
            epochs=elo_cfg.epochs,
            seed=elo_cfg.seed,
        ),
    )


def run(*, model: str) -> None:
    """Compute and write standings.json for `model`'s committed run data."""
    standings = compute_standings(model)
    out_path = run_dir(model) / "standings.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(standings.model_dump_json(indent=2), encoding="utf-8")

    print(f"aggregate: {len(standings.elo)} outlets ranked for model {model}")
    for s in standings.elo:
        flag = " (provisional)" if s.provisional else ""
        print(f"  #{s.rank} {s.outlet_id:10} {s.rating:.1f}  n={s.n_battles}{flag}")
