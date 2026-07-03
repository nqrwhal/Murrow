"""aggregate.py contract tests."""

from __future__ import annotations

from murrow.models import Battle
from murrow.stages.aggregate import _to_battle_results


def _battle(a: str, b: str, winner: str | None) -> Battle:
    return Battle(
        event_id="e", model="m", outlet_a=a, outlet_b=b, prompt_version="v1",
        winner_outlet=winner, order_swap_agreed=(winner is not None),
    )


def test_winner_outlet_a_scores_one():
    results = _to_battle_results([_battle("a", "b", "a")])
    assert results[0].score_a == 1.0


def test_winner_outlet_b_scores_zero():
    results = _to_battle_results([_battle("a", "b", "b")])
    assert results[0].score_a == 0.0


def test_none_winner_is_a_tie():
    """None can mean a genuine tie verdict OR an order-swap disagreement -- both
    must contribute 0.5/0.5, which is the numeric meaning of "cancels position
    bias by construction" from battles.py's design."""
    results = _to_battle_results([_battle("a", "b", None)])
    assert results[0].score_a == 0.5
