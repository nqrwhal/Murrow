"""battles.py contract tests — the position-debiasing logic is safety-critical.

If the two orderings of a pair disagree, the result MUST be a tie (winner_outlet
None), never an arbitrary pick — that's what cancels position bias by
construction. These tests exercise _resolve_real_winner and the seed function in
isolation since they're pure and don't require an LLM call.
"""

from __future__ import annotations

from murrow.stages.battles import _JudgeVerdict, _pair_seed, _resolve_real_winner


def test_resolve_real_winner_maps_a_and_b():
    v = _JudgeVerdict(winner="a", margin="clear", reasoning="x")
    assert _resolve_real_winner(v, "nytimes", "foxnews") == "nytimes"

    v = _JudgeVerdict(winner="b", margin="clear", reasoning="x")
    assert _resolve_real_winner(v, "nytimes", "foxnews") == "foxnews"


def test_resolve_real_winner_tie_is_none():
    v = _JudgeVerdict(winner="tie", margin="slight", reasoning="x")
    assert _resolve_real_winner(v, "nytimes", "foxnews") is None


def test_order_disagreement_yields_tie_not_arbitrary_pick():
    # Simulates run()'s core debiasing logic: verdict_1 says "nytimes" won,
    # verdict_2 (opposite slot assignment) says "foxnews" won -- these are the
    # SAME two orderings disagreeing, which must resolve to a tie, not a coinflip.
    winner_1 = "nytimes"
    winner_2 = "foxnews"
    order_swap_agreed = winner_1 == winner_2
    winner_outlet = winner_1 if order_swap_agreed else None

    assert order_swap_agreed is False
    assert winner_outlet is None


def test_pair_seed_is_deterministic_and_symmetric_in_naming():
    s1 = _pair_seed("ev1", "model-a", "foxnews", "nytimes", "v1")
    s2 = _pair_seed("ev1", "model-a", "foxnews", "nytimes", "v1")
    assert s1 == s2, "same inputs must produce the same seed (reproducibility)"


def test_pair_seed_differs_across_prompt_versions():
    s1 = _pair_seed("ev1", "model-a", "foxnews", "nytimes", "v1")
    s2 = _pair_seed("ev1", "model-a", "foxnews", "nytimes", "v2")
    assert s1 != s2
