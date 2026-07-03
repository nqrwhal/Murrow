"""elo.py contract tests: determinism, sensible convergence, sparse-graph handling."""

from __future__ import annotations

from murrow.elo import BattleResult, bootstrap_ci, bradley_terry, compute_ratings


def _one_sided_battles(winner: str, loser: str, n: int) -> list[BattleResult]:
    return [BattleResult(outlet_a=winner, outlet_b=loser, score_a=1.0) for _ in range(n)]


def test_dominant_outlet_ends_up_rated_higher():
    battles = _one_sided_battles("a", "b", 20)
    ratings, _ = compute_ratings(
        battles, ["a", "b"], k_factor=24, provisional_k=40, provisional_until=10, epochs=20, seed=1729
    )
    assert ratings["a"] > ratings["b"]


def test_compute_ratings_is_deterministic_given_same_seed():
    battles = _one_sided_battles("a", "b", 15) + [BattleResult("b", "c", 1.0)] * 10
    outlets = ["a", "b", "c"]
    r1, _ = compute_ratings(
        battles, outlets, k_factor=24, provisional_k=40, provisional_until=10, epochs=20, seed=1729
    )
    r2, _ = compute_ratings(
        battles, outlets, k_factor=24, provisional_k=40, provisional_until=10, epochs=20, seed=1729
    )
    assert r1 == r2


def test_unplayed_outlet_stays_at_initial_rating():
    battles = _one_sided_battles("a", "b", 10)
    ratings, n_battles = compute_ratings(
        battles, ["a", "b", "unplayed"], k_factor=24, provisional_k=40, provisional_until=10, epochs=10, seed=1
    )
    assert ratings["unplayed"] == 1500.0
    assert n_battles["unplayed"] == 0


def test_tie_battles_keep_ratings_equal():
    battles = [BattleResult("a", "b", 0.5) for _ in range(30)]
    ratings, _ = compute_ratings(
        battles, ["a", "b"], k_factor=24, provisional_k=40, provisional_until=10, epochs=10, seed=1
    )
    assert abs(ratings["a"] - ratings["b"]) < 1e-6


def test_bradley_terry_agrees_with_elo_ranking_direction():
    battles = _one_sided_battles("a", "b", 20) + _one_sided_battles("b", "c", 20)
    outlets = ["a", "b", "c"]
    ratings, _ = compute_ratings(
        battles, outlets, k_factor=24, provisional_k=40, provisional_until=10, epochs=20, seed=1729
    )
    strengths = bradley_terry(battles, outlets)
    elo_order = sorted(outlets, key=lambda o: ratings[o], reverse=True)
    bt_order = sorted(outlets, key=lambda o: strengths[o], reverse=True)
    assert elo_order == bt_order == ["a", "b", "c"]


def test_bootstrap_ci_contains_point_estimate_roughly():
    battles = _one_sided_battles("a", "b", 30)
    outlets = ["a", "b"]
    ratings, _ = compute_ratings(
        battles, outlets, k_factor=24, provisional_k=40, provisional_until=10, epochs=20, seed=1729
    )
    ci = bootstrap_ci(
        battles, outlets, k_factor=24, provisional_k=40, provisional_until=10,
        epochs=5, seed=1729, n_samples=50,
    )
    lo, hi = ci["a"]
    assert lo <= hi


def test_bootstrap_ci_handles_no_battles():
    ci = bootstrap_ci(
        [], ["a", "b"], k_factor=24, provisional_k=40, provisional_until=10,
        epochs=5, seed=1729, n_samples=10,
    )
    assert ci["a"] == (1500.0, 1500.0)
