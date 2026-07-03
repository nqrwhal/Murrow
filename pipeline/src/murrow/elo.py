"""Multi-epoch seeded ELO over a sparse pairwise battle graph.

Naive sequential ELO is sensitive to the order battles are processed in, which is
bad for a reproducible, rebuildable site: the same committed battles should always
produce the same standings. We fix this by pooling every battle globally and
replaying the full set for several epochs over a seeded shuffle, so path-dependence
washes out and ratings converge to a stable fixpoint regardless of battle order.

Not every outlet covers every event, so the pairwise graph is sparse by
construction — an outlet's rating is informed only by the battles it actually
played. A provisional K-factor speeds early convergence for outlets with few
battles without destabilizing well-established ones, and a bootstrap resample
gives an honest confidence interval instead of presenting a single number as if
it had no uncertainty.

A Bradley-Terry MLE is computed alongside as a validation cross-check: it's the
strictly order-independent estimator for the same pairwise data, so a large
disagreement in outlet ranking between it and the multi-epoch ELO above would be a
signal the ELO hasn't actually converged.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

INITIAL_RATING = 1500.0


@dataclass(frozen=True)
class BattleResult:
    """One pairwise outcome, in the form the ELO/Bradley-Terry math consumes.

    score_a: 1.0 if outlet_a won, 0.0 if outlet_b won, 0.5 for a tie.
    """

    outlet_a: str
    outlet_b: str
    score_a: float


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _battle_counts(battles: list[BattleResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in battles:
        counts[b.outlet_a] = counts.get(b.outlet_a, 0) + 1
        counts[b.outlet_b] = counts.get(b.outlet_b, 0) + 1
    return counts


def compute_ratings(
    battles: list[BattleResult],
    outlet_ids: list[str],
    *,
    k_factor: int,
    provisional_k: int,
    provisional_until: int,
    epochs: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute ELO ratings for `outlet_ids` from `battles`.

    K-factor is decided by an outlet's TOTAL real battle count (not an epoch-scaled
    counter) so it stays constant across every epoch of the replay — an outlet
    either is or isn't "provisional" for this whole computation, independent of
    which epoch or shuffle order a given battle lands in.

    Returns (ratings, n_battles) — n_battles is the true per-outlet battle count,
    useful for the caller to decide `provisional` display flags separately.
    """
    n_battles = {oid: 0 for oid in outlet_ids}
    for count_oid, count in _battle_counts(battles).items():
        if count_oid in n_battles:
            n_battles[count_oid] = count

    def k_for(outlet_id: str) -> int:
        return provisional_k if n_battles.get(outlet_id, 0) < provisional_until else k_factor

    ratings = {oid: INITIAL_RATING for oid in outlet_ids}
    rng = random.Random(seed)
    order = list(range(len(battles)))

    for _ in range(max(epochs, 1)):
        rng.shuffle(order)
        for i in order:
            b = battles[i]
            if b.outlet_a not in ratings or b.outlet_b not in ratings:
                continue
            ra, rb = ratings[b.outlet_a], ratings[b.outlet_b]
            expected_a = _expected(ra, rb)
            ratings[b.outlet_a] += k_for(b.outlet_a) * (b.score_a - expected_a)
            ratings[b.outlet_b] += k_for(b.outlet_b) * ((1.0 - b.score_a) - (1.0 - expected_a))

    return ratings, n_battles


def bootstrap_ci(
    battles: list[BattleResult],
    outlet_ids: list[str],
    *,
    k_factor: int,
    provisional_k: int,
    provisional_until: int,
    epochs: int,
    seed: int,
    n_samples: int,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Bootstrap a 95%-style CI per outlet by resampling battles with replacement.

    Uses a seed stream disjoint from the main rating computation's seed so the
    resampling randomness doesn't correlate with the point-estimate's shuffle
    randomness. Returns (1500.0, 1500.0) for any outlet that appears in zero
    battles across every resample, which shouldn't happen for an outlet that had
    at least one real battle but is a safe fallback rather than a KeyError.
    """
    if not battles:
        return {oid: (INITIAL_RATING, INITIAL_RATING) for oid in outlet_ids}

    rng = random.Random(seed + 1)
    n = len(battles)
    samples: dict[str, list[float]] = {oid: [] for oid in outlet_ids}

    for i in range(n_samples):
        resample = [battles[rng.randrange(n)] for _ in range(n)]
        ratings, _ = compute_ratings(
            resample,
            outlet_ids,
            k_factor=k_factor,
            provisional_k=provisional_k,
            provisional_until=provisional_until,
            epochs=epochs,
            seed=seed + i + 1,
        )
        for oid in outlet_ids:
            samples[oid].append(ratings[oid])

    ci: dict[str, tuple[float, float]] = {}
    for oid in outlet_ids:
        values = sorted(samples[oid])
        if not values:
            ci[oid] = (INITIAL_RATING, INITIAL_RATING)
            continue
        lo_idx = int(len(values) * (alpha / 2))
        hi_idx = min(int(len(values) * (1 - alpha / 2)), len(values) - 1)
        ci[oid] = (values[lo_idx], values[hi_idx])
    return ci


def bradley_terry(
    battles: list[BattleResult],
    outlet_ids: list[str],
    *,
    iterations: int = 200,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Bradley-Terry MLE strength estimate via the standard MM fixed-point iteration.

    Used only as a validation cross-check against the ELO ranking above — it is
    the strictly order-independent estimator for the same pairwise-comparison
    data, so a rank-order disagreement between the two is a signal worth
    investigating rather than a number published to the site.
    """
    strength = {oid: 1.0 for oid in outlet_ids}
    wins = {oid: 0.0 for oid in outlet_ids}
    pair_counts: dict[tuple[str, str], int] = {}

    for b in battles:
        if b.outlet_a not in strength or b.outlet_b not in strength:
            continue
        wins[b.outlet_a] += b.score_a
        wins[b.outlet_b] += 1.0 - b.score_a
        key = (b.outlet_a, b.outlet_b) if b.outlet_a < b.outlet_b else (b.outlet_b, b.outlet_a)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    for _ in range(iterations):
        updated: dict[str, float] = {}
        for oid in outlet_ids:
            denom = 0.0
            for (a, b), count in pair_counts.items():
                if a == oid:
                    other = b
                elif b == oid:
                    other = a
                else:
                    continue
                denom += count / (strength[oid] + strength[other])
            updated[oid] = wins[oid] / denom if denom > eps else strength[oid]

        log_mean = sum(math.log(max(v, eps)) for v in updated.values()) / len(updated)
        norm = math.exp(log_mean)
        strength = {oid: v / norm for oid, v in updated.items()}

    return strength
