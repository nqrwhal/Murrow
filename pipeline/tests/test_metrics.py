"""metrics.py contract tests for the pure closeness_score derivation."""

from __future__ import annotations

from murrow.models import (
    Attribution,
    HeadlineFidelity,
    LoadedLanguage,
    SelectionOmission,
)
from murrow.stages.battles import _clip
from murrow.stages.metrics import _closeness_score, _MetricsExtraction


def _extraction(**overrides) -> _MetricsExtraction:
    defaults = dict(
        loaded_language=LoadedLanguage(score=0.0),
        selection_omission=SelectionOmission(kept=["f1", "f2"], dropped=[]),
        headline_fidelity=HeadlineFidelity(score=1.0),
        attribution=Attribution(balance_score=1.0),
    )
    defaults.update(overrides)
    return _MetricsExtraction(**defaults)


def test_closeness_score_perfect_article_scores_near_one():
    m = _extraction()
    assert _closeness_score(m, n_baseline_facts=2) == 1.0


def test_closeness_score_penalizes_loaded_language():
    m = _extraction(loaded_language=LoadedLanguage(score=1.0))
    score = _closeness_score(m, n_baseline_facts=2)
    assert score < 1.0


def test_closeness_score_penalizes_dropped_facts():
    m = _extraction(selection_omission=SelectionOmission(kept=["f1"], dropped=["f2"]))
    score = _closeness_score(m, n_baseline_facts=2)
    assert score < 1.0


def test_closeness_score_penalizes_unverified_additions():
    m = _extraction(
        selection_omission=SelectionOmission(kept=["f1", "f2"], added_unverified=["made up claim"])
    )
    score = _closeness_score(m, n_baseline_facts=2)
    assert score < 1.0


def test_closeness_score_handles_zero_baseline_facts():
    m = _extraction(selection_omission=SelectionOmission(kept=[], dropped=[]))
    score = _closeness_score(m, n_baseline_facts=0)
    assert 0.0 <= score <= 1.0


def test_clip_leaves_short_text_untouched():
    assert _clip("short reasoning") == "short reasoning"


def test_clip_truncates_over_length_text():
    long_text = "x" * 300
    clipped = _clip(long_text)
    assert len(clipped) <= 240
    assert clipped.endswith("…")
