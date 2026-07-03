"""Murrow data contracts.

These Pydantic models are the *frozen interface* every pipeline stage and every
build agent codes against. Two ideas dominate the design:

1. **Corpus / run split.** The *corpus* (events, articles, baseline key-facts) is
   model-independent and built once. A *run* is one benchmark model's measurements
   over that corpus. This lets any model on Pioneer re-run the same benchmark and be
   compared apples-to-apples against an identical yardstick.

2. **Derived-data-only legal firewall.** Published artifacts carry NO article body —
   only headlines, numbers, links, and short fair-use *receipt* snippets. Snippet
   length is capped in the type system (`Snippet`) so an over-long quote is a
   validation error, not a judgment call. Full article text lives only in the
   gitignored ``data/fulltext/`` dir and is never serialized into these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# --------------------------------------------------------------------------- #
# Legal firewall primitive.
# A receipt snippet is a short fair-use quote (a phrase to ~one sentence). Capping
# length in the type means no artifact can accumulate a reconstructable article body.
# --------------------------------------------------------------------------- #
SNIPPET_MAX_LEN = 240

Snippet = Annotated[str, StringConstraints(max_length=SNIPPET_MAX_LEN, strip_whitespace=True)]


class Spectrum(StrEnum):
    """Editorial-lean bucket sourced from external raters (AllSides/Ad Fontes)."""

    L = "L"  # left
    LC = "LC"  # lean left
    C = "C"  # center
    RC = "RC"  # lean right
    R = "R"  # right


# --------------------------------------------------------------------------- #
# Config-derived registry models
# --------------------------------------------------------------------------- #
class Outlet(BaseModel):
    """One news outlet. Loaded from config/outlets.toml."""

    id: str
    name: str
    domains: list[str]
    spectrum: Spectrum
    spectrum_score: int = Field(ge=-6, le=6, description="AllSides-style -6..+6, external context only")
    homepage_url: str = ""
    is_baseline: bool = False


class EventSpec(BaseModel):
    """A curated event query. Loaded from config/events.toml. Pins *intent*."""

    id: str
    title: str
    query: str
    start: str = Field(description="GDELT YYYYMMDDHHMMSS, within last 3 months")
    end: str
    category: str = "general"


# --------------------------------------------------------------------------- #
# Corpus models (model-independent, committed)
# --------------------------------------------------------------------------- #
class ArticleRef(BaseModel):
    """An outlet's article for an event. Resolved by `collect` from GDELT. No body."""

    outlet_id: str
    url: str
    headline: str
    domain: str
    published_at: datetime | None = None
    gdelt_relevance: float | None = None


FetchStatus = Literal["ok", "thin", "paywalled", "failed"]


class KeyFact(BaseModel):
    """One neutral, paraphrased fact from the wire baseline. Yardstick for scoring.

    Text is a Murrow-authored neutral paraphrase (not a verbatim wire sentence) and
    is capped at snippet length — safe to commit and publish.
    """

    id: str  # "f1", "f2", ...
    text: Snippet


class Baseline(BaseModel):
    """Neutral wire distillation for one event. Committed; the wire body is discarded.

    Produced once by a pinned *reference* model so every benchmark model is judged
    against the identical set of key facts.
    """

    event_id: str
    outlet_id: str  # which wire (ap/reuters)
    headline: str
    url: str
    key_facts: list[KeyFact]
    reference_model: str  # model id that extracted these facts
    prompt_version: str


class CorpusEvent(BaseModel):
    """The committed, model-independent record for one event."""

    id: str
    title: str
    slug: str
    date: str
    category: str
    articles: list[ArticleRef]
    baseline_present: bool
    quarantined: bool = False
    quarantine_reason: str = ""


# --------------------------------------------------------------------------- #
# Metric models (one benchmark model's measurement of one article)
# --------------------------------------------------------------------------- #
class LoadedPhrase(BaseModel):
    phrase: Snippet
    neutral_alt: Snippet


class LoadedLanguage(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    examples: list[LoadedPhrase] = Field(default_factory=list)


class SelectionOmission(BaseModel):
    kept: list[str] = Field(default_factory=list, description="baseline key-fact ids present")
    dropped: list[str] = Field(default_factory=list, description="baseline key-fact ids omitted")
    added_unverified: list[Snippet] = Field(default_factory=list)


class WordSwap(BaseModel):
    baseline_term: Snippet
    outlet_term: Snippet
    snippet: Snippet


class HeadlineFidelity(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="1.0 = headline matches body")
    oversell: bool = False
    note: Snippet = ""
    headline_snippet: Snippet = ""


class AttributionSide(BaseModel):
    side: Snippet
    count: int = Field(ge=0)


class Attribution(BaseModel):
    sources: list[AttributionSide] = Field(default_factory=list)
    balance_score: float = Field(ge=0.0, le=1.0, description="1.0 = perfectly balanced sourcing")


class ArticleMetrics(BaseModel):
    """Full measurement of one article by one benchmark model, vs the event baseline."""

    event_id: str
    outlet_id: str
    model: str
    prompt_version: str
    loaded_language: LoadedLanguage
    selection_omission: SelectionOmission
    word_swaps: list[WordSwap] = Field(default_factory=list)
    headline_fidelity: HeadlineFidelity
    attribution: Attribution
    closeness_score: float = Field(ge=0.0, le=1.0, description="derived overall closeness to baseline")


# --------------------------------------------------------------------------- #
# Battle models (blind pairwise judgment by one benchmark model)
# --------------------------------------------------------------------------- #
Winner = Literal["a", "b", "tie"]
Margin = Literal["clear", "slight"]


class SubVerdict(BaseModel):
    """One ordering's verdict. `winner` is in blinded A/B space."""

    winner: Winner
    margin: Margin
    reasoning: Snippet


class Battle(BaseModel):
    """One unordered outlet pair for one event, judged in both orderings (debiased)."""

    event_id: str
    model: str
    outlet_a: str  # real outlet ids (unblinded, for aggregation)
    outlet_b: str
    prompt_version: str
    # Resolved verdict in real-outlet space after collapsing both orderings:
    winner_outlet: str | None  # outlet_a, outlet_b, or None for tie
    order_swap_agreed: bool
    receipt_a: Snippet = ""
    receipt_b: Snippet = ""
    reasoning: Snippet = ""


# --------------------------------------------------------------------------- #
# Published / aggregate artifacts (what the site consumes)
# --------------------------------------------------------------------------- #
class EloStanding(BaseModel):
    outlet_id: str
    rating: float
    rank: int
    n_battles: int
    ci95: tuple[float, float]
    provisional: bool


class AxisScores(BaseModel):
    outlet_id: str
    loaded_language: float
    selection: float
    word_choice: float
    headline_fidelity: float
    attribution: float


class EloMethod(BaseModel):
    k_factor: int
    provisional_k: int
    epochs: int
    seed: int


class Standings(BaseModel):
    """standings.json for one run (one benchmark model)."""

    model: str
    elo: list[EloStanding]
    axes: list[AxisScores]
    method: EloMethod


class RunMeta(BaseModel):
    """meta.json — provenance for the methodology page."""

    model: str
    reference_model: str
    built_at: str
    git_sha: str = ""
    prompt_versions: dict[str, str]
    n_events: int
    n_articles: int
    n_battles: int
    gdelt_window: str = "rolling-3mo"


class PublishedEventIndexEntry(BaseModel):
    """events.json — lightweight index imported at Astro build time."""

    id: str
    title: str
    slug: str
    date: str
    category: str
    baseline_outlet: str
    n_outlets: int


class PublishedArticle(BaseModel):
    """One outlet's article within a published event detail — no article body."""

    outlet_id: str
    headline: str
    url: str
    published_at: datetime | None = None
    metrics: ArticleMetrics
    closeness_score: float


class PublishedBattle(BaseModel):
    """One battle within a published event detail, in real-outlet space."""

    outlet_a: str
    outlet_b: str
    winner_outlet: str | None
    order_swap_agreed: bool
    receipt_a: Snippet = ""
    receipt_b: Snippet = ""
    reasoning: Snippet = ""


class PublishedEventDetail(BaseModel):
    """events/<id>.json — the heavy per-event artifact, fetched client-side."""

    id: str
    title: str
    date: str
    baseline: Baseline
    articles: list[PublishedArticle]
    battles: list[PublishedBattle]
