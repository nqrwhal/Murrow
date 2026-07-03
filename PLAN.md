# Murrow Implementation Plan

Copied from the Claude project plan at
`~/.claude/plans/enumerated-orbiting-elephant.md` and normalized for the current
repo/package name.

## Context

Murrow is a build-once, statically served portfolio project for measuring how
news outlets across the political spectrum cover the same event differently,
measured against a neutral AP/Reuters wire baseline.

The core framing is that an LLM is a measurement instrument, not a partisan
judge. On political topics, methodological rigor and visible neutrality are the
product: every score should be receipt-backed, symmetric, auditable, and tied to
a published methodology.

Two feasibility constraints shape the design:

- GDELT DOC 2.0 is the event-discovery source. It is free and keyless, but has a
  rolling three-month window and rate limits. Discovery snapshots are committed
  once instead of repeatedly re-querying.
- Article body text is copyrighted, and AP/Reuters have explicit scraping
  restrictions. Full text is scraped transiently at build time only, then
  discarded. The committed/deployed site stores only derived data: headlines,
  numeric metrics, short fair-use snippets, and source links.

## Goals And Locked Decisions

- Data source: GDELT-assisted discovery plus direct transient full-text scrape.
- Mechanics: offline blind pairwise ELO leaderboard plus transparent multi-axis
  metric index.
- Scope: curated evergreen set of roughly 30 to 75 recent multi-outlet stories.
- Outlets: balanced set of roughly 12, with external spectrum labels shown as
  context rather than treated as ground truth.
- Receipts: short fair-use snippets, schema-capped and always linked to source.
- Stack: Python with uv for the pipeline; Astro and React islands for the static
  frontend.
- Model strategy: model-agnostic through Pioneer-compatible inference. The
  current repo targets `openai/gpt-oss-120b` first while preserving the same
  corpus for comparable future runs.

## Metrics

Per article, compared with the event baseline:

1. Loaded or emotive language: charged words plus neutral alternatives.
2. Selection and omission: baseline facts kept, dropped, or embellished.
3. Word-choice swaps: contested terms with receipt snippets.
4. Headline-to-body fidelity: whether the headline oversells the article body.
5. Attribution balance: whose voices are quoted or centered.

## Architecture

Repo layout:

```text
Murrow/
├── pipeline/                    # uv project
│   ├── pyproject.toml  uv.lock  .python-version
│   ├── config/                  # outlets.toml, events.toml, pipeline.toml
│   ├── src/murrow/
│   │   ├── models.py            # Pydantic schemas and legal invariants
│   │   ├── config.py cache.py gdelt.py scrape.py elo.py
│   │   ├── llm/                 # client and prompts
│   │   ├── stages/              # discover through publish
│   │   └── cli.py
│   ├── data/
│   │   ├── raw/                 # committed discovery/corpus snapshots
│   │   ├── fulltext/            # gitignored transient article text
│   │   ├── runs/                # committed derived run artifacts
│   │   └── manifest.json
│   └── tests/
├── site/                        # Astro frontend consuming published JSON
│   ├── src/data/
│   ├── public/data/events/
│   └── src/pages/
└── README.md
```

The seam between halves is the pipeline `publish` stage: it writes static JSON
that the Astro site imports at build time or lazy-loads from `public/data`.

## Pipeline Stages

```text
discover -> collect -> fetch -> baseline -> metrics -> battles -> aggregate -> publish
```

1. `discover`: query GDELT and write committed raw snapshots under
   `pipeline/data/raw/events/<id>/gdelt.json`.
2. `collect`: map domains to configured outlets, choose one article per outlet,
   and write `articles.json`.
3. `fetch`: scrape full article text only into gitignored
   `pipeline/data/fulltext/`; classify each URL as `ok`, `thin`, `paywalled`, or
   `failed`.
4. `baseline`: distill the neutral wire article into a headline and numbered
   short key facts. Wire body text is not stored.
5. `metrics`: run structured LLM extraction per article against baseline facts;
   cache by article content, prompt version, model, and schema.
6. `battles`: run blind pairwise order-swapped judgments; store both orderings,
   reasoning, receipts, and whether the swap agreed.
7. `aggregate`: compute deterministic multi-epoch seeded ELO and per-axis
   metric aggregates.
8. `publish`: validate and write site-ready JSON, failing on any legal/schema
   invariant breach.

The intended rebuild contract is that a fresh clone can rebuild the site from
committed derived artifacts with zero network or model spend.

## Cache And Legal Contract

- GDELT is touched only by `discover`.
- Scraping is touched only by `fetch`.
- Model calls are touched only by `baseline`, `metrics`, and `battles`, and only
  on cache misses.
- Full article text may exist only under `pipeline/data/fulltext/`.
- No committed JSON may contain full article bodies.
- Published receipt snippets must be short and schema-capped.

## LLM Call Design

- Use structured outputs everywhere; no free-text verdicts.
- Freeze prompt versions and include prompt/model/schema versions in cache keys.
- Store raw responses for audit.
- Strip outlet names for pairwise judgments and use seeded A/B randomization.
- Run both orderings; disagreement becomes a tie or is explicitly flagged.
- Judge closeness to neutral wire baseline, not ideological agreeability.

## ELO Computation

- Unit: within-event round-robin article pairs.
- Order-independent aggregation: pool all battles and run multi-epoch seeded
  shuffles to converge.
- Ties score 0.5/0.5.
- Sparse coverage is allowed, with provisional flags below the configured battle
  threshold.
- Show confidence intervals rather than false precision.
- Wires are the yardstick, not competitors.
- Use Bradley-Terry MLE as an optional validation cross-check.

## Scraping

- Use `httpx`, `trafilatura`, `selectolax`, and retry/backoff handling.
- Send a real user agent and respect 429 retry headers.
- Classify failures without blocking the full build.
- Quarantine events without a usable baseline and enough usable non-baseline
  outlet articles.

## Frontend

The Astro site should statically render:

- ELO leaderboard with confidence intervals and provisional flags.
- Multi-axis table.
- Event detail pages showing side-by-side coverage differences, highlighted
  receipts, and source links.
- Methodology page explaining baselines, blinding, seeds, prompt versions, and
  external lean-rating attribution.

Large event detail JSON should live under `site/public/data/events/<id>.json`
and be lazy-loaded by React islands. Summary JSON can be imported at build time.

## Config And Rebuild

- `pipeline/config/outlets.toml`: outlet id, name, domains, spectrum context,
  homepage, and baseline flag.
- `pipeline/config/events.toml`: pinned event ids, queries, windows, and
  categories.
- `pipeline/config/pipeline.toml`: thresholds, model settings, prompt versions,
  and ELO parameters.
- `murrow build --from-committed` should rebuild the site artifacts without
  network or model calls once derived data exists.

## Suggested Build Order

1. Lock schemas and config loading.
2. Build GDELT discovery and article collection.
3. Build transient scraper and cache layer.
4. Build prompts, structured model client usage, baseline, metrics, and battles.
5. Build ELO aggregation and publish validation.
6. Build Astro pages against published JSON.

Start with one or two events end-to-end, validate the full seam, then scale the
curated event set.

## Critical Files

- `pipeline/src/murrow/models.py`: Pydantic domain and published-artifact
  schemas; legal invariants belong here.
- `pipeline/src/murrow/cache.py`: content-addressed stage cache and fulltext
  firewall helpers.
- `pipeline/src/murrow/llm/client.py`: Pioneer-compatible structured-call
  wrapper and retry behavior.
- `pipeline/src/murrow/llm/prompts.py`: frozen prompt versions.
- `pipeline/src/murrow/elo.py`: deterministic ELO, provisional flags, and
  confidence intervals.
- `pipeline/config/outlets.toml` and `pipeline/config/events.toml`: curated
  inputs.

## Verification

1. `cd pipeline && uv sync`
2. `uv run murrow --help`
3. Run discovery for one or two pinned events and confirm GDELT snapshots.
4. Run collection and confirm at least the configured minimum outlet coverage.
5. Run fetch and confirm `pipeline/data/fulltext/` is populated but untracked.
6. Run baseline, metrics, and battles; confirm structured JSON validates and
   reruns hit cache.
7. Run aggregate and publish; confirm over-length snippets fail validation.
8. Purge full text and rebuild from committed artifacts.
9. `cd site && npm run build`
10. `cd pipeline && uv run pytest`

## Later Enhancements

- Historical backfill beyond GDELT's rolling window using GDELT raw or BigQuery
  datasets.
- Human voting layered on top of offline ELO.
- Additional model runs over the same frozen corpus for model-to-model
  comparison.
