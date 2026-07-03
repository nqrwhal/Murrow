# Murrow

**Murrow** measures how news outlets across the political spectrum cover the
*same* event differently, compared against a neutral wire-service baseline
(AP/Reuters). It's a benchmark, not an opinion: an LLM is used as a
**measurement instrument** — extracting loaded language, omitted facts,
contested word-choice swaps, headline/body fidelity, and attribution balance —
never as a partisan judge. A blind, order-randomized pairwise comparison
produces an ELO leaderboard that converges fully offline, so the site needs no
visitor traffic to be complete on day one.

The project is explicitly **model-agnostic**: the corpus (events, articles, the
neutral baseline) is built once, and any model available through
[Pioneer](https://pioneer.ai)'s inference API (Claude, gpt-oss, Qwen, DeepSeek,
Gemini, Llama, and more) can run the same benchmark against it, so results are
comparable model-to-model. The first benchmark run targets `openai/gpt-oss-120b`.

## Why this design

- **Neutrality is the feature.** A media-bias tool only has credibility if its
  method is more rigorous and symmetric than what it critiques. Every score is
  receipt-backed, judging is blind + position-debiased, the pipeline treats
  every outlet identically, and the full methodology is published.
- **Legal-safe by construction.** Full article text is copyrighted and some
  wires explicitly bar scraping. Murrow scrapes full text only *transiently* at
  build time to compute measurements, then discards it. Nothing committed or
  deployed contains article bodies — only headlines, numeric metrics, short
  fair-use snippets (capped in the schema itself), and links to originals.
- **Build-once, static, model-agnostic.** The pipeline is a chain of cached,
  idempotent stages; a fresh clone can rebuild the deployed site from committed
  derived artifacts with zero network or LLM spend. Swapping in a new benchmark
  model re-uses the same corpus.

## Repo layout

```
pipeline/   Python (uv) — corpus builder + benchmark runner, see pipeline/README.md
site/       Astro + React islands — static frontend consuming pipeline output (coming soon)
```

## Status

Early build-out in progress. See `PLAN.md` for the implementation plan and
`pipeline/README.md` for pipeline usage.
