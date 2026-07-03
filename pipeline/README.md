# Murrow — pipeline

Python pipeline that builds the Murrow corpus (events, articles, neutral wire
baseline) and runs any model on the [Pioneer](https://pioneer.ai) inference API
against it as a media-bias benchmark.

See the [repo root README](../README.md) for the project overview.

## Setup

```bash
uv sync
cp .env.example .env   # fill in PIONEER_API_KEY
```

## Usage

```bash
uv run murrow discover --event-id <id> --query '"..."' --start YYYYMMDDHHMMSS --end YYYYMMDDHHMMSS
uv run murrow collect  --event-id <id>
uv run murrow fetch    --event-id <id>
uv run murrow baseline --event-id <id>
```

## Layout

- `src/murrow/models.py` — frozen Pydantic contracts (corpus + run + published artifacts)
- `src/murrow/config.py` — loads `config/*.toml`
- `src/murrow/cache.py` — content-addressed cache / idempotency layer
- `src/murrow/llm/client.py` — Pioneer client (forced tool-calling for structured output)
- `src/murrow/stages/` — pipeline stages (discover → collect → fetch → baseline → metrics → battles → aggregate → publish)
- `data/fulltext/` — **gitignored**: transient full article text, never committed
