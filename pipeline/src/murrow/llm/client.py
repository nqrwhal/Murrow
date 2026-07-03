"""Pioneer LLM client — model-agnostic structured calls.

Pioneer's Anthropic-compatible endpoint (https://api.pioneer.ai/v1) fronts ~150
models (Claude, gpt-oss, Qwen, DeepSeek, Gemini, Llama, ...) but reports
`structured_outputs: false` and no prompt caching for every model we probed. So:

- Structured output is achieved via **forced tool-calling** (`tool_choice`), which
  we verified live works even for open-weight models like openai/gpt-oss-120b.
- Determinism comes from `temperature=0` (accepted by Pioneer, unlike native Opus
  4.8 which rejects the param entirely) plus frozen, versioned prompts.
- No prompt caching is available, so cost control relies entirely on the on-disk
  content-addressed cache (see cache.py) — never repeat a call whose inputs match.

This client is intentionally model-agnostic: every call takes `model` explicitly so
the same code path benchmarks any model in the Pioneer catalogue.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic, APIStatusError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config import require_env

_TOOL_NAME = "emit_result"


class LLMCallError(RuntimeError):
    """Raised when a model fails to produce a valid structured result after retries."""


def _client() -> Anthropic:
    api_key = require_env("PIONEER_API_KEY")
    base_url = require_env("PIONEER_BASE_URL")
    return Anthropic(api_key=api_key, base_url=base_url)


def _retryable(exc: BaseException) -> bool:
    # Retry on rate limits / server errors / transient network issues; not on 4xx logic errors.
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, (TimeoutError, ConnectionError))


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=30),
    retry=retry_if_exception_type((APIStatusError, TimeoutError, ConnectionError)),
)
def _call_with_backoff(client: Anthropic, **kwargs: Any):
    try:
        return client.messages.create(**kwargs)
    except APIStatusError as exc:
        if _retryable(exc):
            raise
        raise LLMCallError(f"Non-retryable API error {exc.status_code}: {exc.message}") from exc


def structured_call[T: BaseModel](
    *,
    model: str,
    system: str,
    user: str,
    schema_model: type[T],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> tuple[T, dict]:
    """Call `model` and force it to emit `schema_model` via tool-calling.

    Returns (parsed_result, raw_response_dict) — the raw dict is for llm_raw/ audit
    logging so every measurement is reproducible/inspectable.
    """
    schema = schema_model.model_json_schema()
    # Anthropic tool schemas don't want a top-level $defs-only ref; inline is fine
    # for our flat-ish models. Strip pydantic's "title" noise on the root only.
    schema.pop("title", None)

    tools = [
        {
            "name": _TOOL_NAME,
            "description": f"Emit the result conforming to the {schema_model.__name__} schema.",
            "input_schema": schema,
        }
    ]

    client = _client()
    response = _call_with_backoff(
        client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=tools,
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": user}],
    )

    raw = response.model_dump()

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise LLMCallError(f"Model {model} did not return a tool_use block (stop_reason={response.stop_reason})")

    try:
        parsed = schema_model.model_validate(tool_block.input)
    except Exception as exc:  # noqa: BLE001 - surface as LLMCallError for callers
        raise LLMCallError(f"Model {model} tool input failed schema validation: {exc}") from exc

    return parsed, raw
