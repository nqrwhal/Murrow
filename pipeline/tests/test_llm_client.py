"""llm/client.py contract tests for the schema self-correction retry.

Weaker models occasionally emit a tool call that doesn't match the schema; the
client should feed the validation error back and retry rather than immediately
failing, since a single field-name typo shouldn't cost a model an entire
article's worth of benchmark coverage.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import murrow.llm.client as client_mod
from murrow.llm.client import LLMCallError, structured_call


class _Result(BaseModel):
    value: int


def _fake_response(tool_input: dict | None, stop_reason: str = "tool_use"):
    content = []
    if tool_input is not None:
        content.append(SimpleNamespace(type="tool_use", input=tool_input))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model_dump=lambda: {"content": tool_input, "stop_reason": stop_reason},
    )


def test_structured_call_succeeds_on_first_try(monkeypatch):
    monkeypatch.setattr(client_mod, "_client", lambda: object())
    monkeypatch.setattr(client_mod, "_call_with_backoff", lambda client, **kw: _fake_response({"value": 42}))

    parsed, raw = structured_call(model="m", system="s", user="u", schema_model=_Result)
    assert parsed.value == 42


def test_structured_call_self_corrects_after_invalid_schema(monkeypatch):
    monkeypatch.setattr(client_mod, "_client", lambda: object())
    responses = iter([_fake_response({"wrong_field": 1}), _fake_response({"value": 7})])
    monkeypatch.setattr(client_mod, "_call_with_backoff", lambda client, **kw: next(responses))

    parsed, raw = structured_call(model="m", system="s", user="u", schema_model=_Result)
    assert parsed.value == 7


def test_structured_call_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(client_mod, "_client", lambda: object())
    monkeypatch.setattr(
        client_mod, "_call_with_backoff", lambda client, **kw: _fake_response({"wrong_field": 1})
    )

    with pytest.raises(LLMCallError):
        structured_call(model="m", system="s", user="u", schema_model=_Result)


def test_structured_call_retries_on_missing_tool_use_block(monkeypatch):
    monkeypatch.setattr(client_mod, "_client", lambda: object())
    responses = iter([_fake_response(None, stop_reason="end_turn"), _fake_response({"value": 3})])
    monkeypatch.setattr(client_mod, "_call_with_backoff", lambda client, **kw: next(responses))

    parsed, raw = structured_call(model="m", system="s", user="u", schema_model=_Result)
    assert parsed.value == 3
