"""Legal-invariant tests for models.py — the Snippet cap is the copyright firewall."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from murrow.models import SNIPPET_MAX_LEN, KeyFact


def test_snippet_rejects_over_length_text():
    with pytest.raises(ValidationError):
        KeyFact(id="f1", text="x" * (SNIPPET_MAX_LEN + 1))


def test_snippet_accepts_max_length_text():
    KeyFact(id="f1", text="x" * SNIPPET_MAX_LEN)
