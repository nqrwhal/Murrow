"""Cache/idempotency layer tests — this is the mechanism every stage's cost control relies on."""

from __future__ import annotations

from murrow.cache import battle_path, content_key, exists, model_slug, write_json


def test_content_key_stable_and_order_sensitive():
    assert content_key("a", "b") == content_key("a", "b")
    assert content_key("a", "b") != content_key("b", "a")


def test_model_slug_filesystem_safe():
    assert model_slug("openai/gpt-oss-120b") == "openai__gpt-oss-120b"
    assert "/" not in model_slug("a/b/c")


def test_battle_path_order_normalized():
    assert battle_path("m", "ev1", "nytimes", "foxnews") == battle_path("m", "ev1", "foxnews", "nytimes")


def test_exists_false_for_empty_file(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("")
    assert not exists(p)


def test_write_json_roundtrip_dict(tmp_path):
    p = tmp_path / "x" / "y.json"
    write_json(p, {"a": 1})
    assert exists(p)
    import json

    assert json.loads(p.read_text()) == {"a": 1}
