"""工具调用守卫纯单测：三态校验、默认值、LLM 抽取、守卫消息。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.guard import (
    ToolDecision,
    fill_defaults,
    guard_tool_message,
    llm_extract_parameters,
    validate_args,
)


SCHEMA = {
    "properties": {
        "author": {"type": "string", "description": "画家英文名"},
        "top_k": {"type": "integer", "default": 5},
        "analyze": {"type": "boolean", "enum": [True, False], "default": False},
    },
    "required": ["author"],
}


def test_success_fills_defaults():
    d = validate_args(SCHEMA, {"author": "Monet"})
    assert d.status == "SUCCESS"
    assert d.params == {"author": "Monet", "top_k": 5, "analyze": False}


def test_missing_required_is_clarification():
    d = validate_args(SCHEMA, {"top_k": 3})
    assert d.status == "NEED_CLARIFICATION"
    assert d.missing == ["author"]


def test_null_required_is_clarification():
    d = validate_args(SCHEMA, {"author": None})
    assert d.status == "NEED_CLARIFICATION"
    assert d.missing == ["author"]


def test_enum_violation_is_failed():
    d = validate_args(SCHEMA, {"author": "Monet", "analyze": "yes"})
    assert d.status == "FAILED"
    assert any("枚举" in e for e in d.errors)


def test_type_mismatch_is_failed():
    d = validate_args(SCHEMA, {"author": "Monet", "top_k": "five"})
    assert d.status == "FAILED"
    assert any("top_k" in e for e in d.errors)


def test_unknown_key_is_failed():
    d = validate_args(SCHEMA, {"author": "Monet", "artist": "x"})
    assert d.status == "FAILED"
    assert any("未知参数" in e for e in d.errors)


def test_non_dict_args_is_failed():
    d = validate_args(SCHEMA, ["Monet"])
    assert d.status == "FAILED"


def test_union_type_null_passes_for_optional_field():
    schema = {
        "properties": {
            "title": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": [],
    }
    assert validate_args(schema, {}).status == "SUCCESS"
    assert validate_args(schema, {"title": None}).status == "SUCCESS"
    assert validate_args(schema, {"title": "Starry Night"}).status == "SUCCESS"
    assert validate_args(schema, {"title": 3}).status == "FAILED"


def test_fill_defaults_only_missing_keys():
    out = fill_defaults({"author": "Monet"}, SCHEMA["properties"])
    assert out["top_k"] == 5
    assert out["author"] == "Monet"


def test_llm_extract_success():
    def fake_llm(prompt):
        assert "exact_lookup" in prompt
        return '{"params": {"author": "Monet"}, "missing": []}'

    d = llm_extract_parameters("exact_lookup", SCHEMA, "查莫奈的画", llm=fake_llm)
    assert d.status == "SUCCESS"
    assert d.params["author"] == "Monet"


def test_llm_extract_need_clarification_merges_missing():
    def fake_llm(prompt):
        return '{"params": {"top_k": 3}, "missing": ["author"]}'

    d = llm_extract_parameters("exact_lookup", SCHEMA, "随便查查", llm=fake_llm)
    assert d.status == "NEED_CLARIFICATION"
    assert d.missing == ["author"]


def test_llm_extract_failed_on_llm_exception():
    def boom(prompt):
        raise RuntimeError("down")

    d = llm_extract_parameters("exact_lookup", SCHEMA, "q", llm=boom)
    assert d.status == "FAILED"


def test_llm_extract_failed_on_malformed():
    d = llm_extract_parameters("exact_lookup", SCHEMA, "q", llm=lambda p: "nope")
    assert d.status == "FAILED"


def test_guard_message_shapes():
    msg = guard_tool_message("c1", "exact_lookup", ToolDecision(status="FAILED", errors=["bad"]))
    assert msg.tool_call_id == "c1"
    assert "FAILED" in str(msg.content)
    assert msg.id == "guard:c1"

    msg2 = guard_tool_message(
        "c2", "exact_lookup", ToolDecision(status="NEED_CLARIFICATION", missing=["author"])
    )
    assert "author" in str(msg2.content)

    try:
        guard_tool_message("c3", "exact_lookup", ToolDecision(status="SUCCESS", params={}))
        raise AssertionError("SUCCESS 不应生成守卫消息")
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] tool_guard 全部 {len(fns)} 个单测通过")
