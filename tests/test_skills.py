"""技能系统单测：SKILL.md 解析、注册、迷你 ReAct 执行器（不触发真实 LLM/网络）。"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from src.skills.activation import (
    apply_slash_activation,
    build_activation_block,
    parse_slash_skill,
)
from src.skills.loader import (
    Skill,
    _parse_front_matter,
    _parse_list,
    _skill_runner,
    _validate_output,
    load_skills,
    register_skills,
)
from src.tools.guard import validate_args


def _artwork_skill():
    """取 artwork_deep_analysis（按 id 定位，不依赖目录排序）。"""
    return next(s for s in load_skills() if s.id == "artwork_deep_analysis")


def test_parse_front_matter_and_list():
    text = """---
name: x
tools: [a, "b"]
max_steps: 4
---
正文"""
    kv, body = _parse_front_matter(text)
    assert kv["name"] == "x"
    assert kv["max_steps"] == "4"
    assert body == "正文"
    assert _parse_list('[a, "b"]') == ["a", "b"]
    assert _parse_list("[]") == []


def test_load_skills_finds_all_skills():
    skills = load_skills()
    ids = {s.id for s in skills}
    assert {
        "artwork_deep_analysis",
        "document_summary",
        "exhibition_research",
        "art_comparison",
        "art_timeline",
        "art_recommendation",
    } <= ids
    by_id = {s.id: s for s in skills}
    assert "exact_lookup" in by_id["artwork_deep_analysis"].tools
    assert "web_search" in by_id["exhibition_research"].tools
    assert all(s.instructions for s in skills)
    # v2：结构化步骤 + 输出 schema
    art = by_id["artwork_deep_analysis"]
    assert len(art.steps) >= 4
    assert {"composition", "color", "brushwork", "subject", "verdict"} <= set(art.output_schema)
    comp = by_id["art_comparison"]
    assert {"subjects", "dimensions", "comparison", "conclusion"} <= set(comp.output_schema)
    tl = by_id["art_timeline"]
    assert {"subject", "periods", "conclusion"} <= set(tl.output_schema)
    rec = by_id["art_recommendation"]
    assert {"candidates", "by_artist"} <= set(rec.output_schema)


def test_register_skills_returns_guard_valid_tools():
    tools = register_skills()
    names = {t.name for t in tools}
    assert {"skill_artwork_deep_analysis", "skill_document_summary",
            "skill_exhibition_research", "skill_art_comparison",
            "skill_art_timeline", "skill_art_recommendation"} <= names
    skill_tool = [t for t in tools if t.name == "skill_document_summary"][0]
    schema = skill_tool.args_schema.model_json_schema() if hasattr(
        skill_tool.args_schema, "model_json_schema"
    ) else skill_tool.args_schema.schema()
    assert validate_args(schema, {"task": "总结画册"}).status == "SUCCESS"
    assert validate_args(schema, {}).status == "NEED_CLARIFICATION"


def test_skill_runner_executes_tool_then_finishes():
    skill = _artwork_skill()
    final_json = json.dumps({k: "填充内容" for k in skill.output_schema}, ensure_ascii=False)
    calls = [
        AIMessage(content="", tool_calls=[
            {"name": "exact_lookup", "args": {"title": "X"}, "id": "c1"},
        ]),
        AIMessage(content=final_json),
    ]

    class FakeLLM:
        def __init__(self):
            self.i = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            c = calls[min(self.i, len(calls) - 1)]
            self.i += 1
            return c

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = [{"title": "X", "author": "A"}]
    runner = _skill_runner(skill)
    with patch("src.skills.loader.get_deterministic_llm", return_value=FakeLLM()), \
         patch.dict("src.skills.loader.TOOL_REGISTRY", {"exact_lookup": fake_tool}, clear=True):
        out = runner("分析这幅画")
    assert json.loads(out)["composition"] == "填充内容"
    fake_tool.invoke.assert_called_once_with({"title": "X"})


def test_skill_runner_hits_step_cap():
    class LoopingLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[
                {"name": "exact_lookup", "args": {"title": "X"}, "id": "c1"},
            ])

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = "OK"
    skill = _artwork_skill()
    runner = _skill_runner(skill)
    with patch("src.skills.loader.get_deterministic_llm", return_value=LoopingLLM()), \
         patch.dict("src.skills.loader.TOOL_REGISTRY", {"exact_lookup": fake_tool}, clear=True):
        out = runner("无限循环任务")
    assert "步数上限" in out


def test_skill_runner_fills_missing_fields():
    skill = _artwork_skill()
    final_json = json.dumps({k: "补齐" for k in skill.output_schema}, ensure_ascii=False)

    class FillLLM:
        def __init__(self):
            self.i = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.i += 1
            if self.i == 1:
                return AIMessage(content="这不是 JSON")
            return AIMessage(content=final_json)

    runner = _skill_runner(skill)
    with patch("src.skills.loader.get_deterministic_llm", return_value=FillLLM()):
        out = runner("分析")
    assert json.loads(out)["verdict"] == "补齐"


def test_validate_output():
    schema = {"composition": "构图", "verdict": "点评"}
    ok, missing = _validate_output('{"composition": "a", "verdict": "b"}', schema)
    assert ok and missing == []
    ok, missing = _validate_output('{"composition": "a"}', schema)
    assert not ok and missing == ["verdict"]
    ok, missing = _validate_output("不是 JSON", schema)
    assert not ok
    ok, _ = _validate_output("随便", {})
    assert ok


# ── 技能斜杠激活 ────────────────────────────────────────────
def _fake_skills() -> list[Skill]:
    return [
        Skill(
            id="artwork_deep_analysis",
            name="artwork_deep_analysis",
            description="深度分析画作",
            when_to_use="用户要求深度分析",
            tools=["exact_lookup"],
            max_steps=3,
            instructions="必须先用 exact_lookup 定位画作。",
            steps=["定位画作", "分析构图", "输出 JSON"],
            output_schema={"title": "画作标题"},
        )
    ]


def test_parse_slash_skill_by_id():
    skills = _fake_skills()
    parsed = parse_slash_skill("/artwork_deep_analysis 分析这幅画", skills)
    assert parsed is not None
    skill, task = parsed
    assert skill.id == "artwork_deep_analysis"
    assert task == "分析这幅画"


def test_parse_slash_skill_unknown_returns_none():
    skills = _fake_skills()
    assert parse_slash_skill("/not-a-skill 分析", skills) is None
    assert parse_slash_skill("普通问题", skills) is None


def test_parse_slash_skill_empty_task():
    skills = _fake_skills()
    parsed = parse_slash_skill("/artwork_deep_analysis", skills)
    assert parsed is not None
    assert parsed[1] == ""


def test_apply_slash_activation_rewrites_last_user_message():
    skills = _fake_skills()
    messages = [
        HumanMessage(content="上一轮问题", name="user-input"),
        HumanMessage(content="/artwork_deep_analysis 分析莫奈的睡莲", name="user-input"),
    ]
    out, name, block = apply_slash_activation(messages, skills)
    assert name == "artwork_deep_analysis"
    assert block is not None
    assert "必须先用 exact_lookup 定位画作" in block
    assert out[-1].content == "分析莫奈的睡莲"
    assert out[0].content == "上一轮问题"


def test_apply_slash_activation_no_match():
    skills = _fake_skills()
    messages = [HumanMessage(content="普通问题", name="user-input")]
    out, name, block = apply_slash_activation(messages, skills)
    assert name is None
    assert block is None
    assert out[0].content == "普通问题"


def test_build_activation_block_contains_steps_and_schema():
    block = build_activation_block(_fake_skills()[0])
    assert "1. 定位画作" in block
    assert "输出 JSON schema" in block
    assert '"title"' in block


def test_load_real_skills_are_activatable():
    skills = load_skills()
    assert skills, "agent_skills 目录应有至少一个技能"
    parsed = parse_slash_skill(f"/{skills[0].id} 执行", skills)
    assert parsed is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] skills 全部 {len(fns)} 个单测通过")
