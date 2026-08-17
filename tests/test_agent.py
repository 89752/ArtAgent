"""Agent 层统一单测：图结构 / 节点接线 / 澄清 / 治理 / 守卫 / 上下文 / 改写 / 集成。

纯函数级测试，不触发真实 LLM/网络/模型加载。
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.context import (
    ContextBlocks,
    ContextBudget,
    apply_budget,
    build_profile_block,
    build_session_block,
    build_summary_block,
    condense_tool_messages,
    dedup_artworks,
    estimate_context_chars,
    extract_evidence_from_messages,
    format_numbered_evidence_block,
    format_multi_evidence,
    trim_history,
)
from src.agent.graph import get_graph
from src.agent.nodes.common import _info_gap, ask_user
from src.agent.nodes.general import MAX_TOOL_ROUNDS, _guarded_tool_calls, _ledger_updates, general_tools
from src.agent.rewrite import RewriteResult, normalize_query, rewrite_and_split, rewrite_enabled
from src.agent.state import AgentState
from src.observability import runs as runs_mod
from src.retrieval import structured_retriever as sr
from src.retrieval.base import RetrievalResult
from src.tools.guard import (
    ToolDecision,
    fill_defaults,
    guard_tool_message,
    llm_extract_parameters,
    validate_args,
)
from src.tools.retrieval import _format_result
from src.utils import governance
from web import service as service_mod


@pytest.fixture(autouse=True)
def _runs_db():
    tmp = Path(tempfile.mkdtemp(prefix="agent_runs_"))
    runs_mod._reset_for_tests(tmp / "obs.db")
    try:
        yield
    finally:
        runs_mod._reset_for_tests()


# ══════════════ 图结构 ══════════════
_BRANCH_NODES = {
    "comp_decompose", "comp_retrieve", "comp_synthesize",
    "tl_subject", "tl_periods", "tl_synthesize",
    "rec_extract", "rec_search", "rec_filter", "rec_synthesize",
}
_CORE_NODES = {
    "load_memory", "ask_user", "general_agent", "general_tools",
    "reflection", "save_memory",
}
_REMOVED_NODES = {
    "rewrite_split", "classify", "rag_gate", "direct_answer",
    "multi_retrieve", "tool_upgrade", "web_fallback",
}


def _edges_of(graph) -> list[tuple[str, str]]:
    out = []
    for e in graph.edges:
        start = getattr(e, "start", None) or getattr(e, "source", None)
        end = getattr(e, "end", None) or getattr(e, "target", None)
        if start and end:
            out.append((start, end))
    return out


def test_graph_has_no_branch_nodes():
    graph = get_graph().get_graph()
    nodes = set(graph.nodes.keys())
    assert nodes & _BRANCH_NODES == set()
    assert _CORE_NODES <= nodes
    assert nodes & _REMOVED_NODES == set()


def test_linear_flow_load_memory_to_ask_user_to_react():
    edges = _edges_of(get_graph().get_graph())
    assert ("load_memory", "ask_user") in edges
    assert ("ask_user", "general_agent") in edges
    assert not any(start.startswith(("comp_", "tl_", "rec_")) for start, _ in edges)


def test_general_react_loop_present():
    edges = _edges_of(get_graph().get_graph())
    assert ("general_tools", "general_agent") in edges


# ══════════════ general 节点接线 ══════════════
def _msg(name, content):
    return ToolMessage(content=json.dumps(content, ensure_ascii=False), name=name,
                       tool_call_id="c1")


def test_ledger_records_shown_artworks_deduped():
    state = AgentState(user_query="q", shown_artworks=["A"])
    merged = [
        _msg("semantic_search", [{"title": "A"}, {"title": "B", "author": "Monet"}]),
        _msg("exact_lookup", [{"title": "C"}]),
    ]
    updates = _ledger_updates(merged, state)
    assert updates["shown_artworks"] == ["A", "B", "C"]


def test_ledger_records_recommended_artists():
    state = AgentState(user_query="q")
    merged = [
        _msg("skill_art_recommendation",
             {"features": "f", "liked_artists": [], "candidates": [
                 {"author": "Rubens"}, {"author": "Caravaggio"}, {"author": "Rubens"},
             ]}),
    ]
    updates = _ledger_updates(merged, state)
    assert updates["recommended_artists"] == ["Rubens", "Caravaggio"]


def test_ledger_ignores_unrelated_tools():
    state = AgentState(user_query="q")
    merged = [_msg("web_search", [{"title": "web hit", "url": "x"}])]
    updates = _ledger_updates(merged, state)
    assert updates["shown_artworks"] == []
    assert updates["recommended_artists"] == []


def test_tool_round_cap_blocks_execution():
    ai = AIMessage(content="", tool_calls=[
        {"name": "exact_lookup", "args": {"author": "Monet"}, "id": "c1"},
    ])
    state = AgentState(user_query="q", tool_rounds=MAX_TOOL_ROUNDS, messages=[ai])
    with patch("src.agent.nodes.general._guarded_tool_calls") as mock:
        out = general_tools(state)
        mock.assert_not_called()
    assert "LIMIT" in str(out["messages"][0].content)
    assert out["tool_rounds"] == MAX_TOOL_ROUNDS


def test_tool_rounds_increment_on_normal_path():
    ai = AIMessage(content="", tool_calls=[
        {"name": "exact_lookup", "args": {"author": "Monet"}, "id": "c9"},
    ])
    state = AgentState(user_query="q", tool_rounds=2, messages=[ai])
    with patch("src.agent.nodes.general._guarded_tool_calls", return_value=([], ["sig1"])):
        out = general_tools(state)
    assert out["tool_rounds"] == 3
    assert out["executed_tool_signatures"] == ["sig1"]


def test_repeat_tool_call_blocked_by_guard():
    ai = AIMessage(content="", tool_calls=[
        {"name": "semantic_search", "args": {"query": "莫奈手稿"}, "id": "c1"},
    ])
    sig = 'semantic_search:{"query": "莫奈手稿"}'
    state = AgentState(user_query="q", executed_tool_signatures=[sig], messages=[ai])
    merged, new_sigs = _guarded_tool_calls(state, None)
    assert "REPEAT" in str(merged[0].content)
    assert "read_page_image" in str(merged[0].content)
    assert new_sigs == []


# ══════════════ 信息澄清 ══════════════
def test_short_query_is_gap():
    gap, msg = _info_gap("推荐", "recommendation")
    assert gap is True
    assert "具体" in msg


def test_recommendation_without_preference_signal_is_gap():
    gap, msg = _info_gap("给我推荐几幅画", "recommendation")
    assert gap is True
    assert "偏好" in msg


def test_recommendation_with_preference_passes():
    gap, _ = _info_gap("我喜欢浓烈奔放的风格，推荐几位画家", "recommendation")
    assert gap is False


def test_recommendation_with_style_word_not_gap():
    gap, _ = _info_gap("推荐几幅浓烈奔放的画", "recommendation")
    assert gap is False


def test_normal_queries_pass():
    assert _info_gap("对比莫奈和梵高的色彩", "comparison")[0] is False
    assert _info_gap("梳理伦勃朗的风格演变", "timeline")[0] is False
    assert _info_gap("梵高有哪些作品", "general")[0] is False


def test_ask_user_asks_and_short_circuits():
    state = AgentState(user_query="给我推荐几幅画", intent="recommendation")
    out = ask_user(state)
    assert out["ask_user"] == "ask"
    assert out["pending_clarification"]
    assert out["final_answer"] == out["pending_clarification"]


def test_ask_user_continues_when_info_sufficient():
    state = AgentState(
        user_query="我喜欢浓烈奔放的风格，推荐几位画家",
        intent="recommendation",
        pending_clarification="旧问题",
    )
    out = ask_user(state)
    assert out["ask_user"] == "continue"
    assert out["pending_clarification"] == ""


def test_rewrite_ambiguous_triggers_ask():
    state = AgentState(
        user_query="关于那幅画你了解吗", intent="general", rewrite_ambiguous=True,
    )
    out = ask_user(state)
    assert out["ask_user"] == "ask"
    assert "确定" in out["pending_clarification"]


def test_rewrite_not_ambiguous_passes_on_normal_question():
    state = AgentState(user_query="梵高有哪些作品", intent="general")
    out = ask_user(state)
    assert out["ask_user"] == "continue"


def test_ask_user_uses_original_query_for_gap_check():
    state = AgentState(user_query="莫奈晚年", original_user_query="他晚年怎么了？", intent="general")
    out = ask_user(state)
    assert out["ask_user"] == "continue"


# ══════════════ 工具治理 ══════════════
class _FakeTool:
    name = "fake"

    def __init__(self, result="ok", fail_times=0, sleep=0):
        self.result = result
        self.fail_times = fail_times
        self.sleep = sleep
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return self.result


def test_run_with_timeout_ok():
    assert governance.run_with_timeout(lambda: 42, 2) == 42


def test_run_with_timeout_raises():
    with pytest.raises(governance.ToolTimeout):
        governance.run_with_timeout(lambda: time.sleep(5), 0.2)


def test_governed_invoke_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("TOOL_RETRIES", "2")
    tool = _FakeTool(result="done", fail_times=2)
    assert governance.governed_invoke(tool, {}) == "done"
    assert tool.calls == 3


def test_governed_invoke_returns_error_json(monkeypatch):
    monkeypatch.setenv("TOOL_RETRIES", "0")
    tool = _FakeTool(fail_times=99)
    data = json.loads(governance.governed_invoke(tool, {}))
    assert data["status"] == "TOOL_ERROR"
    assert data["tool"] == "fake"


def test_governed_invoke_timeout_returns_error(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SEC", "0.2")
    monkeypatch.setenv("TOOL_RETRIES", "0")
    tool = _FakeTool(sleep=5)
    assert json.loads(governance.governed_invoke(tool, {}))["status"] == "TOOL_ERROR"


def test_truncate_payload_preserves_shape():
    payload = {"title": "short", "items": [{"a": "y" * 1000}, {"b": 2}]}
    shrunk = governance._truncate_payload(payload, limit=200)
    text = json.dumps(shrunk, ensure_ascii=False)
    assert len(text) <= 600
    assert "截断" in str(shrunk["items"][0]["a"])
    assert shrunk["items"][-1].get("truncated") is True


def test_governed_invoke_truncates_long_output(monkeypatch):
    monkeypatch.setenv("TOOL_OUTPUT_MAX_CHARS", "200")
    tool = _FakeTool(result="x" * 5000)
    assert len(governance.governed_invoke(tool, {})) <= 500


# ══════════════ 工具守卫 ══════════════
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


# ══════════════ 上下文工程 ══════════════
def _item(title, author="Monet", date="1872", score=0.8, aid=""):
    return {
        "artwork_id": aid,
        "title": title,
        "author": author,
        "date": date,
        "description_snippet": "d" * 300,
        "relevance_score": score,
    }


def test_dedup_by_artwork_id_keeps_highest_score():
    items = [
        _item("Water Lilies", aid="Q1", score=0.7),
        _item("Water Lilies", aid="Q1", score=0.9),
        _item("Other", aid="Q2", score=0.5),
    ]
    out = dedup_artworks(items)
    assert len(out) == 2
    assert out[0]["relevance_score"] == 0.9


def test_dedup_by_title_author_when_no_id():
    items = [_item("Water Lilies", score=0.7), _item("Water Lilies", score=0.8)]
    out = dedup_artworks(items)
    assert len(out) == 1
    assert out[0]["relevance_score"] == 0.8


def test_dedup_ignores_garbage_and_empty():
    assert dedup_artworks([]) == []
    assert dedup_artworks([None, "x", {}]) == []


def test_evidence_block_numbered_and_truncated_snippet():
    items = [_item("A", aid="Q1"), _item("B", aid="Q2")]
    block = format_numbered_evidence_block(items)
    assert "[1]" in block and "[2]" in block
    assert "..." in block


def test_evidence_block_respects_budget():
    items = [_item("A", aid="Q1"), _item("B", aid="Q2")]
    block = format_numbered_evidence_block(items, budget=20)
    assert len(block) <= 20
    assert "[2]" not in block


def test_multi_evidence_groups_by_subtask_with_global_numbering():
    grouped = {
        "对比莫奈和梵高": [_item("A", aid="Q1")],
        "推荐类似画": [_item("B", aid="Q2"), _item("A", aid="Q1")],
    }
    block = format_multi_evidence(grouped)
    assert "【子任务1】对比莫奈和梵高" in block
    assert "【子任务2】推荐类似画" in block
    assert "[1] A" in block
    assert "[2] B" in block
    assert block.count("[1] A") == 1


def test_multi_evidence_empty():
    assert format_multi_evidence({}) == ""


def test_profile_and_session_and_summary_blocks():
    assert "喜欢画家：Monet" in build_profile_block({"artists": ["Monet"]})
    assert "已推荐画家：Rubens" in build_session_block({"recommended_artists": ["Rubens"]})
    assert "summary" in build_summary_block("summary text")
    assert build_profile_block({}) == ""
    assert build_session_block({}) == ""
    assert build_summary_block("") == ""


def test_session_block_renders_uploaded_docs_with_image_hint():
    docs = [
        {"doc_name": "莫奈手稿", "pages": 16, "kind": "pdf",
         "text_chunks": 0, "image_pages": 16},
        {"doc_name": "画册", "pages": 4, "kind": "pdf",
         "text_chunks": 10, "image_pages": 0},
    ]
    block = build_session_block({"uploaded_docs": docs})
    assert "莫奈手稿" in block and "16页" in block
    assert "无文字索引" in block and "read_page_image" in block
    assert "画册" in block and "无文字索引" not in block.split("画册")[1]


def test_trim_history_keeps_system_and_recent():
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        ToolMessage(content="t", tool_call_id="c1"),
        AIMessage(content="a2"),
    ]
    out = trim_history(msgs, max_turns=1)
    assert out[0].content == "sys"
    assert len(out) == 3
    assert out[-1].content == "a2"


def test_estimate_context_chars_counts_body_and_history():
    msgs = [HumanMessage(content="abc"), AIMessage(content="def")]
    blocks = ContextBlocks(system="sys", summary="s" * 100, evidence="e" * 100, history=msgs)
    assert estimate_context_chars(blocks) == 209


def test_apply_budget_shrinks_history_when_over():
    msgs = []
    for i in range(30):
        msgs.append(HumanMessage(content=f"q{i}" + "x" * 300))
        msgs.append(AIMessage(content=f"a{i}" + "x" * 300))
    blocks = ContextBlocks(system="sys", summary="", evidence="", history=msgs)
    out = apply_budget(blocks, ContextBudget(total_chars=4000))
    assert estimate_context_chars(out) <= 4200
    humans = [m for m in out.history if m.type == "human"]
    assert len(humans) < 30


def test_apply_budget_keeps_min_turns():
    msgs = [HumanMessage(content="q" * 1000) for _ in range(20)]
    blocks = ContextBlocks(system="s", summary="", evidence="", history=msgs)
    out = apply_budget(blocks, ContextBudget(total_chars=100))
    humans = [m for m in out.history if m.type == "human"]
    assert len(humans) >= ContextBudget().history_min_turns


def test_apply_budget_truncates_text_blocks_not_system():
    blocks = ContextBlocks(system="keep-me", summary="x" * 3000,
                           evidence="y" * 6000, subtasks="z" * 4000, history=[])
    out = apply_budget(blocks, ContextBudget(total_chars=10**9))
    assert out.system == "keep-me"
    assert len(out.summary) <= 1200
    assert len(out.evidence) <= 4500
    assert len(out.subtasks) <= 3000


def test_extract_evidence_from_nested_tool_results():
    msgs = [
        ToolMessage(content=json.dumps([
            {"title": "A", "author": "Monet"},
            {"title": "B", "author": "van Gogh"},
        ]), name="semantic_search", tool_call_id="c1"),
        ToolMessage(content=json.dumps({
            "subject": "Monet", "query": "q", "evidence": [{"title": "C", "author": "Monet"}],
        }), name="compare_subjects", tool_call_id="c2"),
        ToolMessage(content=json.dumps({
            "periods": [{"period": "p", "evidence": [{"title": "D", "author": "Rembrandt"}]}],
        }), name="timeline_by_periods", tool_call_id="c3"),
        ToolMessage(content='{"status": "FAILED", "message": "bad"}', name="exact_lookup",
                    tool_call_id="c4"),
    ]
    out = extract_evidence_from_messages(msgs)
    titles = {d["title"] for d in out}
    assert titles == {"A", "B", "C", "D"}


def test_condense_tool_messages_compresses_long_json():
    long_items = json.dumps([{"title": "x", "author": "y"}] * 50)
    msgs = [
        ToolMessage(content=long_items, name="semantic_search", tool_call_id="c1", id="m1"),
        ToolMessage(content="tool execution error", name="web_search", tool_call_id="c2"),
    ]
    out = condense_tool_messages(msgs, limit=100)
    assert len(str(out[0].content)) <= 140
    assert "evidence" in str(out[0].content)
    assert out[0].tool_call_id == "c1" and out[0].id == "m1"
    assert out[1].content == "tool execution error"


# ══════════════ 查询改写 ══════════════
def test_normalize_query_strips_quotes():
    assert normalize_query('  "这幅画"  ') == "这幅画"
    assert normalize_query("“星月夜”") == "星月夜"
    assert normalize_query("") == ""


def test_rewrite_enabled_default_and_toggle():
    assert rewrite_enabled() is True
    os.environ["REWRITE_ENABLED"] = "0"
    try:
        assert rewrite_enabled() is False
    finally:
        os.environ.pop("REWRITE_ENABLED", None)
    assert rewrite_enabled() is True


def test_llm_rewrite_and_split_success():
    def fake_llm(prompt):
        assert "最新问题" in prompt
        return (
            '{"rewritten_question": "对比莫奈和梵高的色彩，'
            '并推荐类似的画", "sub_questions": ["对比莫奈和梵高的色彩", '
            '"推荐几幅类似莫奈的风景画"]}'
        )

    result = rewrite_and_split("对比莫奈和梵高，顺便推荐类似的画", llm=fake_llm)
    assert result.rewritten_question.startswith("对比莫奈和梵高的色彩")
    assert len(result.sub_questions) == 2


def test_llm_failure_falls_back_to_normalized():
    def boom(prompt):
        raise RuntimeError("llm down")

    result = rewrite_and_split("  找梵高的画  ", llm=boom)
    assert result == RewriteResult("找梵高的画", ["找梵高的画"])


def test_malformed_json_falls_back():
    result = rewrite_and_split("找莫奈的画", llm=lambda p: "not json")
    assert result.rewritten_question == "找莫奈的画"
    assert result.sub_questions == ["找莫奈的画"]


def test_empty_sub_questions_becomes_rewritten():
    raw = '{"rewritten_question": "什么是巴洛克", "sub_questions": []}'
    result = rewrite_and_split("什么是巴洛克", llm=lambda p: raw)
    assert result.rewritten_question == "什么是巴洛克"
    assert result.sub_questions == ["什么是巴洛克"]


def test_key_entities_and_ambiguous_parsed():
    raw = (
        '{"rewritten_question": "莫奈的睡莲有哪些", "sub_questions": [], '
        '"key_entities": ["Monet", "Water Lilies"], "ambiguous": true}'
    )
    result = rewrite_and_split("就是那个，莫奈的睡莲，你懂的", llm=lambda p: raw)
    assert result.key_entities == ["Monet", "Water Lilies"]
    assert result.ambiguous is True


def test_missing_new_fields_default_safe():
    raw = '{"rewritten_question": "什么是巴洛克", "sub_questions": []}'
    result = rewrite_and_split("什么是巴洛克", llm=lambda p: raw)
    assert result.key_entities == []
    assert result.ambiguous is False


def test_over_compressed_rewrite_falls_back_to_original():
    def fake_llm(prompt):
        return (
            '{"rewritten_question": "莫奈晚年", "sub_questions": ["莫奈晚年"], '
            '"key_entities": ["莫奈"], "ambiguous": false}'
        )

    result = rewrite_and_split("他晚年怎么了？", llm=fake_llm)
    assert result.rewritten_question == "他晚年怎么了？"
    assert result.sub_questions == ["他晚年怎么了？"]


def test_rewrite_prompt_asks_compression_and_extraction():
    def fake_llm(prompt):
        assert "压缩" in prompt or "去掉口头禅" in prompt
        assert "key_entities" in prompt
        assert "ambiguous" in prompt
        return ('{"rewritten_question": "q", "sub_questions": [], '
                '"key_entities": [], "ambiguous": false}')

    rewrite_and_split("就是那个，嗯，你懂的", llm=fake_llm)


def test_history_only_keeps_last_two_turns():
    history = [
        HumanMessage(content="第1轮：找梵高的画"),
        AIMessage(content="第1轮回答"),
        HumanMessage(content="第2轮：这幅画呢"),
        AIMessage(content="第2轮回答"),
        HumanMessage(content="第3轮：它现在在哪里"),
    ]

    def fake_llm(prompt):
        assert "第1轮：找梵高的画" not in prompt
        assert "第3轮" in prompt
        assert "第2轮" in prompt
        return '{"rewritten_question": "《星月夜》现在收藏在哪里？", "sub_questions": []}'

    result = rewrite_and_split("它现在在哪里", history, llm=fake_llm)
    assert "《星月夜》" in result.rewritten_question


def test_disabled_skips_llm():
    os.environ["REWRITE_ENABLED"] = "0"
    try:
        def boom(prompt):
            raise AssertionError("关闭后不应调用 LLM")

        result = rewrite_and_split(" 找伦勃朗的画 ", llm=boom)
        assert result.rewritten_question == "找伦勃朗的画"
    finally:
        os.environ.pop("REWRITE_ENABLED", None)


# ══════════════ 可观测轨迹 ══════════════
def test_record_and_list():
    rid = runs_mod.record_run(
        request_id="r1", session_id="s1", intent="comparison",
        steps=[{"node": "classify"}], tools=["semantic_search", "exact_lookup"],
        context_chars=4000, tool_rounds=2, latency_ms=1234.5,
        final_answer_len=200, reflection_triggered=True, web_fallback=True,
    )
    assert rid >= 1
    rows = runs_mod.list_runs()
    assert len(rows) == 1
    assert rows[0]["intent"] == "comparison"
    assert rows[0]["tools"] == ["semantic_search", "exact_lookup"]
    assert rows[0]["latency_ms"] == 1234.5


def test_metrics_summary(monkeypatch):
    monkeypatch.setenv("COST_PER_1K_INPUT_TOKENS", "1.0")
    monkeypatch.setenv("COST_PER_1K_OUTPUT_TOKENS", "2.0")
    for i in range(10):
        runs_mod.record_run(
            session_id=f"s{i}", intent="general", tools=["web_search"],
            context_chars=1000, tool_rounds=1, latency_ms=float(i * 10),
            final_answer_len=100, web_fallback=(i % 2 == 0),
            error=("x" if i == 0 else ""),
        )
    m = runs_mod.metrics(limit=10)
    assert m["count"] == 10
    assert m["latency_ms"]["p50"] == pytest.approx(45.0, abs=1.0)
    assert m["latency_ms"]["p95"] >= 85.0
    assert m["web_fallback_rate"] == pytest.approx(0.5)
    assert m["error_rate"] == pytest.approx(0.1)
    assert m["tool_calls"].get("web_search") == 10
    assert m["est_cost_total"] > 0


# ══════════════ core 数据源集成 ══════════════
def test_retrieval_source_accepts_core():
    r = RetrievalResult(content="x", source="core", metadata={"title": "t"})
    assert r.source == "core"


def test_format_result_core_shape():
    meta = {
        "title": "The Bedroom",
        "artist": "Vincent van Gogh",
        "year_display": "1889",
        "year": 1889,
        "material": "Oil on canvas",
        "movement": "Post-Impressionism",
        "year_bucket": "1851-1900",
        "image_url": "https://www.artic.edu/iiif/2/x/full/843,/0/default.jpg",
        "description": "A" * 300,
    }
    out = _format_result(RetrievalResult(content="c", source="core", score=0.5, metadata=meta))
    assert out["title"] == "The Bedroom"
    assert out["author"] == "Vincent van Gogh"
    assert out["date"] == "1889"
    assert out["technique"] == "Oil on canvas"
    assert out["school"] == "Post-Impressionism"
    assert out["timeframe"] == "1851-1900"
    assert out["image_file"] == meta["image_url"]
    assert len(out["description_snippet"]) <= 203
    assert out["relevance_score"] == 0.5
    assert "source" not in out


def test_thumb_url_url_passthrough():
    assert service_mod._thumb_url("https://example.com/a.jpg") == "https://example.com/a.jpg"
    assert service_mod._thumb_url("http://example.com/a.jpg") == "http://example.com/a.jpg"
    assert service_mod._thumb_url("") == ""


def test_thumb_url_local_file_becomes_api_url():
    assert service_mod._thumb_url("28496-early05.jpg") == "/api/images/28496-early05.jpg"
    assert service_mod._thumb_url("../evil/../x.jpg") == "/api/images/x.jpg"


def test_get_structured_retriever_core_missing_raises():
    old = sr.CORE_DATA_PATH
    sr.CORE_DATA_PATH = Path("C:/nonexistent/artworks_core.csv")
    sr._REGISTRY.pop("core", None)
    try:
        try:
            sr.get_structured_retriever("core")
            raise AssertionError("数据缺失时应抛 KeyError")
        except KeyError as e:
            assert "核心库数据未就绪" in str(e)
    finally:
        sr.CORE_DATA_PATH = old
        sr._REGISTRY.pop("core", None)


def test_get_structured_retriever_core_registers_from_csv():
    old = sr.CORE_DATA_PATH
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "artworks_core.csv"
        pd.DataFrame([{
            "artwork_id": "Q1", "title": "T", "artist_qid": "Q2", "artist_name": "A",
            "collection_name": "", "location": "", "inception": "", "year": 1800,
            "year_bucket": "1776-1825", "material": "", "genre": "", "movement": "",
            "series": "", "description": "d", "image_url": "", "license": "",
            "dimensions_raw": "", "width_cm": "", "height_cm": "", "source_api": "wikidata",
            "dedup_key": "a|t|1800",
        }]).to_csv(csv_path, index=False, encoding="utf-8-sig")
        sr.CORE_DATA_PATH = csv_path
        sr._REGISTRY.pop("core", None)
        try:
            retriever = sr.get_structured_retriever("core")
            assert retriever.dataset_id == "core"
            assert retriever.schema.supports_timeline is True
            assert retriever.schema.supports_recommendation is True
            assert retriever.source == "core"
        finally:
            sr.CORE_DATA_PATH = old
            sr._REGISTRY.pop("core", None)
