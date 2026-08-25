"""Agent 层统一单测：图结构 / 节点接线 / 澄清 / 治理 / 守卫 / 上下文 / 改写 / 集成。

纯函数级测试，不触发真实 LLM/网络/模型加载。
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
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
    trim_history,
)
from src.agent.graph import get_graph
from src.agent.nodes.common import _info_gap, ask_user, classify_intent
from src.agent.nodes.general import MAX_TOOL_ROUNDS, _guarded_tool_calls, _ledger_updates, general_tools
from src.agent.state import AgentState
from src.observability import runs as runs_mod
from src.retrieval import structured_retriever as sr
from src.retrieval.agentic import adaptive_retrieve, coverage_check
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


def test_general_react_has_no_special_recommendation_tool():
    from src.tools.registry import TOOL_BY_NAME

    assert "recommendation_search" not in TOOL_BY_NAME
    assert "semantic_search" in TOOL_BY_NAME


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


def test_ledger_ignores_unrelated_tools():
    state = AgentState(user_query="q")
    merged = [_msg("web_search", [{"title": "web hit", "url": "x"}])]
    updates = _ledger_updates(merged, state)
    assert updates["shown_artworks"] == []


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


def test_tool_call_cap_blocks_parameter_churn():
    ai = AIMessage(content="", tool_calls=[
        {"name": "semantic_search", "args": {"query": "莫奈晚年"}, "id": "c1"},
    ])
    state = AgentState(
        user_query="q",
        executed_tool_signatures=[
            'semantic_search:{"query": "莫奈视力"}',
            'semantic_search:{"query": "莫奈白内障"}',
        ],
        messages=[ai],
    )
    merged, new_sigs = _guarded_tool_calls(state, None)
    assert "TOOL_LIMIT" in str(merged[0].content)
    assert new_sigs == []


def test_memory_recall_call_cap_blocks_parameter_churn():
    ai = AIMessage(content="", tool_calls=[
        {"name": "recall", "args": {"query": "巴洛克"}, "id": "c1"},
    ])
    state = AgentState(
        user_query="q",
        executed_tool_signatures=[
            'recall:{"query": "喜欢巴洛克"}',
            'recall:{"query": "巴洛克风格"}',
        ],
        messages=[ai],
    )
    merged, new_sigs = _guarded_tool_calls(state, None)
    assert "TOOL_LIMIT" in str(merged[0].content)
    assert new_sigs == []


def test_explicit_forget_request_is_confirmed_for_governance(monkeypatch):
    ai = AIMessage(content="", tool_calls=[
        {"name": "forget", "args": {"entity": "巴洛克"}, "id": "c1"},
    ])
    state = AgentState(user_query="忘掉我喜欢巴洛克风格", messages=[ai])
    seen = {}

    def invoke(_tool, _args, *, context="main", user_id=""):
        seen["context"] = context
        seen["user_id"] = user_id
        return "已删除 1 条"

    monkeypatch.setattr("src.utils.governance.governed_invoke", invoke)
    out = general_tools(state)
    assert seen["context"] == "confirmed"
    assert seen["user_id"] == ""
    assert "已删除" in str(out["messages"][0].content)


@pytest.mark.parametrize("query_text", ["忘掉我喜欢巴洛克风格这件事", "遗忘我喜欢巴洛克风格这件事"])
def test_explicit_forget_is_forced_before_a_recall_loop(query_text):
    from src.agent.nodes.general import _enforce_memory_forget

    response = AIMessage(content="", tool_calls=[
        {"name": "recall", "args": {"query": "巴洛克"}, "id": "c1"},
    ])
    state = AgentState(
        user_query=query_text,
        memory_items=[{"entity": "巴洛克", "content": "我喜欢巴洛克风格"}],
    )
    forced = _enforce_memory_forget(response, state)
    assert forced.tool_calls[0]["name"] == "forget"
    assert forced.tool_calls[0]["args"] == {"entity": "巴洛克"}


def test_forget_all_related_information_overrides_a_single_item_id():
    from src.agent.nodes.general import _enforce_memory_forget

    response = AIMessage(content="", tool_calls=[
        {"name": "forget", "args": {"item_id": "mem_profile"}, "id": "c1"},
    ])
    state = AgentState(user_query="请遗忘我关于巴洛克的一切偏好和相关信息")

    forced = _enforce_memory_forget(response, state)

    assert forced.tool_calls == [
        {"name": "forget", "args": {"entity": "巴洛克"}, "id": forced.tool_calls[0]["id"], "type": "tool_call"}
    ]


def test_a_previous_turn_forget_does_not_suppress_a_new_forget_request():
    from src.agent.nodes.general import _enforce_memory_forget

    response = AIMessage(content="")
    state = AgentState(
        user_query="请遗忘我关于巴洛克的一切信息",
        messages=[ToolMessage(content="已删除", name="forget", tool_call_id="old")],
    )

    forced = _enforce_memory_forget(response, state)

    assert forced.tool_calls[0]["args"] == {"entity": "巴洛克"}


def test_explicit_uploaded_pdf_only_query_scopes_semantic_search():
    from src.agent.nodes.general import _scope_uploaded_pdf_search

    scoped = _scope_uploaded_pdf_search(
        {"name": "semantic_search", "args": {"query": "莫奈"}, "id": "c1"},
        "请仅根据我上传的 PDF 回答",
    )

    assert scoped["args"]["filters"] == {"source": "user_pdf_text"}


def test_existing_semantic_source_filter_is_not_overridden():
    from src.agent.nodes.general import _scope_uploaded_pdf_search

    original = {"name": "semantic_search", "args": {"filters": {"source": "core"}}}
    assert _scope_uploaded_pdf_search(original, "请仅根据我上传的 PDF 回答") is original


# ══════════════ 信息澄清 ══════════════
def test_short_query_is_gap():
    gap, msg = _info_gap("推荐")
    assert gap is True
    assert "具体" in msg


def test_open_ended_preference_query_passes_without_special_clarification():
    gap, _ = _info_gap("给我推荐几幅画")
    assert gap is False
    gap, _ = _info_gap("推荐几幅浓烈奔放的画")
    assert gap is False


def test_normal_queries_pass():
    assert _info_gap("对比莫奈和梵高的色彩")[0] is False
    assert _info_gap("梳理伦勃朗的风格演变")[0] is False
    assert _info_gap("梵高有哪些作品")[0] is False


def test_classify_intent_keywords():
    assert classify_intent("对比莫奈和梵高的色彩") == "comparison"
    assert classify_intent("莫奈和梵高谁更擅长光影") == "comparison"
    assert classify_intent("梳理透纳的风格演变") == "timeline"
    assert classify_intent("我喜欢浓烈奔放的风格，推荐几位画家") == "general"
    assert classify_intent("还有哪些作品采用了类似的光影处理") == "general"
    assert classify_intent("梵高有哪些作品") == "general"


def test_ask_user_passes_open_ended_preference_request():
    state = AgentState(user_query="给我推荐几幅画")
    out = ask_user(state)
    assert out["ask_user"] == "continue"
    assert out["pending_clarification"] == ""
    assert out["intent"] == "general"


def test_ask_user_passes_preference_request_with_loaded_memory():
    state = AgentState(
        user_query="给我推荐一幅画",
        user_preferences={"preferences": ["用户偏好巴洛克的强烈明暗对照"]},
        memory_items=[{"kind": "preference", "content": "用户偏好巴洛克的强烈明暗对照"}],
    )
    out = ask_user(state)
    assert out["ask_user"] == "continue"


def test_ask_user_continues_when_info_sufficient():
    state = AgentState(
        user_query="我喜欢浓烈奔放的风格，推荐几位画家",
        pending_clarification="旧问题",
    )
    out = ask_user(state)
    assert out["ask_user"] == "continue"
    assert out["pending_clarification"] == ""
    assert out["intent"] == "general"


def test_ask_user_writes_derived_intent():
    state = AgentState(user_query="我喜欢莫奈，推荐几幅类似的画")
    out = ask_user(state)
    assert out["intent"] == "general"
    assert out["ask_user"] == "continue"


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
    tool.name = "semantic_search"
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
    assert json.loads(governance.governed_invoke(tool, {}))["status"] == "UNKNOWN_EXECUTION_STATE"


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


def test_profile_and_session_and_summary_blocks():
    assert "喜欢画家：Monet" in build_profile_block({"artists": ["Monet"]})
    assert "已展示画作：Water Lilies" in build_session_block({"shown_artworks": ["Water Lilies"]})
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


def test_trace_uses_provider_usage_and_redacts_tool_arguments():
    run_id = runs_mod.record_run(
        user_id="trace-owner",
        session_id="s1",
        model_calls=[{
            "model": "test-model", "input_tokens": 12, "output_tokens": 8,
            "total_tokens": 20, "token_source": "provider",
        }],
        tool_calls=[{
            "tool_name": "web_search", "args": {"api_key": "secret", "query": "Monet"},
        }],
    )
    detail = runs_mod.get_run_detail(run_id, "trace-owner")
    assert detail is not None
    assert detail["token_source"] == "provider"
    assert detail["input_tokens"] == 12
    assert detail["tool_calls"][0]["args"]["api_key"] == "[REDACTED]"
    assert runs_mod.get_run_detail(run_id, "another-user") is None


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
            assert retriever.source == "core"
        finally:
            sr.CORE_DATA_PATH = old
            sr._REGISTRY.pop("core", None)


# ══════════════ P2: 多模型路由 / Agentic RAG ══════════════
def test_complex_query_routes_to_reasoning_model():
    class FakeModel:
        def bind_tools(self, tools):
            return self

    with patch("src.agent.nodes.general.get_reasoning_llm", return_value=FakeModel()) as reasoning, patch(
        "src.agent.nodes.general.get_deterministic_llm", return_value=FakeModel()
    ) as main:
        from src.agent.nodes.general import _get_llm_with_tools

        _get_llm_with_tools("请对比莫奈和透纳的风格演变，并给出作品证据")
        _get_llm_with_tools("你好")
    reasoning.assert_called_once()
    main.assert_called_once()


def test_simple_recommendation_uses_main_but_compound_request_uses_reasoning():
    from src.agent.nodes.general import _model_role_for_query

    assert _model_role_for_query("我喜欢宁静的印象派风景，推荐三幅作品") == "main"
    assert _model_role_for_query("推荐三幅作品，并比较它们与莫奈的色彩差异") == "reasoning"


def test_agentic_retrieval_rewrites_once_and_merges_evidence():
    calls = []

    def retrieve(query):
        calls.append(query)
        if len(calls) == 1:
            return [{"source": "core", "title": "莫奈", "description_snippet": "色彩"}]
        return [
            {"source": "core", "title": "莫奈", "description_snippet": "色彩"},
            {"source": "core", "title": "睡莲", "description_snippet": "莫奈的光影色彩技巧"},
            {"source": "core", "title": "日出", "description_snippet": "印象派光影"},
        ]

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="莫奈 光影 色彩 技巧")

    evidence, audit = adaptive_retrieve("莫奈的色彩与光影技巧", retrieve, llm=FakeLLM())
    assert len(calls) == 2
    assert audit["rewritten"] is True
    assert len(evidence) == 3  # duplicate Monet evidence is preserved only once
    assert coverage_check("莫奈的色彩与光影技巧", evidence)["evidence_count"] == 3
