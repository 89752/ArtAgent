# tests/test_relevance.py
"""
相关性校正（src/retrieval/relevance.py + general 分支 general_tools 节点）纯单测：
注入 fake LLM 验证过滤/兜底/降级逻辑与 ToolMessage 重写，不加载数据集、
不调真实 LLM、不联网，秒级完成。
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.nodes import general as general_mod
from src.retrieval import relevance as relevance_mod
from src.retrieval.relevance import _filter_enabled, llm_relevance_filter


def _items(n=4):
    return [
        {"title": f"Work {i}", "description_snippet": f"snippet {i}"} for i in range(n)
    ]


class _FakeLLM:
    """按预设内容应答的假 LLM；called 记录是否被调用过。"""

    def __init__(self, content="[0, 1]", error=None):
        self.content = content
        self.error = error
        self.called = False

    def invoke(self, prompt):
        self.called = True
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def _patch(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    return old


# ── 开关 ─────────────────────────────────────────────────────────
def test_enabled_override_wins_over_env():
    os.environ["RELEVANCE_FILTER_ENABLED"] = "0"
    try:
        assert _filter_enabled(True) is True
        assert _filter_enabled(None) is False
    finally:
        os.environ.pop("RELEVANCE_FILTER_ENABLED", None)
    assert _filter_enabled(None) is True  # 默认开


def test_disabled_returns_same_object_without_llm():
    llm = _FakeLLM()
    items = _items()
    out = llm_relevance_filter("q", items, llm=llm, enabled=False)
    assert out is items and not llm.called


# ── 免过滤情形 ───────────────────────────────────────────────────
def test_small_list_passthrough_without_llm():
    llm = _FakeLLM()
    items = _items(2)
    out = llm_relevance_filter("q", items, min_keep=2, llm=llm)
    assert out is items and not llm.called  # 候选 ≤ min_keep 无可过滤


# ── 正常过滤 ─────────────────────────────────────────────────────
def test_filters_to_llm_selection_in_original_order():
    items = _items(4)
    # LLM 逆序选 [2, 0]：结果必须保持原相对顺序 [0, 2]（只删不重排）
    out = llm_relevance_filter("q", items, llm=_FakeLLM("[2, 0]"))
    assert [d["title"] for d in out] == ["Work 0", "Work 2"]


def test_markdown_wrapped_json_accepted():
    items = _items(3)
    out = llm_relevance_filter("q", items, llm=_FakeLLM("```json\n[1]\n```"), min_keep=1)
    assert [d["title"] for d in out] == ["Work 1"]


def test_garbage_and_out_of_range_indices_ignored():
    items = _items(4)
    # 只有 1 有效；bool/字符串/越界/负数全忽略 → 选中 [1] 后兜底补足 min_keep
    out = llm_relevance_filter("q", items, llm=_FakeLLM('[true, "x", 99, -1, 1]'), min_keep=2)
    assert [d["title"] for d in out] == ["Work 0", "Work 1"]


def test_llm_rejects_all_falls_back_to_min_keep():
    items = _items(4)
    out = llm_relevance_filter("q", items, llm=_FakeLLM("[]"), min_keep=2)
    assert [d["title"] for d in out] == ["Work 0", "Work 1"]  # 永不返回空证据


def test_beyond_max_candidates_passes_through():
    items = _items(5)
    out = llm_relevance_filter(
        "q", items, max_candidates=3, min_keep=1, llm=_FakeLLM("[0]")
    )
    # 前 3 个参与过滤（只留 0），尾部 2 个原样透传
    assert [d["title"] for d in out] == ["Work 0", "Work 3", "Work 4"]


def test_user_pdf_image_always_preserved():
    """整页图结果文本 snippet 只是占位标题，不能靠 LLM 判断相关性，必须保留。"""
    items = [
        {"title": "《手稿》第1页（整页图）", "description_snippet": "[整页图]", "source": "user_pdf_image"},
        {"title": "《手册》第2页（整页图）", "description_snippet": "[整页图]", "source": "user_pdf_image"},
        {"title": "相关画作", "description_snippet": "一幅风景画"},
        {"title": "无关画作", "description_snippet": "另一幅静物画"},
    ]
    # LLM 只选文本候选中的 [3]，但整页图必须原样保留
    out = llm_relevance_filter("手稿内容", items, min_keep=1, llm=_FakeLLM("[3]"))
    sources = [d.get("source") for d in out]
    assert sources.count("user_pdf_image") == 2
    assert out[0]["source"] == "user_pdf_image"
    assert out[1]["source"] == "user_pdf_image"


# ── 降级 ─────────────────────────────────────────────────────────
def test_invalid_json_returns_original():
    items = _items()
    out = llm_relevance_filter("q", items, llm=_FakeLLM("这不是JSON"))
    assert out is items


def test_non_list_json_returns_original():
    items = _items()
    out = llm_relevance_filter("q", items, llm=_FakeLLM('{"keep": [0]}'))
    assert out is items


def test_llm_exception_returns_original():
    items = _items()
    out = llm_relevance_filter("q", items, llm=_FakeLLM(error=RuntimeError("boom")))
    assert out is items


# ── general_tools 节点：ToolMessage 过滤 ─────────────────────────
def _make_state_with_search_call():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "semantic_search", "args": {"query": "星空 画作"}, "id": "call-1"},
            {"name": "exact_lookup", "args": {"author": "Monet"}, "id": "call-2"},
        ],
    )
    return AgentState(user_query="找星空的画", messages=[HumanMessage(content="找星空的画"), ai])


def _make_tool_output(search_items):
    return {
        "messages": [
            ToolMessage(content=json.dumps(search_items, ensure_ascii=False),
                        name="semantic_search", tool_call_id="call-1", id="m1"),
            ToolMessage(content=json.dumps([{"title": "Monet"}], ensure_ascii=False),
                        name="exact_lookup", tool_call_id="call-2", id="m2"),
        ]
    }


def test_general_tools_filters_semantic_search_message():
    items = _items(4)
    state = _make_state_with_search_call()

    class _StubToolNode:
        def invoke(self, s):
            return _make_tool_output(items)

    # 假过滤器：只留第 0、2 条（验证节点把 query 传对、消息被重写）
    seen = {}

    def _fake_filter(query, got_items, min_keep=2):
        seen["query"] = query
        return [got_items[0], got_items[2]]

    old_node = _patch(general_mod, "_tool_node", _StubToolNode())
    old_filter = _patch(general_mod, "llm_relevance_filter", _fake_filter)
    try:
        out = general_mod.general_tools(state)
        msgs = out["messages"]
        kept = json.loads(msgs[0].content)
        assert [d["title"] for d in kept] == ["Work 0", "Work 2"]
        assert seen["query"] == "星空 画作"  # query 取自 tool_call args
        assert msgs[0].tool_call_id == "call-1" and msgs[0].id == "m1"  # 身份字段保留
        # 非 semantic_search 的消息原样不动
        assert json.loads(msgs[1].content) == [{"title": "Monet"}]
    finally:
        _patch(general_mod, "_tool_node", old_node)
        _patch(general_mod, "llm_relevance_filter", old_filter)


def test_general_tools_untouched_when_no_search_call():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "exact_lookup", "args": {"author": "Monet"}, "id": "c9"}],
    )
    state = AgentState(user_query="莫奈的画", messages=[HumanMessage(content="莫奈的画"), ai])
    payload = {"messages": [ToolMessage(content="[]", name="exact_lookup", tool_call_id="c9", id="m9")]}

    class _StubToolNode:
        def invoke(self, s):
            return payload

    old_node = _patch(general_mod, "_tool_node", _StubToolNode())

    def _boom(*a, **kw):
        raise AssertionError("无 semantic_search 调用不应触发过滤")

    old_filter = _patch(general_mod, "llm_relevance_filter", _boom)
    try:
        out = general_mod.general_tools(state)
        assert out["messages"][0].content == "[]"
    finally:
        _patch(general_mod, "_tool_node", old_node)
        _patch(general_mod, "llm_relevance_filter", old_filter)


def test_filter_search_message_tolerates_non_json_content():
    msg = ToolMessage(content="tool execution error", name="semantic_search",
                      tool_call_id="c1", id="m1")
    out = general_mod._filter_search_message(msg, "q")
    assert out is msg  # 非 JSON 内容原样返回


def test_filter_search_message_skips_when_nothing_dropped():
    items = _items(2)  # ≤ min_keep，过滤器原样返回
    msg = ToolMessage(content=json.dumps(items), name="semantic_search",
                      tool_call_id="c1", id="m1")
    old_filter = _patch(general_mod, "llm_relevance_filter", lambda q, xs, min_keep=2: xs)
    try:
        assert general_mod._filter_search_message(msg, "q") is msg  # 无删减不重序列化
    finally:
        _patch(general_mod, "llm_relevance_filter", old_filter)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 relevance 全部 {len(fns)} 个单测通过！")
