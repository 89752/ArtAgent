"""上下文工程纯函数单测：去重、编号引用、预算、历史窗口。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    assert "...]" in block or "..." in block  # 长 snippet 被截断


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
    # 全局编号：第二组从 [2] 开始；重复的 A 在第二组被去重
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
    assert len(out) == 3  # system + 最近 1 轮（2 条）
    assert out[-1].content == "a2"


def test_estimate_context_chars_counts_body_and_history():
    msgs = [HumanMessage(content="abc"), AIMessage(content="def")]
    blocks = ContextBlocks(system="sys", summary="s" * 100, evidence="e" * 100,
                           history=msgs)
    assert estimate_context_chars(blocks) == 209  # 3 + 100 + 100 + 6


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
    import json

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
    assert titles == {"A", "B", "C", "D"}  # 守卫消息不产生证据


def test_condense_tool_messages_compresses_long_json():
    import json

    long_items = json.dumps([{"title": "x", "author": "y"}] * 50)
    msgs = [
        ToolMessage(content=long_items, name="semantic_search", tool_call_id="c1", id="m1"),
        ToolMessage(content="tool execution error", name="web_search", tool_call_id="c2"),
    ]
    out = condense_tool_messages(msgs, limit=100)
    assert len(str(out[0].content)) <= 140
    assert "evidence" in str(out[0].content)
    assert out[0].tool_call_id == "c1" and out[0].id == "m1"
    assert out[1].content == "tool execution error"  # 非 JSON 原样保留


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] context 全部 {len(fns)} 个单测通过")
