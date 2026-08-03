"""多意图并行检索编排器单测：单子任务放行、多子任务并行分组、失败隔离。"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes.common import multi_retrieve
from src.agent.state import AgentState


def _state(sub_questions):
    return AgentState(user_query=" | ".join(sub_questions), sub_questions=sub_questions)


def test_single_sub_question_passes_through_without_search():
    mock_tool = MagicMock()
    with patch("src.tools.retrieval.semantic_search", mock_tool):
        out = multi_retrieve(_state(["对比莫奈和梵高的色彩"]))
        mock_tool.invoke.assert_not_called()
    assert out["multi_evidence"] == {}


def test_multi_sub_questions_parallel_and_grouped():
    def fake_invoke(args):
        q = args.get("query", "")
        return [{"title": q[:4], "author": "A", "description_snippet": "s"}]

    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = fake_invoke
    with patch("src.tools.retrieval.semantic_search", mock_tool):
        out = multi_retrieve(_state(["子问题一：莫奈", "子问题二：梵高"]))
    assert mock_tool.invoke.call_count == 2
    assert set(out["multi_evidence"].keys()) == {"子问题一：莫奈", "子问题二：梵高"}
    assert out["multi_evidence"]["子问题一：莫奈"][0]["title"] == "子问题一"


def test_one_sub_question_failure_does_not_break_others():
    calls = 0

    def fake_invoke(args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return [{"title": "ok", "author": "A"}]

    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = fake_invoke
    with patch("src.tools.retrieval.semantic_search", mock_tool):
        out = multi_retrieve(_state(["坏子问题", "好子问题"]))
    assert out["multi_evidence"]["坏子问题"] == []
    assert len(out["multi_evidence"]["好子问题"]) == 1


def test_parallelism_is_actually_concurrent():
    """两个子问题应并发执行（总耗时接近单次而非两次串行）。"""
    started = {"n": 0}
    gate = {"released": False}

    def slow_invoke(args):
        started["n"] += 1
        if started["n"] == 2:
            gate["released"] = True
        # 第一个子任务等待第二个开始，证明并发；最多等 2 秒
        deadline = time.time() + 2
        while not gate["released"] and time.time() < deadline:
            time.sleep(0.01)
        return [{"title": "x", "author": "A"}]

    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = slow_invoke
    with patch("src.tools.retrieval.semantic_search", mock_tool):
        t0 = time.time()
        multi_retrieve(_state(["a", "b"]))
        elapsed = time.time() - t0
    assert elapsed < 1.5  # 并发：两个任务同时跑，不等 2 秒串行


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] multi_retrieve 全部 {len(fns)} 个单测通过")
