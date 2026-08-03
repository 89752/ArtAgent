"""RAG 开关与直接回答节点单测（不触发真实 LLM/网络）。"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes.common import _rag_gate, direct_answer, rag_gate
from src.agent.state import AgentState


def _scores(kind, leaf_id, score):
    return [{"id": leaf_id, "kind": kind, "score": score, "tool_name": None,
             "path": f"{kind} > {leaf_id}", "reason": ""}]


def test_greeting_high_score_closes_rag():
    assert _rag_gate("你好", _scores("system", "system_greeting", 0.95)) is False


def test_greeting_deterministic_regardless_of_scores():
    # 不依赖 LLM 打分：即便没有 intent_scores 也能识别寒暄
    for q in ("你好", "你好呀！", "谢谢", "你是谁", "hello", "Hi", "再见"):
        assert _rag_gate(q, []) is False, q


def test_greeting_with_substance_keeps_rag():
    assert _rag_gate("你好，帮我查梵高的画", []) is True


def test_normal_questions_keep_rag_on():
    assert _rag_gate("梵高有哪些作品", _scores("capability", "general", 0.95)) is True
    assert _rag_gate("对比莫奈和梵高", _scores("capability", "comparison", 0.9)) is True


def test_low_system_score_on_substantive_question_keeps_rag():
    assert _rag_gate("梵高有哪些作品", _scores("system", "system_greeting", 0.3)) is True


def test_empty_input_keeps_rag():
    assert _rag_gate("", []) is True
    assert _rag_gate("问题", []) is True


def test_rag_gate_node_sets_flag():
    state = AgentState(user_query="你好", intent_scores=_scores("system", "system_greeting", 0.9))
    out = rag_gate(state)
    assert out["rag_needed"] is False


def test_direct_answer_uses_plain_llm():
    class FakeResp:
        content = "你好！有什么可以帮你？"

    with patch("src.utils.llm.get_deterministic_llm") as mock:
        mock.return_value.invoke.return_value = FakeResp()
        out = direct_answer(AgentState(user_query="你好"))
    assert "你好" in out["final_answer"]
    assert out["messages"][0].content == "你好！有什么可以帮你？"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] rag_gate 全部 {len(fns)} 个单测通过")
