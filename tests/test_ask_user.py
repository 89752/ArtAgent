"""信息缺口澄清节点单测：仅信息不足时追问，一般歧义放行。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes.common import _info_gap, ask_user
from src.agent.state import AgentState


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
    assert out["pending_clarification"] == ""  # 信息足够时清掉遗留追问


def test_rewrite_ambiguous_triggers_ask():
    state = AgentState(
        user_query="关于那幅画你了解吗",
        intent="general",
        rewrite_ambiguous=True,
    )
    out = ask_user(state)
    assert out["ask_user"] == "ask"
    assert "确定" in out["pending_clarification"]


def test_rewrite_not_ambiguous_passes_on_normal_question():
    state = AgentState(user_query="梵高有哪些作品", intent="general")
    out = ask_user(state)
    assert out["ask_user"] == "continue"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] ask_user 全部 {len(fns)} 个单测通过")
