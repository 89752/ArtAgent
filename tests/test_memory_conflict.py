"""语义冲突解析单测：REPLACE/MERGE/SKIP 判定 + 守卫规范化。

默认关闭（MEMORY_SMART_MERGE=0），全程 patch LLM 与 embedding，不耗额度。
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.memory.conflict as cf
import src.memory.memory_items as mi


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="mem_conflict_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    monkeypatch.delenv("MEMORY_SMART_MERGE", raising=False)
    with patch("src.memory.memory_items._embed", return_value=None):
        yield


def test_smart_merge_disabled_by_default():
    assert cf.smart_merge_enabled() is False


def test_resolve_conflict_disabled_returns_replace_without_llm():
    def boom(prompt):
        raise AssertionError("关闭时不应调用 LLM")

    assert cf.resolve_conflict("旧", "新", llm=boom) == {"action": "REPLACE", "content": "新"}


def test_resolve_conflict_enabled_parses_actions(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    assert cf.resolve_conflict("旧", "新", llm=lambda p: '{"action": "SKIP", "content": ""}')["action"] == "SKIP"
    assert cf.resolve_conflict("旧", "新", llm=lambda p: '{"action": "MERGE", "content": "新"}')["action"] == "MERGE"
    assert cf.resolve_conflict("旧", "新", llm=lambda p: '{"action": "REPLACE", "content": "新"}')["action"] == "REPLACE"
    # 畸形输出回落 REPLACE
    assert cf.resolve_conflict("旧", "新", llm=lambda p: "not json")["action"] == "REPLACE"


def test_add_memory_smart_skip(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", source="user_explicit")
    item = mi.add_memory(
        "test-user", "用户偏好巴洛克风格", entity="风格", source="user_explicit",
        smart_conflict=True,
        llm=lambda p: '{"action": "SKIP", "content": ""}',
    )
    assert item["action"] == "skip"
    assert len(mi.list_memories("test-user")) == 1


def test_add_memory_smart_merge(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", source="user_explicit")
    item = mi.add_memory(
        "test-user", "用户也喜欢洛可可", entity="风格", source="user_explicit",
        smart_conflict=True,
        llm=lambda p: '{"action": "MERGE", "content": "用户也喜欢洛可可"}',
    )
    assert item["action"] == "merge"
    # 两条都有效（未 supersede）
    assert len(mi.list_memories("test-user")) == 2


def test_add_memory_smart_replace(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", source="user_explicit")
    item = mi.add_memory(
        "test-user", "用户偏好洛可可", entity="风格", source="user_explicit",
        smart_conflict=True,
        llm=lambda p: '{"action": "REPLACE", "content": "用户偏好洛可可"}',
    )
    assert item["action"] == "supersede"
    assert len(mi.list_memories("test-user")) == 1
    assert mi.search_memories("test-user", "洛可可")[0]["content"] == "用户偏好洛可可"


def test_normalize_falls_back_when_disabled():
    def boom(prompt):
        raise AssertionError("关闭时不应调用 LLM")

    assert cf.normalize_memory_text("记住我特别喜欢莫奈", llm=boom) == "记住我特别喜欢莫奈"


def test_normalize_enabled(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    out = cf.normalize_memory_text(
        "记住我特别喜欢莫奈的睡莲",
        llm=lambda p: "用户特别喜欢莫奈的睡莲系列",
    )
    assert out == "用户特别喜欢莫奈的睡莲系列"


def test_guard_uses_normalized_content_when_enabled(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    from langchain_core.messages import AIMessage

    from src.agent.nodes.general import _enforce_memory_write
    from src.agent.state import AgentState

    with patch("src.memory.conflict.normalize_memory_text", return_value="用户偏好莫奈睡莲"):
        state = AgentState(user_query="记住我特别喜欢莫奈的睡莲系列")
        forced = _enforce_memory_write(AIMessage(content="好的，记住了"), state)
    calls = getattr(forced, "tool_calls", None) or []
    assert calls and calls[0]["name"] == "remember"
    assert calls[0]["args"]["content"] == "用户偏好莫奈睡莲"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] memory_conflict 全部 {len(fns)} 个单测通过")
