"""自动抽取层单测：开关/解析/敏感过滤/落库/节流。

全程 patch 掉 bge-m3 embedding 与真实 LLM，不消耗 API 额度。
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.memory.extract as ex
import src.memory.memory_items as mi


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="mem_extract_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    monkeypatch.delenv("MEMORY_AUTO_EXTRACT", raising=False)
    monkeypatch.delenv("MEMORY_EXTRACT_INTERVAL", raising=False)
    with patch("src.memory.memory_items._embed", return_value=None):
        yield


def test_extract_enabled_default_off():
    assert ex.extract_enabled() is False


def test_extract_enabled_toggle():
    import os

    os.environ["MEMORY_AUTO_EXTRACT"] = "1"
    try:
        assert ex.extract_enabled() is True
        assert ex.extract_interval() == 2
    finally:
        os.environ.pop("MEMORY_AUTO_EXTRACT", None)


def test_extract_interval_parsed():
    import os

    os.environ["MEMORY_EXTRACT_INTERVAL"] = "3"
    try:
        assert ex.extract_interval() == 3
    finally:
        os.environ.pop("MEMORY_EXTRACT_INTERVAL", None)


def test_extract_memories_parses_items():
    def fake_llm(prompt):
        assert "【对话】" in prompt
        return (
            '{"items": [{"action": "ADD", "kind": "preference", "entity": "莫奈", '
            '"content": "用户特别喜欢莫奈的睡莲系列", "importance": 0.8}, '
            '{"action": "NOOP", "kind": "fact", "entity": "", "content": "", '
            '"importance": 0.0}]}'
        )

    items = ex.extract_memories("用户：我喜欢莫奈的睡莲系列", llm=fake_llm)
    assert len(items) == 2
    assert items[0]["action"] == "ADD"
    assert items[0]["content"] == "用户特别喜欢莫奈的睡莲系列"
    assert items[1]["action"] == "NOOP"


def test_extract_memories_malformed_returns_empty():
    assert ex.extract_memories("x", llm=lambda p: "not json") == []


def test_extract_memories_skips_sensitive():
    def fake_llm(prompt):
        return (
            '{"items": [{"action": "ADD", "kind": "fact", "entity": "身份证", '
            '"content": "用户身份证号是 110101199001011234", "importance": 0.9}]}'
        )

    assert ex.extract_memories("x", llm=fake_llm) == []


def test_apply_extracted_writes_with_source():
    stats = ex.apply_extracted("test-user", None, [
        {"action": "ADD", "kind": "preference", "entity": "莫奈",
         "content": "用户偏好莫奈睡莲", "importance": 0.8},
    ])
    assert stats["added"] == 1
    rows = mi.list_memories("test-user")
    assert rows[0]["source"] == "extracted"
    assert rows[0]["entity"] == "莫奈"


def test_apply_extracted_delete():
    mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", source="extracted")
    stats = ex.apply_extracted("test-user", None, [
        {"action": "DELETE", "kind": "fact", "entity": "风格", "content": ""},
    ])
    assert stats["deleted"] == 1
    assert mi.list_memories("test-user") == []


def test_apply_extracted_update_supersedes_old():
    mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", source="extracted")
    stats = ex.apply_extracted("test-user", None, [
        {"action": "UPDATE", "kind": "preference", "entity": "风格",
         "content": "用户偏好洛可可", "importance": 0.9},
    ])
    assert stats["superseded"] == 1
    hits = mi.search_memories("test-user", "洛可可")
    assert hits and hits[0]["content"] == "用户偏好洛可可"
    assert mi.search_memories("test-user", "巴洛克") == []


def test_apply_extracted_dedups_explicit_memory():
    """自动抽取与用户明确记忆重复时跳过，不产生双份。"""
    mi.add_memory(
        "test-user", "记住我特别喜欢莫奈的睡莲系列",
        entity="莫奈", source="user_explicit",
    )
    stats = ex.apply_extracted("test-user", None, [{
        "action": "ADD", "kind": "preference", "entity": "莫奈",
        "content": "用户特别喜欢莫奈的睡莲系列", "importance": 0.8,
    }])
    assert stats["dup"] == 1
    assert len(mi.list_memories("test-user")) == 1


def test_load_memory_falls_back_to_recent_items():
    """问题与记忆词面不重叠时，注入兜底的最远/最重要条目。"""
    from src.agent.nodes.common import load_memory
    from src.agent.state import AgentState

    mi.add_memory("test-user", "用户喜欢莫奈睡莲", entity="莫奈", importance=0.9)
    out = load_memory(AgentState(user_query="今天天气怎么样"))
    assert "莫奈" in out["memory_block"]


def test_maybe_extract_disabled_skips_llm():
    messages = [HumanMessage(content="我喜欢莫奈"), AIMessage(content="好的")]
    turns, result = ex.maybe_extract(messages, "test-user", 0)
    assert result == {}
    assert turns == 0


def test_maybe_extract_throttle_and_write(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTO_EXTRACT", "1")
    messages = [
        HumanMessage(content="我住上海"),
        AIMessage(content="好的"),
        HumanMessage(content="我喜欢宁静的风景"),
    ]

    def fake_extract(conversation):
        return [{
            "action": "ADD", "kind": "fact", "entity": "上海",
            "content": "用户住在上海", "importance": 0.7,
        }]

    with patch.object(ex, "extract_memories", side_effect=fake_extract):
        # 未到间隔（2 轮）：跳过
        turns, result = ex.maybe_extract(messages, "test-user", 2)
        assert turns == 2 and result == {}
        # 从 0 起到达间隔：抽取并落库
        turns, result = ex.maybe_extract(messages, "test-user", 0)
        assert turns == 2
        assert result["stats"]["added"] == 1
        assert mi.search_memories("test-user", "上海")[0]["content"] == "用户住在上海"


def test_save_memory_runs_without_auto_extract():
    from src.agent.nodes.common import save_memory
    from src.agent.state import AgentState

    out = save_memory(AgentState(messages=[HumanMessage(content="你好")]))
    assert out["current_step"] == "save_memory"
    assert out["memory_extracted_turns"] == 0
    assert out["memory_extract_result"] == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] memory_extract 全部 {len(fns)} 个单测通过")
