"""memory_items 存储单测：写入/检索/覆盖/软删除/隔离 + 工具 + 守卫。

全程 patch 掉 bge-m3 embedding（避免加载模型），用关键词回退路径验证逻辑。
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.memory.memory_items as mi
import src.memory.store as store


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="mem_items_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    with patch("src.memory.memory_items._embed", return_value=None):
        yield


def test_add_and_recall_keyword():
    mi.add_memory("test-user", "用户偏好莫奈的睡莲系列", entity="莫奈")
    hits = mi.search_memories("test-user", "莫奈")
    assert len(hits) == 1
    assert hits[0]["content"] == "用户偏好莫奈的睡莲系列"
    assert hits[0]["source"] == "user_explicit"


def test_add_same_content_updates_not_duplicates():
    a = mi.add_memory("test-user", "用户偏好莫奈睡莲", entity="莫奈")
    b = mi.add_memory("test-user", "用户偏好莫奈睡莲", entity="莫奈")
    assert a["id"] == b["id"]
    assert b["action"] == "update"
    assert len(mi.list_memories("test-user")) == 1


def test_drift_supersedes_old_value():
    old = mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", importance=0.8)
    new = mi.add_memory("test-user", "用户偏好洛可可", entity="风格", importance=0.9)
    assert old["id"] != new["id"]
    assert new["action"] == "supersede"
    # 旧条目保留可追溯但不再参与检索
    assert len(mi.search_memories("test-user", "巴洛克")) == 0
    assert len(mi.search_memories("test-user", "洛可可")) == 1
    rows = mi.list_memories("test-user", include_superseded=True)
    assert len(rows) == 2
    superseded = [r for r in rows if r["superseded_by"] == new["id"]]
    assert len(superseded) == 1


def test_isolation_between_users():
    mi.add_memory("u-a", "用户偏好莫奈", entity="莫奈")
    mi.add_memory("u-b", "用户偏好卡拉瓦乔", entity="卡拉瓦乔")
    assert len(mi.search_memories("u-a", "卡拉瓦乔")) == 0
    assert len(mi.search_memories("u-b", "莫奈")) == 0


def test_delete_memory_soft_and_audit():
    item = mi.add_memory("test-user", "用户偏好莫奈", entity="莫奈")
    assert mi.delete_memory("test-user", item["id"]) is True
    assert mi.search_memories("test-user", "莫奈") == []
    assert mi.delete_memory("test-user", item["id"]) is False
    # 审计表有 create/delete
    conn = mi._get_conn()
    actions = [r[0] for r in conn.execute(
        "SELECT action FROM memory_events WHERE user_id = 'test-user'"
    ).fetchall()]
    assert "create" in actions and "delete" in actions


def test_delete_by_entity_and_clear():
    mi.add_memory("test-user", "喜欢莫奈", entity="莫奈")
    mi.add_memory("test-user", "莫奈睡莲", entity="莫奈")
    assert mi.delete_by_entity("test-user", "莫奈") == 2
    assert mi.list_memories("test-user") == []
    mi.add_memory("test-user", "新记忆", entity="X")
    assert mi.clear_user_memories("test-user") >= 1  # 硬删全部行（含已软删）
    assert mi.list_memories("test-user") == []


def test_store_load_preferences_prefers_memory_items():
    mi.add_memory("test-user", "用户偏好莫奈睡莲系列", entity="莫奈")
    prefs = store.load_preferences("test-user")
    assert prefs["artists"] == ["用户偏好莫奈睡莲系列"]
    items = store.list_preferences("test-user")
    assert items[0]["kind"] == "preference"
    assert items[0]["value"] == "用户偏好莫奈睡莲系列"


def test_remember_tool_and_guard():
    from langchain_core.messages import AIMessage

    from src.agent.nodes.general import _enforce_memory_write
    from src.agent.state import AgentState
    from src.tools.memory import recall, remember

    out = remember.invoke({
        "content": "用户偏好莫奈的睡莲系列",
        "entity": "莫奈",
    })
    assert "已记住" in out
    assert "[" not in out  # 不暴露内部记忆 id
    hits = recall.invoke({"query": "莫奈"})
    assert hits and hits[0]["content"] == "用户偏好莫奈的睡莲系列"

    # 守卫：AI 口头确认但没调 remember → 强制补 tool_call
    resp = AIMessage(content="好的，我已经记住了！")
    state = AgentState(user_query="记住我特别喜欢莫奈的睡莲系列")
    forced = _enforce_memory_write(resp, state)
    calls = getattr(forced, "tool_calls", None) or []
    assert calls and calls[0]["name"] == "remember"
    # 已带 remember 调用时不重复强制
    # 本轮已执行过 remember（ToolMessage 在历史里）→ 不再强制，防死循环
    from langchain_core.messages import ToolMessage

    done_state = AgentState(
        user_query="记住我特别喜欢莫奈的睡莲系列",
        messages=[ToolMessage(content="已记住", name="remember", tool_call_id="t1")],
    )
    done_forced = _enforce_memory_write(AIMessage(content="好的，已经记住了"), done_state)
    assert not getattr(done_forced, "tool_calls", None)
    resp2 = AIMessage(
        content="正在保存", tool_calls=[{"name": "remember", "args": {}, "id": "x", "type": "tool_call"}]
    )
    assert _enforce_memory_write(resp2, state) is resp2
    # 非记忆意图不触发
    resp3 = AIMessage(content="好的")
    assert _enforce_memory_write(resp3, AgentState(user_query="梵高有哪些代表作")) is resp3
