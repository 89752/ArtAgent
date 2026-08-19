"""ArtAgent 记忆系统统一单测（合并 extract / conflict / capacity / lifecycle / user_doc / metrics）。

覆盖：自动抽取、写入门控、冲突合并、容量淘汰、画像聚合、结构化用户文档、
失效审查、相似事实合并、抽取质量指标、shutdown flush、记忆导入。
全程 patch 掉 embedding 与真实 LLM，不耗 API 额度、不碰真实索引。
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.memory.conflict as cf
import src.memory.extract as ex
import src.memory.lifecycle as lc
import src.memory.memory_items as mi
import src.memory.metrics as mtr
import src.memory.collections as col
import src.memory.summary as summary_mod
import src.memory.user_doc as ud
from src.data import db


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="memory_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    ud._reset_for_tests(tmp)
    lc._reset_for_tests(tmp)
    mtr._reset_for_tests(tmp)
    summary_mod._DB_PATH = tmp.parent / "conversations.db"
    summary_mod._db_ready = False
    col._DB_PATH = tmp
    col._db_ready = False
    db.close_all()
    mi.clear_active_user_id()
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    for name in (
        "MEMORY_AUTO_EXTRACT",
        "MEMORY_EXTRACT_INTERVAL",
        "MEMORY_EXTRACT_DEBOUNCE_SEC",
        "MEMORY_SMART_MERGE",
        "MEMORY_PROFILE_REFRESH",
        "MEMORY_MAINTENANCE_INTERVAL_HOURS",
        "MEMORY_VECTOR_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    ex._pending.clear()
    ex._worker_started = False
    with patch("src.memory.memory_items._embed", return_value=None):
        try:
            yield
        finally:
            mi.clear_active_user_id()


def _backdate(item_id: str, days: int) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    mi._get_conn().execute(
        "UPDATE memory_items SET updated_at = ? WHERE id = ?", (ts, item_id)
    )
    mi._get_conn().commit()


# ══════════════ 自动抽取 / 写入门控 ══════════════
def test_extract_enabled_default_on():
    assert ex.extract_enabled() is True
    assert ex.extract_interval() == 1


def test_extract_prompt_covers_meta_preferences():
    assert "语言倾向" in ex.EXTRACT_PROMPT
    assert "回复风格" in ex.EXTRACT_PROMPT
    assert "personalContext" in ex.EXTRACT_PROMPT
    assert "correction" in ex.EXTRACT_PROMPT
    assert "纠正" in ex.EXTRACT_PROMPT


def test_extract_enabled_toggle():
    os.environ["MEMORY_AUTO_EXTRACT"] = "1"
    try:
        assert ex.extract_enabled() is True
        assert ex.extract_interval() == 1
    finally:
        os.environ.pop("MEMORY_AUTO_EXTRACT", None)


def test_extract_interval_parsed():
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
            '"content": "用户特别喜欢莫奈的睡莲系列", "importance": 0.8, '
            '"scope": "user", "durability": "durable", "authority": "descriptive"}, '
            '{"action": "NOOP", "kind": "fact", "entity": "", "content": "", '
            '"importance": 0.0, "scope": "thread", "durability": "temporary", '
            '"authority": "descriptive"}]}'
        )

    items = ex.extract_memories("用户：我喜欢莫奈的睡莲系列", llm=fake_llm)
    assert len(items) == 2
    assert items[0]["action"] == "ADD"
    assert items[0]["content"] == "用户特别喜欢莫奈的睡莲系列"
    assert items[0]["scope"] == "user"
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


def test_gate_rejects_thread_scope():
    passed, rejected = ex.gate_items([{
        "action": "ADD", "kind": "preference", "entity": "莫奈",
        "content": "用户喜欢莫奈", "importance": 0.8,
        "scope": "thread", "durability": "durable", "authority": "descriptive",
    }])
    assert passed == []
    assert "scope:thread" in rejected


def test_gate_rejects_temporary_durability():
    passed, rejected = ex.gate_items([{
        "action": "ADD", "kind": "fact", "entity": "画展",
        "content": "用户本周日去看莫奈展", "importance": 0.8,
        "scope": "user", "durability": "temporary", "authority": "descriptive",
    }])
    assert passed == []
    assert "durability:temporary" in rejected


def test_gate_rejects_transactional_authority():
    passed, rejected = ex.gate_items([{
        "action": "ADD", "kind": "fact", "entity": "",
        "content": "用户要求删除上传的 PDF", "importance": 0.9,
        "scope": "user", "durability": "durable", "authority": "transactional",
    }])
    assert passed == []
    assert "authority:transactional" in rejected


def test_gate_rejects_low_confidence(monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACT_CONFIDENCE", "0.8")
    passed, rejected = ex.gate_items([{
        "action": "ADD", "kind": "fact", "entity": "上海",
        "content": "用户可能住在上海", "importance": 0.5,
        "scope": "user", "durability": "durable", "authority": "descriptive",
    }])
    assert passed == []
    assert any(r.startswith("confidence:") for r in rejected)


def test_gate_passes_user_durable_descriptive():
    passed, rejected = ex.gate_items([{
        "action": "ADD", "kind": "preference", "entity": "莫奈",
        "content": "用户喜欢莫奈睡莲", "importance": 0.8,
        "scope": "user", "durability": "durable", "authority": "descriptive",
    }])
    assert len(passed) == 1
    assert rejected == []


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


def test_maybe_extract_disabled_skips_llm():
    os.environ["MEMORY_AUTO_EXTRACT"] = "0"
    messages = [HumanMessage(content="我喜欢莫奈"), AIMessage(content="好的")]
    try:
        turns, result = ex.maybe_extract(messages, "test-user", 0)
        assert result == {}
        assert turns == 0
    finally:
        os.environ.pop("MEMORY_AUTO_EXTRACT", None)


def test_maybe_extract_throttle_and_write(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTO_EXTRACT", "1")
    monkeypatch.setenv("MEMORY_EXTRACT_INTERVAL", "2")
    messages = [
        HumanMessage(content="我住上海"),
        AIMessage(content="好的"),
        HumanMessage(content="我喜欢宁静的风景"),
    ]

    def fake_extract(conversation):
        return [{
            "action": "ADD", "kind": "fact", "entity": "上海",
            "content": "用户住在上海", "importance": 0.7,
            "scope": "user", "durability": "durable", "authority": "descriptive",
        }]

    with patch.object(ex, "extract_memories", side_effect=fake_extract):
        turns, result = ex.maybe_extract(messages, "test-user", 2)
        assert turns == 2 and result == {}
        turns, result = ex.maybe_extract(messages, "test-user", 0)
        assert turns == 2
        assert result["stats"]["added"] == 1
        assert mi.search_memories("test-user", "上海")[0]["content"] == "用户住在上海"


def test_maybe_extract_single_turn_default_on(monkeypatch):
    messages = [HumanMessage(content="我喜欢莫奈的睡莲系列")]

    def fake_extract(conversation):
        return [{
            "action": "ADD", "kind": "preference", "entity": "莫奈",
            "content": "用户特别喜欢莫奈的睡莲系列", "importance": 0.8,
            "scope": "user", "durability": "durable", "authority": "descriptive",
        }]

    with patch.object(ex, "extract_memories", side_effect=fake_extract):
        turns, result = ex.maybe_extract(messages, "test-user", 0)
    assert turns == 1
    assert result["stats"]["added"] == 1
    assert mi.search_memories("test-user", "睡莲")[0]["content"] == "用户特别喜欢莫奈的睡莲系列"


def test_schedule_extract_default_on(monkeypatch):
    with patch.object(ex, "maybe_extract", return_value=(0, {})), patch.object(
        ex, "_extract_worker", lambda: None
    ):
        result = ex.schedule_extract(
            [HumanMessage(content="我喜欢梵高的星空")], "test-user"
        )
    assert result.get("scheduled") is True


def test_record_language_preference_detects_chinese():
    out = ex.record_language_preference(
        [HumanMessage(content="你好，帮我推荐几幅画")], "test-user"
    )
    assert out.get("added") == "中文"
    hits = mi.search_memories("test-user", "中文")
    assert hits and "中文" in hits[0]["content"]
    out2 = ex.record_language_preference(
        [HumanMessage(content="再推荐一幅")], "test-user"
    )
    assert out2.get("skipped") == "exists"
    assert len(mi.list_memories("test-user")) == 1


def test_record_language_preference_english():
    out = ex.record_language_preference(
        [HumanMessage(content="Please recommend some paintings")], "test-user"
    )
    assert out.get("added") == "英文"


def test_shutdown_flush_idle_returns_true():
    assert ex.shutdown_flush(0.2) is True


def test_shutdown_flush_timeout_when_pending(monkeypatch):
    monkeypatch.setenv("MEMORY_EXTRACT_DEBOUNCE_SEC", "2")
    with patch.object(ex, "maybe_extract", return_value=(0, {})), patch.object(
        ex, "_extract_worker", lambda: None
    ):
        ex.schedule_extract([HumanMessage(content="我喜欢莫奈")], "test-user")
    assert ex.shutdown_flush(0.1) is False
    ex._pending.clear()


def test_save_memory_runs_without_auto_extract(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTO_EXTRACT", "0")
    from src.agent.nodes.common import save_memory
    from src.agent.state import AgentState

    out = save_memory(AgentState(messages=[HumanMessage(content="你好")]))
    assert out["current_step"] == "save_memory"
    assert out["memory_extracted_turns"] == 0
    assert out["memory_extract_result"] == {}


# ══════════════ 语义冲突合并 ══════════════
def test_smart_merge_enabled_by_default():
    assert cf.smart_merge_enabled() is True


def test_smart_merge_disabled_when_env_zero(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "0")
    assert cf.smart_merge_enabled() is False


def test_resolve_conflict_disabled_returns_replace_without_llm():
    os.environ["MEMORY_SMART_MERGE"] = "0"

    def boom(prompt):
        raise AssertionError("关闭时不应调用 LLM")

    try:
        assert cf.resolve_conflict("旧", "新", llm=boom) == {"action": "REPLACE", "content": "新"}
    finally:
        os.environ.pop("MEMORY_SMART_MERGE", None)


def test_resolve_conflict_enabled_parses_actions(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    assert cf.resolve_conflict("旧", "新", llm=lambda p: '{"action": "SKIP", "content": ""}')["action"] == "SKIP"
    assert cf.resolve_conflict("旧", "新", llm=lambda p: '{"action": "MERGE", "content": "新"}')["action"] == "MERGE"
    assert cf.resolve_conflict("旧", "新", llm=lambda p: '{"action": "REPLACE", "content": "新"}')["action"] == "REPLACE"
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
    os.environ["MEMORY_SMART_MERGE"] = "0"

    def boom(prompt):
        raise AssertionError("关闭时不应调用 LLM")

    try:
        assert cf.normalize_memory_text("记住我特别喜欢莫奈", llm=boom) == "记住我特别喜欢莫奈"
    finally:
        os.environ.pop("MEMORY_SMART_MERGE", None)


def test_normalize_enabled(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    out = cf.normalize_memory_text(
        "记住我特别喜欢莫奈的睡莲",
        llm=lambda p: "用户特别喜欢莫奈的睡莲系列",
    )
    assert out == "用户特别喜欢莫奈的睡莲系列"


def test_guard_uses_normalized_content_when_enabled(monkeypatch):
    monkeypatch.setenv("MEMORY_SMART_MERGE", "1")
    from src.agent.nodes.general import _enforce_memory_write
    from src.agent.state import AgentState

    with patch("src.memory.conflict.normalize_memory_text", return_value="用户偏好莫奈睡莲"):
        state = AgentState(user_query="记住我特别喜欢莫奈的睡莲系列")
        forced = _enforce_memory_write(AIMessage(content="好的，记住了"), state)
    calls = getattr(forced, "tool_calls", None) or []
    assert calls and calls[0]["name"] == "remember"
    assert calls[0]["args"]["content"] == "用户偏好莫奈睡莲"


# ══════════════ 容量 / 淘汰 / 画像聚合 ══════════════
def test_evicts_lowest_importance_when_over_cap(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_ITEMS_PER_USER", "2")
    mi.add_memory("test-user", "低价值记忆A", entity="A", importance=0.1)
    mi.add_memory("test-user", "高价值记忆B", entity="B", importance=0.9)
    mi.add_memory("test-user", "中价值记忆C", entity="C", importance=0.5)
    contents = [i["content"] for i in mi.list_memories("test-user")]
    assert "低价值记忆A" not in contents
    assert "高价值记忆B" in contents and "中价值记忆C" in contents
    actions = [r[0] for r in mi._get_conn().execute(
        "SELECT action FROM memory_events WHERE user_id = 'test-user'"
    ).fetchall()]
    assert "evict" in actions


def test_profile_item_protected_from_eviction(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_ITEMS_PER_USER", "2")
    mi.add_memory("test-user", "用户画像：喜欢印象派", kind="profile",
                  entity="user_profile", importance=0.5)
    mi.add_memory("test-user", "普通记忆A", entity="A", importance=0.1)
    mi.add_memory("test-user", "普通记忆B", entity="B", importance=0.2)
    contents = [i["content"] for i in mi.list_memories("test-user")]
    assert "用户画像：喜欢印象派" in contents
    assert len(contents) == 2


def test_evicts_by_chars_when_over_chars(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_ITEMS_PER_USER", "100")
    monkeypatch.setenv("MEMORY_MAX_CHARS_PER_USER", "20")
    mi.add_memory("test-user", "很长很长很长很长很长很长很长很长很长很长很长", entity="B", importance=0.9)
    mi.add_memory("test-user", "短", entity="A", importance=0.1)
    contents = [i["content"] for i in mi.list_memories("test-user")]
    assert contents == ["短"]


def test_chroma_backend_falls_back_on_unavailable(monkeypatch):
    monkeypatch.setenv("MEMORY_VECTOR_BACKEND", "chroma")
    mi.add_memory("test-user", "莫奈睡莲", entity="莫奈", importance=0.8)

    def boom(*args, **kwargs):
        raise RuntimeError("chroma down")

    with patch("src.retrieval.hybrid.get_or_create_chroma_collection", side_effect=boom):
        mi.add_memory("test-user", "卡拉瓦乔明暗", entity="卡拉瓦乔", importance=0.7)
        hits = mi.search_memories("test-user", "莫奈")
    assert any("莫奈" in i["content"] for i in hits)


def test_chroma_candidate_path(monkeypatch):
    monkeypatch.setenv("MEMORY_VECTOR_BACKEND", "chroma")

    class FakeCol:
        def __init__(self, ids):
            self._ids = ids

        def upsert(self, **kwargs):
            pass

        def query(self, **kwargs):
            return {"ids": [self._ids]}

        def delete(self, ids):
            pass

    ids_box = []
    fake = FakeCol(ids_box)
    with patch("src.retrieval.hybrid.get_or_create_chroma_collection", return_value=fake), \
         patch("src.memory.memory_items._embed", return_value="[0.1, 0.2, 0.3]"):
        mi.add_memory("test-user", "用户喜欢莫奈睡莲", entity="莫奈", importance=0.8)
        ids_box.append(mi.list_memories("test-user")[0]["id"])
        hits = mi.search_memories("test-user", "莫奈")
    assert hits and hits[0]["content"] == "用户喜欢莫奈睡莲"


def test_profile_disabled_when_env_zero(monkeypatch):
    from src.memory.profile import maybe_refresh_profile

    def boom(prompt):
        raise AssertionError("关闭时不应调用 LLM")

    monkeypatch.setenv("MEMORY_PROFILE_REFRESH", "0")
    assert maybe_refresh_profile("test-user", llm=boom) == {}


def test_profile_refresh_writes_and_supersedes(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE_REFRESH", "1")
    from src.memory.profile import load_profile_item, maybe_refresh_profile

    mi.add_memory("test-user", "用户喜欢莫奈睡莲", entity="莫奈", importance=0.9)
    mi.add_memory("test-user", "用户住在上海", entity="上海", importance=0.6)
    out = maybe_refresh_profile(
        "test-user",
        llm=lambda p: "用户喜欢印象派与莫奈睡莲；用户住在上海。",
    )
    assert out["action"] in {"create", "supersede"}
    item = load_profile_item("test-user")
    assert item is not None and "印象派" in item["content"]
    assert maybe_refresh_profile("test-user", llm=lambda p: "不应被调用")["skipped"] == "fresh"


def test_profile_force_refresh_bypasses_freshness(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE_REFRESH", "1")
    from src.memory.profile import load_profile_item, maybe_refresh_profile

    mi.add_memory("test-user", "用户喜欢莫奈", entity="莫奈", importance=0.9)
    maybe_refresh_profile("test-user", llm=lambda p: "用户喜欢莫奈。")
    assert load_profile_item("test-user") is not None
    assert maybe_refresh_profile("test-user", llm=lambda p: "不应被调用")["skipped"] == "fresh"
    mi.add_memory("test-user", "用户主要使用中文交流", entity="语言", importance=0.9)
    out = maybe_refresh_profile(
        "test-user",
        force=True,
        llm=lambda p: "用户喜欢莫奈；用户主要使用中文交流。",
    )
    assert out["action"] == "supersede"
    assert "中文" in load_profile_item("test-user")["content"]


def test_profile_fallback_when_llm_fails(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE_REFRESH", "1")
    from src.memory.profile import load_profile_item, maybe_refresh_profile

    mi.add_memory("test-user", "用户喜欢莫奈", entity="莫奈", importance=0.9)

    def boom(prompt):
        raise RuntimeError("llm down")

    out = maybe_refresh_profile("test-user", llm=boom)
    assert out["action"] == "create"
    item = load_profile_item("test-user")
    assert item is not None and "莫奈" in item["content"]


def test_load_memory_injects_profile():
    from src.agent.nodes.common import load_memory
    from src.agent.state import AgentState
    from src.memory.profile import maybe_refresh_profile

    mi.add_memory("test-user", "用户喜欢莫奈睡莲", entity="莫奈", importance=0.9)
    os.environ["MEMORY_PROFILE_REFRESH"] = "1"
    maybe_refresh_profile("test-user", llm=lambda p: "用户喜欢印象派与莫奈。")
    os.environ.pop("MEMORY_PROFILE_REFRESH", None)
    out = load_memory(AgentState(user_query="推荐几幅画"))
    assert "【用户画像】" in out["memory_block"]
    assert "莫奈" in out["memory_block"]


def test_load_memory_falls_back_to_recent_items():
    from src.agent.nodes.common import load_memory
    from src.agent.state import AgentState

    mi.add_memory("test-user", "用户喜欢莫奈睡莲", entity="莫奈", importance=0.9)
    out = load_memory(AgentState(user_query="今天天气怎么样"))
    assert "莫奈" in out["memory_block"]


# ══════════════ 结构化用户文档 / 生命周期 ══════════════
def test_doc_roundtrip():
    doc = ud.empty_doc()
    doc["personalContext"]["summary"] = "用户主要使用中文交流"
    doc["topOfMind"]["summary"] = "正在研究印象派"
    ud.save_doc("test-user", doc)
    loaded = ud.load_doc("test-user")
    assert loaded["personalContext"]["summary"] == "用户主要使用中文交流"
    assert loaded["topOfMind"]["summary"] == "正在研究印象派"
    assert loaded["longTerm"]["summary"] == ""


def test_update_doc_merges_sections():
    doc = ud.empty_doc()
    doc["topOfMind"]["summary"] = "旧关注：巴洛克"
    ud.save_doc("test-user", doc)

    def fake_llm(prompt):
        assert "【当前文档】" in prompt
        return (
            '{"personalContext": {"shouldUpdate": true, '
            '"summary": "用户主要使用中文交流，偏好简洁回答。"}, '
            '"topOfMind": {"shouldUpdate": false, "summary": ""}, '
            '"recent": {"shouldUpdate": true, '
            '"summary": "用户最近在对比莫奈、梵高、毕加索。"}}'
        )

    result = ud.update_user_doc(
        "test-user",
        [HumanMessage(content="我喜欢莫奈，请简短回答")],
        llm=fake_llm,
    )
    assert result["updated"] == ["personalContext", "recent"]
    loaded = ud.load_doc("test-user")
    assert "中文" in loaded["personalContext"]["summary"]
    assert loaded["topOfMind"]["summary"] == "旧关注：巴洛克"
    assert "莫奈" in loaded["recent"]["summary"]


def test_update_doc_no_update_keeps_doc():
    doc = ud.empty_doc()
    doc["personalContext"]["summary"] = "旧画像"
    ud.save_doc("test-user", doc)

    result = ud.update_user_doc(
        "test-user",
        [HumanMessage(content="你好")],
        llm=lambda p: '{"personalContext": {"shouldUpdate": false, "summary": ""}}',
    )
    assert result["skipped"] == "no_update"
    assert ud.load_doc("test-user")["personalContext"]["summary"] == "旧画像"


def test_update_doc_failure_keeps_doc():
    doc = ud.empty_doc()
    doc["personalContext"]["summary"] = "旧画像"
    ud.save_doc("test-user", doc)

    result = ud.update_user_doc(
        "test-user",
        [HumanMessage(content="你好")],
        llm=lambda p: (_ for _ in ()).throw(RuntimeError("llm down")),
    )
    assert "error" in result
    assert ud.load_doc("test-user")["personalContext"]["summary"] == "旧画像"


def test_save_doc_revision_conflict():
    doc = ud.empty_doc()
    doc["personalContext"]["summary"] = "v1"
    assert ud.save_doc("test-user", doc, expected_revision=0) is True
    _doc, rev = ud.load_doc_with_revision("test-user")
    assert rev == 1
    stale = ud.empty_doc()
    stale["personalContext"]["summary"] = "v2-stale"
    assert ud.save_doc("test-user", stale, expected_revision=0) is False
    _doc2, rev2 = ud.load_doc_with_revision("test-user")
    assert rev2 == 1
    assert _doc2["personalContext"]["summary"] == "v1"
    fresh = ud.empty_doc()
    fresh["personalContext"]["summary"] = "v2"
    assert ud.save_doc("test-user", fresh, expected_revision=1) is True
    _doc3, rev3 = ud.load_doc_with_revision("test-user")
    assert rev3 == 2
    assert _doc3["personalContext"]["summary"] == "v2"


def test_sync_profile_item_from_doc():
    doc = ud.empty_doc()
    doc["personalContext"]["summary"] = "用户主要使用中文交流"
    doc["topOfMind"]["summary"] = "正在研究立体主义"
    ud.save_doc("test-user", doc)

    from src.memory.profile import load_profile_item, sync_profile_item_from_doc

    out = sync_profile_item_from_doc("test-user")
    assert out["action"] in {"create", "supersede"}
    item = load_profile_item("test-user")
    assert item is not None
    assert "中文" in item["content"]
    assert "立体主义" in item["content"]


def test_load_memory_injects_doc_sections():
    from src.agent.nodes.common import load_memory
    from src.agent.state import AgentState

    doc = ud.empty_doc()
    doc["personalContext"]["summary"] = "用户主要使用中文交流"
    doc["topOfMind"]["summary"] = "正在研究印象派"
    doc["recent"]["summary"] = "最近对比莫奈梵高毕加索"
    ud.save_doc("test-user", doc)

    out = load_memory(AgentState(user_query="推荐几幅画"))
    assert "【用户画像】" in out["memory_block"]
    assert "中文" in out["memory_block"]
    assert "【当前关注】" in out["memory_block"]
    assert "印象派" in out["memory_block"]
    assert "【近期】" in out["memory_block"]


def test_load_memory_guarantees_language_and_correction():
    from src.agent.nodes.common import load_memory
    from src.agent.state import AgentState

    mi.add_memory(
        "test-user",
        "用户主要使用中文交流",
        kind="preference",
        entity="语言",
        importance=0.7,
    )
    mi.add_memory(
        "test-user",
        "用户纠正过：莫奈不是印象派的创始人，而是代表画家",
        kind="correction",
        entity="纠正",
        importance=0.95,
    )
    out = load_memory(AgentState(user_query="推荐几幅静物画"))
    assert "中文交流" in out["memory_block"]
    assert "纠正过" in out["memory_block"]


def test_add_memory_stores_expected_valid_days():
    item = mi.add_memory(
        "test-user", "用户喜欢莫奈睡莲", entity="莫奈", expected_valid_days=180
    )
    assert item["expected_valid_days"] == 180


def test_staleness_due_finds_expired_only():
    old = mi.add_memory("test-user", "用户偏好巴洛克", entity="风格", expected_valid_days=1)
    fresh = mi.add_memory("test-user", "用户喜欢莫奈", entity="莫奈", expected_valid_days=365)
    _backdate(old["id"], 10)
    due = lc.staleness_due("test-user")
    ids = {i["id"] for i in due}
    assert old["id"] in ids
    assert fresh["id"] not in ids


def test_review_staleness_removes_and_extends():
    rm = mi.add_memory("test-user", "用户去年住在上海", entity="上海", expected_valid_days=30)
    ext = mi.add_memory("test-user", "用户喜欢印象派", entity="印象派", expected_valid_days=30)
    _backdate(rm["id"], 60)
    _backdate(ext["id"], 60)

    def fake_llm(prompt):
        return json.dumps(
            {
                "decisions": [
                    {"id": rm["id"], "action": "remove", "reason": "已搬家"},
                    {"id": ext["id"], "action": "extend", "extend_by_days": 90, "reason": "仍然成立"},
                ]
            },
            ensure_ascii=False,
        )

    out = lc.review_staleness("test-user", llm=fake_llm)
    assert out["removed"] == 1
    assert out["extended"] == 1
    active = {i["id"] for i in mi.list_memories("test-user")}
    assert rm["id"] not in active
    assert ext["id"] in active
    updated = next(i for i in mi.list_memories("test-user") if i["id"] == ext["id"])
    assert updated["expected_valid_days"] == 120


def test_review_staleness_fallback_extends_on_failure():
    item = mi.add_memory("test-user", "用户喜欢梵高", entity="梵高", expected_valid_days=30)
    _backdate(item["id"], 60)

    def boom(prompt):
        raise RuntimeError("llm down")

    out = lc.review_staleness("test-user", llm=boom)
    assert out["extended"] == 1
    assert out["removed"] == 0
    updated = mi.list_memories("test-user")[0]
    assert updated["expected_valid_days"] == 60


def test_maybe_maintenance_throttles(monkeypatch):
    monkeypatch.setenv("MEMORY_MAINTENANCE_INTERVAL_HOURS", "24")
    first = lc.maybe_maintenance("test-user")
    assert "skipped" not in first or first["skipped"] == "no_due"
    second = lc.maybe_maintenance("test-user")
    assert second["skipped"] == "fresh"


def test_consolidate_similar_merges_group():
    a = mi.add_memory(
        "test-user", "用户喜欢莫奈的睡莲", entity="莫奈",
        kind="preference", importance=0.8,
    )
    b = mi.add_memory(
        "test-user", "用户偏爱莫奈的睡莲系列", entity="莫奈",
        kind="fact", importance=0.7,
    )
    c = mi.add_memory(
        "test-user", "用户喜欢梵高的星空", entity="梵高",
        kind="preference", importance=0.9,
    )

    def fake_llm(prompt):
        return json.dumps(
            {"consolidated": "用户非常喜欢莫奈的睡莲系列"},
            ensure_ascii=False,
        )

    out = lc.consolidate_similar("test-user", llm=fake_llm)
    assert out["merged"] == 1
    active = mi.list_memories("test-user")
    assert len(active) == 2
    merged = next(i for i in active if i["source"] == "consolidated")
    assert "睡莲" in merged["content"]
    active_ids = {i["id"] for i in active}
    assert a["id"] not in active_ids
    assert b["id"] not in active_ids
    assert c["id"] in active_ids


# ══════════════ 抽取质量指标 / 记忆导入 ══════════════
def test_record_metrics_breakdown():
    summary = mtr.record_extraction_metrics(
        "test-user",
        extracted=5,
        passed=2,
        rejected=["scope:thread", "durability:temporary", "authority:transactional", "confidence:0.3"],
    )
    assert summary["extracted"] == 5
    assert summary["passed"] == 2
    assert summary["rejected"]["scope"] == 1
    assert summary["rejected"]["confidence"] == 1
    assert summary["rejection_rate"] == 0.8
    rows = mtr.recent_extraction_metrics()
    assert rows and rows[0]["rejected_total"] == 4


def test_maybe_extract_records_metrics(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTO_EXTRACT", "1")
    messages = [HumanMessage(content="我喜欢莫奈的睡莲")]

    def fake_extract(conversation):
        return [
            {
                "action": "ADD", "kind": "preference", "entity": "莫奈",
                "content": "用户喜欢莫奈睡莲", "importance": 0.8,
                "scope": "user", "durability": "durable", "authority": "descriptive",
            },
            {
                "action": "ADD", "kind": "fact", "entity": "",
                "content": "用户这周去看展", "importance": 0.9,
                "scope": "thread", "durability": "temporary", "authority": "descriptive",
            },
        ]

    with patch.object(ex, "extract_memories", side_effect=fake_extract):
        _turns, result = ex.maybe_extract(messages, "test-user", 0)
    metrics = result["metrics"]
    assert metrics["extracted"] == 2
    assert metrics["passed"] == 1
    assert metrics["rejected"]["scope"] == 1
    assert metrics["rejection_rate"] == 0.5
    assert mtr.recent_extraction_metrics()[0]["user_id"] == "test-user"


def test_import_memories_dedups_and_writes():
    mi.add_memory("test-user", "用户喜欢莫奈睡莲", entity="莫奈", source="user_explicit")
    stats = mi.import_memories(
        "test-user",
        [
            {"content": "用户喜欢莫奈睡莲", "kind": "preference"},
            {"content": "用户偏好简洁回复", "kind": "preference", "importance": 0.8},
            {"content": "", "kind": "preference"},
        ],
    )
    assert stats["added"] == 1
    assert stats["dup"] == 1
    assert stats["invalid"] == 1
    rows = mi.list_memories("test-user")
    assert any(r["source"] == "imported" for r in rows)


def test_parse_import_file_txt_lines():
    raw = "用户喜欢莫奈\n用户偏好简洁回复\n\n  \n".encode("utf-8")
    items = mi.parse_import_file("记忆.txt", raw)
    assert len(items) == 2
    assert items[0] == {"content": "用户喜欢莫奈", "kind": "preference"}
    assert items[1] == {"content": "用户偏好简洁回复", "kind": "preference"}


def test_parse_import_file_json_items():
    raw = json.dumps([
        {"content": "用户喜欢莫奈", "kind": "preference", "entity": "莫奈",
         "importance": 0.9},
        {"content": "用户住在上海", "kind": "fact"},
        {"content": "  ", "kind": "fact"},          # 空内容跳过
        "not-a-dict",                                # 非对象跳过
        {"content": "用户画像", "kind": "profile"},  # profile 不允许 → preference
    ]).encode("utf-8")
    items = mi.parse_import_file("mem.json", raw)
    assert len(items) == 3
    assert items[0]["entity"] == "莫奈" and items[0]["importance"] == 0.9
    assert items[1]["kind"] == "fact"
    assert items[2]["kind"] == "preference"


def test_parse_import_file_json_wrapper():
    raw = b'{"items": [{"content": "A"}, {"content": "B"}]}'
    items = mi.parse_import_file("m.json", raw)
    assert [i["content"] for i in items] == ["A", "B"]


def test_parse_import_file_csv_header_and_plain():
    raw = (
        "content,kind,entity,importance\n"
        "用户喜欢莫奈,preference,莫奈,0.9\n"
        "用户住在上海,fact,上海,\n"
    ).encode("utf-8")
    items = mi.parse_import_file("m.csv", raw)
    assert len(items) == 2
    assert items[0]["content"] == "用户喜欢莫奈"
    assert items[0]["entity"] == "莫奈" and items[0]["importance"] == 0.9
    assert items[1]["kind"] == "fact" and items[1]["importance"] == 0.5

    plain = mi.parse_import_file("m.csv", "第一行\n第二行\n".encode("utf-8"))
    assert [i["content"] for i in plain] == ["第一行", "第二行"]


def test_parse_import_file_errors():
    with pytest.raises(ValueError, match="仅支持"):
        mi.parse_import_file("mem.xlsx", b"x")
    with pytest.raises(ValueError, match="UTF-8"):
        mi.parse_import_file("m.txt", b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="JSON"):
        mi.parse_import_file("m.json", b"{bad")
    with pytest.raises(ValueError, match="超过"):
        mi.parse_import_file("m.txt", b"x" * (2 * 1024 * 1024 + 1))


def test_parse_import_file_empty():
    assert mi.parse_import_file("m.txt", b"") == []
    assert mi.parse_import_file("m.txt", b"\n\n  \n") == []


# ══════════════ 会话滚动摘要 / 收藏清单 ══════════════
def _summary_tmp_db():
    tmp = Path(tempfile.mkdtemp()) / "conversations.db"
    summary_mod._DB_PATH = tmp
    summary_mod._db_ready = False
    db.close_all()
    return tmp


def _messages(n_turns: int) -> list:
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"第{i}轮问题"))
        msgs.append(AIMessage(content=f"第{i}轮回答"))
    return msgs


def test_load_save_roundtrip():
    _summary_tmp_db()
    summary_mod._save_summary("c1", "u1", "摘要内容", 10)
    assert summary_mod.load_summary("c1") == "摘要内容"
    assert summary_mod.load_summary("missing") == ""


def test_maybe_summarize_below_trigger_skips_llm():
    _summary_tmp_db()
    with patch.object(summary_mod, "_summarize") as mock:
        out = summary_mod.maybe_summarize(_messages(5), "c1", "u1", llm=lambda p: "x")
    mock.assert_not_called()
    assert out == ""


def test_maybe_summarize_incremental_and_stored():
    _summary_tmp_db()
    fake_llm = lambda prompt: "压缩后的摘要"
    out = summary_mod.maybe_summarize(_messages(10), "c1", "u1", llm=fake_llm)
    assert out == "压缩后的摘要"
    assert summary_mod.load_summary("c1") == "压缩后的摘要"
    with patch.object(summary_mod, "_summarize") as mock:
        out2 = summary_mod.maybe_summarize(_messages(11), "c1", "u1", llm=fake_llm)
    mock.assert_not_called()
    assert out2 == "压缩后的摘要"


def test_human_turn_count():
    assert summary_mod._human_turn_count(_messages(4)) == 4
    assert summary_mod._human_turn_count([]) == 0


def test_volume_trigger_fires_below_turn_threshold():
    _summary_tmp_db()
    fake_llm = lambda p: "体积触发的摘要"
    out = summary_mod.maybe_summarize(
        _messages(5), "c1", "u1", llm=fake_llm, volume_chars=20000
    )
    assert out == "体积触发的摘要"
    assert summary_mod.load_summary("c1") == "体积触发的摘要"


def test_low_volume_below_turn_threshold_skips():
    _summary_tmp_db()
    with patch.object(summary_mod, "_summarize") as mock:
        out = summary_mod.maybe_summarize(
            _messages(5), "c1", "u1", llm=lambda p: "x", volume_chars=100
        )
    mock.assert_not_called()
    assert out == ""


def _col_tmp_db():
    col._DB_PATH = Path(tempfile.mkdtemp()) / "agent_memory.db"
    col._db_ready = False
    db.close_all()


def test_save_and_list_collections():
    _col_tmp_db()
    col.save_collection("u1", "印象派最爱", ["睡莲", "日出·印象"])
    col.save_collection("u1", "巴洛克", ["下十字架"])
    cols = col.list_collections("u1")
    assert {c["name"] for c in cols} == {"印象派最爱", "巴洛克"}
    assert col.list_collections("other") == []


def test_save_collection_overwrites_same_name():
    _col_tmp_db()
    col.save_collection("u1", "k", ["a"])
    col.save_collection("u1", "k", ["b", "c"])
    cols = col.list_collections("u1")
    assert len(cols) == 1
    assert cols[0]["items"] == ["b", "c"]


def test_get_delete_rename_collection():
    _col_tmp_db()
    col.save_collection("u1", "印象派", ["睡莲"])
    got = col.get_collection("u1", "印象派")
    assert got is not None and got["items"] == ["睡莲"]
    assert col.get_collection("u1", "不存在") is None
    assert col.rename_collection("u1", "印象派", "最爱") is True
    assert col.get_collection("u1", "印象派") is None
    assert col.get_collection("u1", "最爱") is not None
    assert col.rename_collection("u1", "最爱", "最爱") is False
    assert col.rename_collection("u1", "不存在", "X") is False
    assert col.delete_collection("u1", "最爱") is True
    assert col.delete_collection("u1", "最爱") is False
    assert col.list_collections("u1") == []


# ══════════════════════════════════════════════════════════════════
# memory_items 存储层独有覆盖（原 test_memory_items.py）
# 依赖上方 autouse _isolate（库隔离 + _embed patch + MEMORY_USER_ID）
# ══════════════════════════════════════════════════════════════════


def test_add_same_content_updates_not_duplicates():
    a = mi.add_memory("test-user", "用户偏好莫奈睡莲", entity="莫奈")
    b = mi.add_memory("test-user", "用户偏好莫奈睡莲", entity="莫奈")
    assert a["id"] == b["id"]
    assert b["action"] == "update"
    assert len(mi.list_memories("test-user")) == 1


def test_no_entity_preferences_coexist():
    """无实体锚点的多条独立偏好不互相覆盖（批量导入/remember 不塌缩）。"""
    a = mi.add_memory("test-user", "用户喜欢莫奈", kind="preference")
    b = mi.add_memory("test-user", "用户偏好简洁回复", kind="preference")
    assert a["action"] == "create"
    assert b["action"] == "create"
    assert len(mi.list_memories("test-user")) == 2


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


def test_preference_items_stored_in_memory_items():
    mi.add_memory("test-user", "用户偏好莫奈睡莲系列", entity="莫奈",
                  kind="preference")
    items = [
        i for i in mi.list_memories("test-user", scope="user")
        if i.get("kind") == "preference"
    ]
    assert items and items[0]["content"] == "用户偏好莫奈睡莲系列"
    assert items[0]["kind"] == "preference"


def test_remember_tool_and_guard():
    from langchain_core.messages import ToolMessage

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
