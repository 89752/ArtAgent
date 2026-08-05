"""记忆容量/淘汰与画像聚合单测：容量/淘汰、向量后端回落、跨线程画像聚合。

全程 patch embedding 与 Chroma/LLM，不耗额度、不碰真实索引。
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.memory.memory_items as mi


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="mem_capacity_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    monkeypatch.setenv("MEMORY_MAX_ITEMS_PER_USER", "200")
    monkeypatch.setenv("MEMORY_MAX_CHARS_PER_USER", "40000")
    monkeypatch.delenv("MEMORY_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("MEMORY_PROFILE_REFRESH", raising=False)
    with patch("src.memory.memory_items._embed", return_value=None):
        yield


# ── 容量 / 淘汰 ────────────────────────────────────────────────
def test_evicts_lowest_importance_when_over_cap(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_ITEMS_PER_USER", "2")
    mi.add_memory("test-user", "低价值记忆A", entity="A", importance=0.1)
    mi.add_memory("test-user", "高价值记忆B", entity="B", importance=0.9)
    mi.add_memory("test-user", "中价值记忆C", entity="C", importance=0.5)
    contents = [i["content"] for i in mi.list_memories("test-user")]
    assert "低价值记忆A" not in contents
    assert "高价值记忆B" in contents and "中价值记忆C" in contents
    # 审计里有 evict 记录
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
    assert contents == ["短"]  # 超长条目单独就超过预算，被淘汰；短条目保留


# ── 向量后端（Chroma 可选，失败回落） ─────────────────────────
def test_chroma_backend_falls_back_on_unavailable(monkeypatch):
    monkeypatch.setenv("MEMORY_VECTOR_BACKEND", "chroma")
    mi.add_memory("test-user", "莫奈睡莲", entity="莫奈", importance=0.8)

    def boom(*args, **kwargs):
        raise RuntimeError("chroma down")

    with patch("src.retrieval.hybrid.get_or_create_chroma_collection", side_effect=boom):
        # 写入不中断、检索回落全量
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


# ── 跨线程用户画像聚合 ─────────────────────────────────────────
def test_profile_disabled_by_default():
    from src.memory.profile import maybe_refresh_profile

    def boom(prompt):
        raise AssertionError("关闭时不应调用 LLM")

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
    # 新鲜期内不重复刷新
    assert maybe_refresh_profile("test-user", llm=lambda p: "不应被调用")["skipped"] == "fresh"


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
    import os

    os.environ["MEMORY_PROFILE_REFRESH"] = "1"
    maybe_refresh_profile("test-user", llm=lambda p: "用户喜欢印象派与莫奈。")
    os.environ.pop("MEMORY_PROFILE_REFRESH", None)
    out = load_memory(AgentState(user_query="推荐几幅画"))
    assert "【用户画像】" in out["memory_block"]
    assert "莫奈" in out["memory_block"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] memory_capacity 全部 {len(fns)} 个单测通过")
