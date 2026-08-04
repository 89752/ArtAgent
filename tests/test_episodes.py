"""记忆系统 Phase 2（L3）单测：情景摘要 upsert/load/list/clear + 注入。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.memory.memory_items as mi
from src.memory.episodes import (
    clear_user_episodes,
    list_episodes,
    load_episode,
    upsert_episode,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="mem_episodes_")) / "agent_memory.db"
    mi._reset_for_tests(tmp)
    monkeypatch.setenv("MEMORY_USER_ID", "test-user")
    with patch("src.memory.memory_items._embed", return_value=None):
        yield


def test_upsert_and_load():
    upsert_episode("test-user", "conv-1", "上次聊了梵高的向日葵", 3)
    ep = load_episode("test-user", "conv-1")
    assert ep is not None
    assert ep["summary"] == "上次聊了梵高的向日葵"
    assert ep["turn_count"] == 3
    assert load_episode("test-user", "conv-404") is None
    assert load_episode("u-other", "conv-1") is None  # 用户隔离


def test_upsert_updates_same_conversation():
    upsert_episode("test-user", "conv-1", "第一版摘要", 2)
    upsert_episode("test-user", "conv-1", "第二版摘要", 5)
    eps = list_episodes("test-user")
    assert len(eps) == 1
    assert eps[0]["summary"] == "第二版摘要"
    assert eps[0]["turn_count"] == 5


def test_list_and_clear():
    upsert_episode("test-user", "conv-1", "摘要一", 2)
    upsert_episode("test-user", "conv-2", "摘要二", 4)
    assert len(list_episodes("test-user")) == 2
    assert clear_user_episodes("test-user") == 2
    assert list_episodes("test-user") == []


def test_load_memory_injects_episode():
    from src.agent.nodes.common import load_memory
    from src.agent.state import AgentState

    upsert_episode("test-user", "conv-9", "上次我们聊了莫奈晚年", 3)
    out = load_memory(AgentState(
        user_query="继续聊莫奈", conversation_id="conv-9",
    ))
    assert "上次对话回顾" in out["memory_block"]
    assert "莫奈晚年" in out["memory_block"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] episodes 全部 {len(fns)} 个单测通过")
