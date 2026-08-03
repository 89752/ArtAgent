"""Agent 主动记忆存储层单测（临时 SQLite 库）。"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.memory.agent_memory as mem


def _tmp_db():
    mem._DB_PATH = Path(tempfile.mkdtemp()) / "agent_memory.db"
    mem._conn = None


def test_remember_recall_forget():
    _tmp_db()
    mem.remember("u1", "preferred_style", "喜欢浓烈奔放的风格")
    hits = mem.recall("u1", "浓烈")
    assert hits and hits[0]["key"] == "preferred_style"
    assert mem.recall("u1", "不存在的词") == []
    assert mem.forget("u1", "preferred_style") is True
    assert mem.recall("u1", "浓烈") == []
    assert mem.forget("u1", "preferred_style") is False


def test_remember_overwrites_same_key():
    _tmp_db()
    mem.remember("u1", "k", "v1")
    mem.remember("u1", "k", "v2")
    hits = mem.recall("u1", "v2")
    assert len(hits) == 1
    assert hits[0]["content"] == "v2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] agent_memory 全部 {len(fns)} 个单测通过")
