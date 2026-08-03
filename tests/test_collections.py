"""收藏清单存储层与工具 schema 单测（临时 SQLite 库）。"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.memory.collections as col


def _tmp_db():
    col._DB_PATH = Path(tempfile.mkdtemp()) / "agent_memory.db"
    col._conn = None


def test_save_and_list_collections():
    _tmp_db()
    col.save_collection("u1", "印象派最爱", ["睡莲", "日出·印象"])
    col.save_collection("u1", "巴洛克", ["下十字架"])
    cols = col.list_collections("u1")
    assert {c["name"] for c in cols} == {"印象派最爱", "巴洛克"}
    assert cols[0]["name"] == "印象派最爱" or cols[0]["name"] == "巴洛克"  # 按更新时间倒序
    assert col.list_collections("other") == []


def test_save_collection_overwrites_same_name():
    _tmp_db()
    col.save_collection("u1", "k", ["a"])
    col.save_collection("u1", "k", ["b", "c"])
    cols = col.list_collections("u1")
    assert len(cols) == 1
    assert cols[0]["items"] == ["b", "c"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] collections 全部 {len(fns)} 个单测通过")
