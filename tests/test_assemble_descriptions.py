# tests/test_assemble_descriptions.py
"""
描述组装纯单测：QID 过滤、sitelink/extract 解析、回填合并。
不联网（解析函数用 fixture JSON）。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pandas as pd

from assemble_descriptions import (
    apply_descriptions,
    parse_extracts,
    parse_sitelinks,
    qids_needing_description,
    split_batches,
)


def test_split_batches():
    assert split_batches(list("abcdef"), 2) == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert split_batches([], 5) == []


def test_parse_sitelinks():
    payload = {
        "entities": {
            "Q82104": {"sitelinks": {"enwiki": {"title": "The Bedroom (Van Gogh painting)"}}},
            "Q99999": {"sitelinks": {"frwiki": {"title": "X"}}},  # 无 enwiki → 跳过
            "Q88888": {},
        }
    }
    out = parse_sitelinks(payload)
    assert out == {"Q82104": "The Bedroom (Van Gogh painting)"}


def test_parse_extracts():
    payload = {
        "query": {"pages": [
            {"title": "The Bedroom (Van Gogh painting)", "extract": "Painting by Vincent van Gogh."},
            {"title": "No Extract Page"},
            {"title": "Empty", "extract": "   "},
        ]}
    }
    out = parse_extracts(payload)
    assert out == {"The Bedroom (Van Gogh painting)": "Painting by Vincent van Gogh."}


def test_qids_needing_description():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.csv"
        pd.DataFrame([
            {"artwork_id": "Q1", "description": "has desc", "title": "a"},
            {"artwork_id": "Q2", "description": "   ", "title": "b"},
            {"artwork_id": "semart:abc", "description": "", "title": "c"},  # 非 QID 跳过
        ]).to_csv(path, index=False, encoding="utf-8-sig")
        assert qids_needing_description(path) == ["Q2"]
        assert qids_needing_description(path, limit=1) == ["Q2"]


def test_apply_descriptions_only_fills_empty():
    df = pd.DataFrame([
        {"artwork_id": "Q1", "description": "has desc", "title": "a"},
        {"artwork_id": "Q2", "description": "", "title": "b"},
        {"artwork_id": "semart:x", "description": "", "title": "c"},
    ])
    filled, total = apply_descriptions(df, {"Q2": "Intro for b.", "Q1": "should not override"})
    assert filled == 1
    assert total == 3
    assert df.loc[1, "description"] == "Intro for b."
    assert df.loc[0, "description"] == "has desc"      # 已有描述不覆盖
    assert df.loc[2, "description"] == ""               # 非 QID 不处理


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] assemble_descriptions 全部 {len(fns)} 个单测通过！")
