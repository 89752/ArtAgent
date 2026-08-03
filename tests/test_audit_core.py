# tests/test_audit_core.py
"""
核心库体检脚本纯单测：compute_stats 统计正确性。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pandas as pd

from audit_core import collision_groups, collision_summary, compute_stats, render_report


def _df():
    return pd.DataFrame([
        {"artwork_id": "Q1", "title": "A", "description": "desc", "image_url": "http://x",
         "year": 1800, "movement": "Neoclassicism", "school": "", "artist_qid": "Q9",
         "source_api": "wikidata", "dedup_key": "a|b|1800"},
        {"artwork_id": "Q2", "title": "B", "description": "", "image_url": "",
         "year": "", "movement": "", "school": "Dutch", "artist_qid": "",
         "source_api": "semart", "dedup_key": "c|d|"},
        {"artwork_id": "Q3", "title": "C", "description": "x", "image_url": "http://y",
         "year": 1950, "movement": "Abstract", "school": "", "artist_qid": "Q8",
         "source_api": "wikidata;aic", "dedup_key": "e|f|1950"},
    ])


def test_compute_stats():
    s = compute_stats(_df())
    assert s["total"] == 3
    assert s["described"] == 2
    assert s["image"] == 2
    assert s["year"] == 2
    assert s["year_min"] == 1800 and s["year_max"] == 1950
    assert s["movement"] == 2
    assert s["school"] == 1
    assert s["artist_qid"] == 2
    assert s["dup_artwork_ids"] == 0
    assert s["dedup_collision_groups"] == 0
    assert s["sources"] == {"wikidata": 2, "semart": 1, "aic": 1}


def test_compute_stats_dedup_conflict():
    df = _df()
    df.loc[2, "dedup_key"] = "a|b|1800"  # 与 Q1 同 dedup_key 但不同 artwork_id
    s = compute_stats(df)
    assert s["dedup_collision_groups"] == 1


def test_render_report_contains_key_metrics():
    report = render_report(compute_stats(_df()))
    assert "总作品数" in report and "有描述" in report and "来源分布" in report


def test_collision_groups_and_summary():
    df = _df()
    df.loc[2, "dedup_key"] = "a|b|1800"          # 与 Q1 冲突
    df.loc[2, "image_url"] = df.loc[0, "image_url"]  # 同图 → 疑似重复记录
    groups = collision_groups(df)
    assert len(groups) == 2
    s = collision_summary(groups)
    assert s["groups"] == 1 and s["same_image_groups"] == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] audit_core 全部 {len(fns)} 个单测通过！")
