"""index_core 续跑选择逻辑纯单测（不加载模型、不连向量库）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.index_core import _select_rows_to_index


def _df():
    return pd.DataFrame(
        {"artwork_id": ["Q1", "Q2", "Q3"], "description": ["a", "b", "c"]}
    )


def test_resume_skips_existing_ids():
    out = _select_rows_to_index(_df(), {"Q1", "Q3"}, force=False)
    assert out["artwork_id"].tolist() == ["Q2"]


def test_resume_no_existing_keeps_all():
    out = _select_rows_to_index(_df(), set(), force=False)
    assert len(out) == 3


def test_force_keeps_all_even_when_existing():
    out = _select_rows_to_index(_df(), {"Q1", "Q2", "Q3"}, force=True)
    assert len(out) == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] index_core 全部 {len(fns)} 个单测通过")
