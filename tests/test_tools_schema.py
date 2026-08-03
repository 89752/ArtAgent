# tests/test_tools_schema.py
"""
schema 驱动工具改造纯单测：exact_lookup / query_painter_knowledge / image_lookup
按 active dataset 工作（patch 掉 hybrid/structured 注册，不加载真实数据、不联网）。
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.retrieval.structured_retriever import CORE_SCHEMA
from src.tools.image_lookup import lookup_images
from src.tools.knowledge import query_painter_knowledge
from src.tools.retrieval import _artwork_from_schema_row, exact_lookup


def _core_df():
    return pd.DataFrame([
        {"artwork_id": "Q1", "title": "Water Lilies", "artist": "Claude Monet",
         "year": 1906, "year_display": "1906", "year_bucket": "1901-1950",
         "material": "Oil on canvas", "movement": "Impressionism", "school": "",
         "genre": "landscape", "description": "A pond scene.", "image_url": "https://x/1.jpg"},
        {"artwork_id": "Q2", "title": "Impression, Sunrise", "artist": "Claude Monet",
         "year": 1872, "year_display": "1872", "year_bucket": "1851-1900",
         "material": "Oil on canvas", "movement": "Impressionism", "school": "",
         "genre": "landscape", "description": "A harbor at sunrise.", "image_url": "https://x/2.jpg"},
    ])


def _patch_retriever():
    retriever = SimpleNamespace(schema=CORE_SCHEMA, df=_core_df())
    hybrid = SimpleNamespace(active_dataset="core")
    return (
        patch("src.retrieval.hybrid.get_hybrid_retriever", return_value=hybrid),
        patch("src.retrieval.structured_retriever.get_structured_retriever",
              return_value=retriever),
    )


def test_artwork_from_schema_row_core_style():
    row = _core_df().iloc[0].to_dict()
    out = _artwork_from_schema_row(CORE_SCHEMA, row)
    assert out["title"] == "Water Lilies"
    assert out["author"] == "Claude Monet"
    assert out["date"] == "1906"
    assert out["technique"] == "Oil on canvas"
    assert out["school"] == "Impressionism"
    assert out["timeframe"] == "1901-1950"
    assert out["image_file"] == "https://x/1.jpg"
    assert out["description_snippet"] == "A pond scene."


def test_exact_lookup_uses_active_dataset():
    p1, p2 = _patch_retriever()
    with p1, p2:
        out = exact_lookup.invoke({"author": "Monet", "top_k": 5})
    assert len(out) == 2
    assert out[0]["author"] == "Claude Monet"
    assert "source" not in out[0]


def test_exact_lookup_school_filter_core():
    p1, p2 = _patch_retriever()
    with p1, p2:
        out = exact_lookup.invoke({"school": "impressionism", "top_k": 5})
    assert len(out) == 2


def test_knowledge_uses_active_dataset():
    p1, p2 = _patch_retriever()
    with p1, p2:
        out = query_painter_knowledge.invoke({"painter_name": "Monet"})
    assert out["found"] is True
    assert out["works_count"] == 2
    assert out["main_schools"] == ["Impressionism"]
    assert out["active_timeframes"] == ["1901-1950", "1851-1900"]
    assert out["sample_works"] == ["Water Lilies", "Impression, Sunrise"]


def test_image_lookup_uses_active_dataset_with_url():
    p1, p2 = _patch_retriever()
    with p1, p2:
        out = lookup_images(author="Monet", top_k=2)
    assert len(out) == 2
    assert out[0]["image_path"] == "https://x/1.jpg"   # URL 直通
    assert out[0]["image_file"] == "https://x/1.jpg"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] tools_schema 全部 {len(fns)} 个单测通过！")
