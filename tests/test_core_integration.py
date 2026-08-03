# tests/test_core_integration.py
"""
M3 运行时改造纯单测：core 数据源接入（RetrievalSource / _format_result /
_thumb_data_uri / 懒注册门控）。不联网、不加载模型、不依赖真实 core 数据。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.retrieval import structured_retriever as sr
from src.retrieval.base import RetrievalResult
from src.tools.retrieval import _format_result
from web import service as service_mod


def test_retrieval_source_accepts_core():
    r = RetrievalResult(content="x", source="core", metadata={"title": "t"})
    assert r.source == "core"


def test_format_result_core_shape():
    meta = {
        "title": "The Bedroom",
        "artist": "Vincent van Gogh",
        "year_display": "1889",
        "year": 1889,
        "material": "Oil on canvas",
        "movement": "Post-Impressionism",
        "year_bucket": "1851-1900",
        "image_url": "https://www.artic.edu/iiif/2/x/full/843,/0/default.jpg",
        "description": "A" * 300,
    }
    out = _format_result(RetrievalResult(content="c", source="core", score=0.5, metadata=meta))
    assert out["title"] == "The Bedroom"
    assert out["author"] == "Vincent van Gogh"
    assert out["date"] == "1889"
    assert out["technique"] == "Oil on canvas"
    assert out["school"] == "Post-Impressionism"
    assert out["timeframe"] == "1851-1900"
    assert out["image_file"] == meta["image_url"]
    assert len(out["description_snippet"]) <= 200 + 3
    assert out["relevance_score"] == 0.5
    assert "source" not in out  # 与 semart 同契约，可进 UI 配图卡片


def test_thumb_url_url_passthrough():
    assert service_mod._thumb_url("https://example.com/a.jpg") == "https://example.com/a.jpg"
    assert service_mod._thumb_url("http://example.com/a.jpg") == "http://example.com/a.jpg"
    assert service_mod._thumb_url("") == ""


def test_thumb_url_local_file_becomes_api_url():
    # 本地文件不再内联 base64，返回可缓存 URL（含 basename 防穿越）
    assert service_mod._thumb_url("28496-early05.jpg") == "/api/images/28496-early05.jpg"
    assert service_mod._thumb_url("../evil/../x.jpg") == "/api/images/x.jpg"


def test_get_structured_retriever_core_missing_raises():
    old = sr.CORE_DATA_PATH
    sr.CORE_DATA_PATH = Path("C:/nonexistent/artworks_core.csv")
    sr._REGISTRY.pop("core", None)
    try:
        try:
            sr.get_structured_retriever("core")
            raise AssertionError("数据缺失时应抛 KeyError")
        except KeyError as e:
            assert "核心库数据未就绪" in str(e)
    finally:
        sr.CORE_DATA_PATH = old
        sr._REGISTRY.pop("core", None)


def test_get_structured_retriever_core_registers_from_csv():
    old = sr.CORE_DATA_PATH
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "artworks_core.csv"
        pd.DataFrame([{
            "artwork_id": "Q1", "title": "T", "artist_qid": "Q2", "artist_name": "A",
            "collection_name": "", "location": "", "inception": "", "year": 1800,
            "year_bucket": "1776-1825", "material": "", "genre": "", "movement": "",
            "series": "", "description": "d", "image_url": "", "license": "",
            "dimensions_raw": "", "width_cm": "", "height_cm": "", "source_api": "wikidata",
            "dedup_key": "a|t|1800",
        }]).to_csv(csv_path, index=False, encoding="utf-8-sig")
        sr.CORE_DATA_PATH = csv_path
        sr._REGISTRY.pop("core", None)
        try:
            retriever = sr.get_structured_retriever("core")
            assert retriever.dataset_id == "core"
            assert retriever.schema.supports_timeline is True
            assert retriever.schema.supports_recommendation is True
            assert retriever.source == "core"
        finally:
            sr.CORE_DATA_PATH = old
            sr._REGISTRY.pop("core", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] core_integration 全部 {len(fns)} 个单测通过！")
