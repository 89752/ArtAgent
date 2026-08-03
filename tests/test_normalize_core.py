# tests/test_normalize_core.py
"""
核心库合并归一化纯单测：AIC 行映射、dedup 合并、流派回填。
不联网、不依赖真实 CSV。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from normalize_core import (
    _dedup_key,
    _normalize_artist_name,
    _year_bucket,
    _year_from_text,
    aic_row_to_core,
    join_movement,
    merge_rows,
    semart_row_to_core,
)


def test_year_bucket():
    assert _year_bucket(1889) == "1851-1900"
    assert _year_bucket(None) == ""


def test_dedup_key():
    assert _dedup_key("Vincent van Gogh", "The Bedroom", 1889) == \
           _dedup_key("vincent van gogh", "the bedroom", 1889)


def test_normalize_artist_name():
    assert _normalize_artist_name("GOGH, Vincent van") == "Vincent van GOGH"
    assert _normalize_artist_name("Vincent van Gogh") == "Vincent van Gogh"
    assert _normalize_artist_name("") == ""


def test_year_from_text():
    assert _year_from_text("1526 and after 1528") == 1526
    assert _year_from_text("1770-75") == 1770
    assert _year_from_text("") is None


def test_semart_row_to_core():
    row = semart_row_to_core({
        "IMAGE_FILE": "18759-guard301.jpg",
        "TITLE": "Landscape with a Fisherman's Tent",
        "AUTHOR": "GUARDI, Francesco",
        "DATE": "1770-75",
        "TIMEFRAME": "1751-1800",
        "TECHNIQUE": "Oil on canvas, 49 x 77 cm",
        "TYPE": "landscape",
        "SCHOOL": "Italian",
        "DESCRIPTION": "A poetic scene with fishing boats.",
    })
    assert row["artwork_id"].startswith("semart:")
    assert row["artist_name"] == "Francesco GUARDI"       # 倒序名已归一
    assert row["year"] == 1770
    assert row["year_bucket"] == "1751-1800"
    assert row["genre"] == "landscape"
    assert row["school"] == "Italian"
    assert row["material"].startswith("Oil on canvas")
    assert row["source_api"] == "semart"
    # 归一化后的 dedup_key 与 Wikidata 显示序（"Francesco Guardi"）一致
    assert row["dedup_key"] == _dedup_key("Francesco Guardi", row["title"], 1770)


def test_merge_semart_with_wikidata():
    semart = {
        "artwork_id": "semart:abc", "title": "The Bedroom", "artist_qid": "",
        "artist_name": "Vincent van GOGH", "year": 1889, "source_api": "semart",
        "description": "A" * 300, "genre": "interior", "school": "Dutch",
        "movement": "", "material": "", "image_url": "img.jpg", "collection_name": "",
        "location": "", "inception": "1889", "series": "", "license": "",
        "dimensions_raw": "", "width_cm": "", "height_cm": "",
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    wd = {
        "artwork_id": "Q82104", "title": "The Bedroom", "artist_qid": "Q5582",
        "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "wikidata",
        "description": "short", "movement": "Post-Impressionism", "genre": "",
        "material": "oil paint", "image_url": "", "collection_name": "AIC",
        "location": "Chicago", "inception": "1889", "series": "", "license": "",
        "dimensions_raw": "", "width_cm": "", "height_cm": "",
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    merged = merge_rows([semart, wd])[0]
    assert merged["artist_qid"] == "Q5582"
    assert len(merged["description"]) == 300          # SemArt 长描述胜出
    assert merged["movement"] == "Post-Impressionism"
    assert merged["school"] == "Dutch"
    assert merged["genre"] == "interior"
    assert merged["source_api"] == "semart;wikidata"


def test_aic_row_to_core():
    row = aic_row_to_core({
        "object_id": "aic:28560",
        "title": "The Bedroom",
        "artist": "Vincent van Gogh",
        "year": 1889,
        "year_display": "1889",
        "year_bucket": "1851-1900",
        "medium": "Oil on canvas",
        "school": "Post-Impressionism",
        "description": "Famous bedroom.",
        "image_url": "http://x/img.jpg",
        "license": "CC0",
        "location": "Europe · Art Institute of Chicago",
        "dimensions_raw": "73.6 × 92.3 cm",
        "width_cm": 73.6,
        "height_cm": 92.3,
    })
    assert row["artwork_id"] == "aic:28560"
    assert row["artist_qid"] == ""
    assert row["collection_name"] == "Art Institute of Chicago"
    assert row["movement"] == "Post-Impressionism"
    assert row["source_api"] == "aic"
    assert row["year"] == 1889


def test_merge_prefers_qid_and_longer_description():
    wd = {
        "artwork_id": "Q82104", "title": "The Bedroom", "artist_qid": "Q5582",
        "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "wikidata",
        "description": "short", "movement": "", "genre": "", "material": "",
        "image_url": "", "collection_name": "", "location": "", "inception": "1889",
        "series": "", "license": "", "dimensions_raw": "", "width_cm": "", "height_cm": "",
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    aic = {
        "artwork_id": "aic:28560", "title": "The Bedroom", "artist_qid": "",
        "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "aic",
        "description": "A much longer curatorial description of the bedroom.", "movement": "Post-Impressionism",
        "genre": "", "material": "oil paint", "image_url": "http://x.jpg",
        "collection_name": "Art Institute of Chicago", "location": "Chicago", "inception": "1889",
        "series": "", "license": "CC0", "dimensions_raw": "73.6 × 92.3 cm",
        "width_cm": 73.6, "height_cm": 92.3,
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    merged = merge_rows([wd, aic])[0]
    assert merged["artist_qid"] == "Q5582"          # 保留 Wikidata QID
    assert merged["description"] == aic["description"]  # 长描述胜出
    assert merged["movement"] == "Post-Impressionism"
    assert merged["material"] == "oil paint"
    assert merged["image_url"] == "http://x.jpg"
    assert merged["source_api"] == "wikidata;aic"


def test_merge_distinct_rows_keep_both():
    a = {"artwork_id": "Q1", "title": "A", "artist_qid": "Q9", "artist_name": "X",
         "year": 1800, "description": "d1", "source_api": "wikidata", "dedup_key": "x|a|1800"}
    b = {"artwork_id": "Q2", "title": "B", "artist_qid": "Q8", "artist_name": "Y",
         "year": 1850, "description": "d2", "source_api": "wikidata", "dedup_key": "y|b|1850"}
    assert len(merge_rows([a, b])) == 2


def test_merge_keeps_distinct_qids_for_same_key():
    """同名同题同年但 QID 不同 → 两幅不同作品，不合并（误合并审计修复）。"""
    wd_a = {"artwork_id": "Q18011394", "title": "Portrait of a Man", "artist_qid": "Q41254",
            "artist_name": "Frans Hals", "year": 1650, "source_api": "wikidata",
            "description": "A", "dedup_key": "frans hals|portrait of a man|1650"}
    wd_b = {"artwork_id": "Q18025591", "title": "Portrait of a Man", "artist_qid": "Q41254",
            "artist_name": "Frans Hals", "year": 1650, "source_api": "wikidata",
            "description": "B", "dedup_key": "frans hals|portrait of a man|1650"}
    merged = merge_rows([wd_a, wd_b])
    assert len(merged) == 2
    assert {r["artwork_id"] for r in merged} == {"Q18011394", "Q18025591"}


def test_merge_semart_with_wikidata_still_merges():
    """无 QID 的 semart 行与有 QID 的 Wikidata 行：同作品，仍合并。"""
    semart = {"artwork_id": "semart:abc", "title": "The Bedroom", "artist_qid": "",
              "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "semart",
              "description": "A" * 100, "dedup_key": "vincent van gogh|the bedroom|1889"}
    wd = {"artwork_id": "Q82104", "title": "The Bedroom", "artist_qid": "Q5582",
          "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "wikidata",
          "description": "short", "dedup_key": "vincent van gogh|the bedroom|1889"}
    merged = merge_rows([semart, wd])
    assert len(merged) == 1
    assert merged[0]["artwork_id"] == "Q82104"


def test_join_movement_fills_from_artist_qid():
    artworks = [
        {"artist_qid": "Q5582", "movement": "", "title": "t"},
        {"artist_qid": "Q999", "movement": "Baroque", "title": "u"},
        {"artist_qid": "", "movement": "", "title": "v"},
    ]
    join_movement(artworks, {"Q5582": "Post-Impressionism"})
    assert artworks[0]["movement"] == "Post-Impressionism"
    assert artworks[1]["movement"] == "Baroque"      # 已有值不覆盖
    assert artworks[2]["movement"] == ""             # 无 QID 跳过


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] normalize_core 全部 {len(fns)} 个单测通过！")
