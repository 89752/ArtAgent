# tests/test_build_extended.py
"""
扩展画作库构建脚本纯单测：清洗、尺寸解析、去重键、字段映射、合格过滤。
不联网、不加载模型、不写数据。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_extended_dataset import (
    _clean_desc,
    _dedup_key,
    _is_fetchable,
    _parse_dimensions_cm,
    _year_bucket,
    aic_record_to_row,
)


def test_year_bucket():
    assert _year_bucket(1889) == "1851-1900"
    assert _year_bucket(1900) == "1851-1900"
    assert _year_bucket(1901) == "1901-1950"
    assert _year_bucket(None) == ""


def test_clean_desc_strips_html_and_entities():
    raw = "<p>The bedroom &amp; studio</p>\n\n<p>Second version&nbsp;of the scene.</p>"
    out = _clean_desc(raw)
    assert "The bedroom & studio" in out
    assert "<" not in out and "&nbsp;" not in out
    assert "\n" not in out


def test_clean_desc_empty():
    assert _clean_desc(None) == ""
    assert _clean_desc("   ") == ""


def test_parse_dimensions_cm_main():
    raw = "73.6 × 92.3 cm (29 × 36 5/8 in.); Framed: 88.9 × 108 × 8.9 cm (35 × 42 1/2 × 3 1/2 in.)"
    assert _parse_dimensions_cm(raw) == (73.6, 92.3)


def test_parse_dimensions_inches_converted():
    assert _parse_dimensions_cm("24 × 18 in.") == (60.96, 45.72)


def test_parse_dimensions_none_or_unparsable():
    assert _parse_dimensions_cm(None) == (None, None)
    assert _parse_dimensions_cm("no dimensions here") == (None, None)


def test_dedup_key_normalization():
    a = _dedup_key("Vincent van Gogh", "The Bedroom", 1889)
    b = _dedup_key("vincent  van gogh", "the  bedroom", 1889)
    c = _dedup_key("Vincent van Gogh", "The Bedroom", 1890)
    assert a == b
    assert a != c


def test_is_fetchable():
    base = {
        "artwork_type_title": "Painting",
        "image_id": "abc",
        "description": "A description.",
    }
    assert _is_fetchable(base)
    assert not _is_fetchable({**base, "artwork_type_title": "Sculpture"})
    assert not _is_fetchable({**base, "image_id": None})
    # 空描述也入库（只进表不进向量），由 index 步骤分流
    assert _is_fetchable({**base, "description": ""})


def test_aic_record_to_row_full_mapping():
    rec = {
        "id": 28560,
        "title": "The Bedroom",
        "artist_title": "Vincent van Gogh",
        "date_start": 1889,
        "date_end": 1889,
        "date_display": "1889",
        "medium_display": "Oil on canvas",
        "classification_title": "oil on canvas",
        "style_title": "Post-Impressionism",
        "description": "<p>Perhaps the most famous depiction of a bedroom.</p>",
        "image_id": "6644829f-f292-c5c4-a73c-0356a6fdbf0d",
        "dimensions": "73.6 × 92.3 cm (29 × 36 5/8 in.)",
        "department_title": "Painting and Sculpture of Europe",
        "artwork_type_title": "Painting",
        "copyright_notice": None,
    }
    row = aic_record_to_row(rec)
    assert row["object_id"] == "aic:28560"
    assert row["artist"] == "Vincent van Gogh"
    assert row["year"] == 1889
    assert row["year_bucket"] == "1851-1900"
    assert row["school"] == "Post-Impressionism"
    assert "<p>" not in row["description"]
    assert row["image_url"].startswith("https://www.artic.edu/iiif/2/6644829f")
    assert row["license"] == "CC0/Public domain (AIC)"
    assert row["width_cm"] == 73.6 and row["height_cm"] == 92.3
    assert "Art Institute of Chicago" in row["location"]
    assert row["source_api"] == "aic"
    assert row["dedup_key"] == "vincent van gogh|the bedroom|1889"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] build_extended_dataset 全部 {len(fns)} 个单测通过！")
