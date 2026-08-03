# tests/test_harvest_wikidata.py
"""
Wikidata 骨干采集脚本纯单测：SPARQL 绑定 → core schema 映射。
不联网、不调 SPARQL 端点。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from harvest_wikidata import (
    _clean_label,
    _dedup_key,
    _qid_from_uri,
    _year_bucket,
    _year_from_iso,
    artist_binding_to_row,
    artwork_binding_to_row,
)


def test_qid_from_uri():
    assert _qid_from_uri("http://www.wikidata.org/entity/Q3305213") == "Q3305213"
    assert _qid_from_uri("") == ""


def test_year_from_iso():
    assert _year_from_iso("1889-01-01T00:00:00Z") == 1889
    assert _year_from_iso("c. 1889") is None
    assert _year_from_iso("") is None


def test_year_bucket():
    assert _year_bucket(1889) == "1851-1900"
    assert _year_bucket(None) == ""


def test_clean_label():
    assert _clean_label("  A &amp; B  ") == "A & B"


def test_dedup_key():
    assert _dedup_key("Vincent van Gogh", "The Bedroom", 1889) == \
           _dedup_key("vincent  van gogh", "the  bedroom", 1889)
    assert _dedup_key("a", "b", 1) != _dedup_key("a", "b", 2)


def _binding(**kw):
    base = {
        "artist": {"value": "http://www.wikidata.org/entity/Q5582"},
        "artistLabel": {"value": "Vincent van Gogh"},
        "birth": {"value": "1853-03-30T00:00:00Z"},
        "death": {"value": "1890-07-29T00:00:00Z"},
        "natLabel": {"value": "Netherlands"},
        "movLabel": {"value": "Post-Impressionism"},
        "img": {"value": "http://commons.wikimedia.org/wiki/Special:FilePath/x.jpg"},
    }
    base.update(kw)
    return base


def test_artist_binding_to_row():
    row = artist_binding_to_row(_binding())
    assert row["artist_qid"] == "Q5582"
    assert row["name"] == "Vincent van Gogh"
    assert row["birth"] == 1853
    assert row["death"] == 1890
    assert row["nationality"] == "Netherlands"
    assert row["movement"] == "Post-Impressionism"
    assert row["source_api"] == "wikidata"


def test_artwork_binding_to_row():
    b = _binding(
        w={"value": "http://www.wikidata.org/entity/Q82104"},
        wLabel={"value": "The Bedroom"},
        collLabel={"value": "Art Institute of Chicago"},
        inception={"value": "1889-01-01T00:00:00Z"},
        materialLabel={"value": "oil paint"},
        genreLabel={"value": "interior view"},
        locLabel={"value": "Art Institute of Chicago"},
        seriesLabel={"value": ""},
        desc={"value": "Bedroom in the Yellow House"},
    )
    row = artwork_binding_to_row(b, "Q239303")
    assert row["artwork_id"] == "Q82104"
    assert row["artist_qid"] == "Q5582"
    assert row["collection_qid"] == "Q239303"
    assert row["year"] == 1889
    assert row["year_bucket"] == "1851-1900"
    assert row["material"] == "oil paint"
    assert row["description"] == "Bedroom in the Yellow House"
    assert row["dedup_key"] == "vincent van gogh|the bedroom|1889"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] harvest_wikidata 全部 {len(fns)} 个单测通过！")
