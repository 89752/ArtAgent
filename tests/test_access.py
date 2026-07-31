# tests/test_access.py
"""
数据访问层（src/data/access.py）纯单测：
不加载 SemArt 数据集、不调 LLM、不联网，秒级完成。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.access import (
    EVIDENCE_SNIPPET_LEN,
    format_evidence_block,
    fuzzy_match,
    row_to_artwork_dict,
)

LONG_DESC = "A swirling night sky over a quiet town. " * 20  # 远超 200 字符


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TITLE": "The Starry Night",
                "AUTHOR": "Vincent van Gogh",
                "DATE": "1889",
                "TECHNIQUE": "Oil on canvas",
                "SCHOOL": "Dutch",
                "TIMEFRAME": "1851-1900",
                "IMAGE_FILE": "starry.jpg",
                "DESCRIPTION": LONG_DESC,
            },
            {
                "TITLE": "The Kiss",
                "AUTHOR": "Gustav Klimt",
                "DATE": "1908",
                "TECHNIQUE": "Oil and gold leaf on canvas",
                "SCHOOL": "Austrian",
                "TIMEFRAME": "1901-1950",
                "IMAGE_FILE": "kiss.jpg",
                "DESCRIPTION": "short desc",
            },
            {
                "TITLE": "Water Lilies",
                "AUTHOR": "Claude Monet",
                "DATE": "1906",
                "TECHNIQUE": "Oil on canvas",
                "SCHOOL": "French",
                "TIMEFRAME": "1901-1950",
                "IMAGE_FILE": "lilies.jpg",
                "DESCRIPTION": "",
            },
        ]
    )


# ── fuzzy_match ────────────────────────────────────────────────
def test_fuzzy_exact_case_insensitive():
    df = _df()
    hit = fuzzy_match(df, "TITLE", "the starry night")
    assert len(hit) == 1 and hit.iloc[0]["IMAGE_FILE"] == "starry.jpg"


def test_fuzzy_author_single_token():
    # 兼容旧 exact_lookup("GOGH") 行为：分词包含匹配
    df = _df()
    hit = fuzzy_match(df, "AUTHOR", "GOGH")
    assert len(hit) == 1 and hit.iloc[0]["AUTHOR"] == "Vincent van Gogh"


def test_fuzzy_author_longest_token_first():
    # "Vincent van Gogh" → 最长词 Vincent 优先，仍能命中
    df = _df()
    hit = fuzzy_match(df, "AUTHOR", "Vincent van Gogh")
    assert len(hit) == 1 and hit.iloc[0]["AUTHOR"] == "Vincent van Gogh"


def test_fuzzy_partial_contains():
    df = _df()
    hit = fuzzy_match(df, "TITLE", "Kiss")
    assert len(hit) == 1 and hit.iloc[0]["TITLE"] == "The Kiss"


def test_fuzzy_no_match_returns_empty():
    df = _df()
    assert fuzzy_match(df, "TITLE", "zzz-nonexistent").empty


def test_fuzzy_empty_inputs():
    df = _df()
    assert fuzzy_match(df, "TITLE", "").empty
    assert fuzzy_match(df.iloc[0:0], "TITLE", "Kiss").empty


# ── row_to_artwork_dict ───────────────────────────────────────
def test_row_to_artwork_dict_from_series():
    df = _df()
    d = row_to_artwork_dict(df.iloc[0])
    assert d["title"] == "The Starry Night"
    assert d["author"] == "Vincent van Gogh"
    assert d["image_file"] == "starry.jpg"
    assert len(d["description_snippet"]) == EVIDENCE_SNIPPET_LEN + 3  # 截断 + "..."
    assert d["description_snippet"].endswith("...")


def test_row_to_artwork_dict_short_desc_not_truncated():
    df = _df()
    d = row_to_artwork_dict(df.iloc[1])
    assert d["description_snippet"] == "short desc"


def test_row_to_artwork_dict_no_truncation_mode():
    df = _df()
    d = row_to_artwork_dict(df.iloc[0], snippet_len=None)
    assert d["description_snippet"] == LONG_DESC


def test_row_to_artwork_dict_from_chroma_meta():
    # Chroma metadata 用小写 key，且图片字段叫 file
    meta = {
        "title": "The Kiss",
        "author": "Gustav Klimt",
        "date": "1908",
        "technique": "Oil",
        "school": "Austrian",
        "timeframe": "1901-1950",
        "file": "kiss.jpg",
        "description": "short desc",
    }
    d = row_to_artwork_dict(meta)
    assert d["image_file"] == "kiss.jpg"
    assert d["title"] == "The Kiss"


def test_row_to_artwork_dict_missing_fields():
    df = _df()
    d = row_to_artwork_dict(df.iloc[2])
    assert d["description_snippet"] == ""
    assert d["title"] == "Water Lilies"


# ── format_evidence_block ─────────────────────────────────────
def test_format_basic():
    docs = [{"title": "The Kiss", "date": "1908", "description_snippet": "short desc"}]
    out = format_evidence_block(docs, "- {title} ({date}): {description_snippet}")
    assert out == "- The Kiss (1908): short desc"


def test_format_missing_date_no_empty_parens():
    docs = [{"title": "The Kiss", "description_snippet": "short desc"}]
    out = format_evidence_block(docs, "- {title} ({date}): {description_snippet}")
    assert "()" not in out
    assert out == "- The Kiss: short desc"


def test_format_missing_desc_no_trailing_colon():
    docs = [{"title": "Water Lilies", "date": "1906", "description_snippet": ""}]
    out = format_evidence_block(docs, "  - {title} ({date}): {description_snippet}")
    assert out == "  - Water Lilies (1906)"


def test_format_pipe_template_missing_author():
    docs = [{"title": "X", "description_snippet": "s"}]
    out = format_evidence_block(docs, "- {author} | {title}: {description_snippet}")
    assert out == "- X: s"


def test_format_pipe_template_full():
    docs = [{"author": "Monet", "title": "Water Lilies", "description_snippet": "s"}]
    out = format_evidence_block(docs, "- {author} | {title}: {description_snippet}")
    assert out == "- Monet | Water Lilies: s"


def test_format_web_results_template():
    docs = [{"title": "t", "snippet": "s", "url": "http://x"}]
    out = format_evidence_block(docs, "- {title}: {snippet} ({url})")
    assert out == "- t: s (http://x)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 access.py 全部 {len(fns)} 个单测通过！")
