"""P1 新工具纯单测（mock 数据源/网络/视觉，秒级）。"""

import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# 本地图片读取白名单扩展点：测试用临时目录生成的图片放行
os.environ.setdefault("ARTAGENT_IMAGE_ROOTS", tempfile.gettempdir())

from PIL import Image

from src.retrieval.structured_retriever import CORE_SCHEMA


def _tmp_png(color=(120, 40, 200)) -> str:
    p = Path(tempfile.mkdtemp()) / "test.png"
    Image.new("RGB", (64, 64), color).save(p)
    return str(p)


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), (120, 40, 200)).save(buf, format="PNG")
    return buf.getvalue()


# ── color_analysis ───────────────────────────────────────────────
def test_color_analysis_local_image():
    from src.tools.color_analysis import color_analysis

    path = _tmp_png()
    with patch(
        "src.tools.image_lookup.lookup_images",
        return_value=[{"title": "T", "author": "A", "date": "1900", "image_path": path}],
    ):
        out = color_analysis.invoke({"title": "T"})
    assert out[0]["success"] is True
    assert out[0]["dominant_colors"]
    assert out[0]["brightness_contrast"]
    assert out[0]["saturation"] in ("vivid", "muted", "moderate")


def test_color_analysis_url_image():
    from src.tools.color_analysis import color_analysis

    with (
        patch(
            "src.tools.image_lookup.lookup_images",
            return_value=[{"title": "T", "author": "A", "date": "1900", "image_path": "https://x/1.jpg"}],
        ),
        patch("src.utils.http.download_bytes", return_value=_png_bytes()),
    ):
        out = color_analysis.invoke({"title": "T"})
    assert out[0]["success"] is True
    assert out[0]["dominant_colors"]


# ── aggregate_stats ──────────────────────────────────────────────
def _patch_core():
    import pandas as pd

    df = pd.DataFrame([
        {"title": "A", "artist": "Monet", "year_display": "1900", "year_bucket": "1851-1900",
         "material": "Oil", "movement": "Impressionism", "description": "x"},
        {"title": "B", "artist": "Monet", "year_display": "1880", "year_bucket": "1851-1900",
         "material": "Oil", "movement": "Impressionism", "description": "y"},
        {"title": "C", "artist": "Rembrandt", "year_display": "1640", "year_bucket": "1601-1700",
         "material": "Oil", "movement": "Baroque", "description": "z"},
    ])
    retriever = SimpleNamespace(schema=CORE_SCHEMA, df=df)
    hybrid = SimpleNamespace(active_dataset="core")
    return (
        patch("src.retrieval.hybrid.get_hybrid_retriever", return_value=hybrid),
        patch("src.retrieval.structured_retriever.get_structured_retriever",
              return_value=retriever),
    )


def test_aggregate_stats_groups_and_filters():
    from src.tools.aggregate_stats import aggregate_stats

    p1, p2 = _patch_core()
    with p1, p2:
        out = aggregate_stats.invoke({"group_by": "school"})
    assert out["total"] == 3
    by_value = {g["value"]: g for g in out["groups"]}
    assert by_value["Impressionism"]["count"] == 2
    assert by_value["Impressionism"]["ratio"] == round(2 / 3, 3)
    with p1, p2:
        out2 = aggregate_stats.invoke(
            {"group_by": "timeframe", "filters": {"author": "Monet"}}
        )
    assert out2["total"] == 2
    assert out2["groups"][0]["value"] == "1851-1900"


# ── compare_images ───────────────────────────────────────────────
def test_compare_images_calls_vision_once():
    from src.tools.compare_images import compare_images

    path_a, path_b = _tmp_png((255, 0, 0)), _tmp_png((0, 0, 255))

    class _FakeVision:
        def invoke(self, msgs):
            # 两图同帧：应包含 2 个 image_url 块 + 1 个 text 块
            blocks = msgs[0].content
            assert sum(1 for b in blocks if b.get("type") == "image_url") == 2
            return SimpleNamespace(content="对比结果：A 偏暖，B 偏冷。")

    with (
        patch(
            "src.tools.image_lookup.lookup_images",
            side_effect=[
                [{"title": "A", "author": "X", "date": "1900", "image_path": path_a}],
                [{"title": "B", "author": "Y", "date": "1910", "image_path": path_b}],
            ],
        ),
        patch("src.utils.llm.get_vision_llm", return_value=_FakeVision()),
    ):
        out = compare_images.invoke({"title_a": "A", "title_b": "B"})
    assert out["success"] is True
    assert "偏暖" in out["comparison"]


def test_compare_images_missing_returns_error():
    from src.tools.compare_images import compare_images

    with patch("src.tools.image_lookup.lookup_images", return_value=[]):
        out = compare_images.invoke({"title_a": "A", "title_b": "B"})
    assert out["success"] is False


def test_compare_images_url_images():
    from src.tools.compare_images import compare_images

    class _FakeVision:
        def invoke(self, msgs):
            blocks = msgs[0].content
            assert sum(1 for b in blocks if b.get("type") == "image_url") == 2
            return SimpleNamespace(content="对比结果：两幅画色彩差异明显。")

    with (
        patch(
            "src.tools.image_lookup.lookup_images",
            side_effect=[
                [{"title": "A", "author": "X", "date": "1900", "image_path": "https://x/a.jpg"}],
                [{"title": "B", "author": "Y", "date": "1910", "image_path": "https://x/b.jpg"}],
            ],
        ),
        patch("src.utils.http.download_bytes", return_value=_png_bytes()),
        patch("src.utils.llm.get_vision_llm", return_value=_FakeVision()),
    ):
        out = compare_images.invoke({"title_a": "A", "title_b": "B"})
    assert out["success"] is True
    assert "差异" in out["comparison"]


def test_analyze_image_file_url():
    import pandas as pd

    from src.tools.image_lookup import _analyze_image_file

    df = pd.DataFrame([{
        "title": "T", "author": "A", "artist": "A", "year_display": "1900",
        "date": "1900", "material": "Oil", "technique": "Oil",
        "movement": "Baroque", "school": "Baroque", "description": "完整描述",
        "image_url": "https://x/1.jpg", "image_file": "https://x/1.jpg",
    }])
    retriever = SimpleNamespace(schema=CORE_SCHEMA, df=df)
    hybrid = SimpleNamespace(active_dataset="core")

    class _FakeVision:
        def invoke(self, msgs):
            blocks = msgs[0].content
            assert blocks[0]["type"] == "image_url"
            assert blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            return SimpleNamespace(content="视觉分析结果")

    with (
        patch("src.retrieval.hybrid.get_hybrid_retriever", return_value=hybrid),
        patch(
            "src.retrieval.structured_retriever.get_structured_retriever",
            return_value=retriever,
        ),
        patch("src.utils.http.download_bytes", return_value=_png_bytes()),
        patch("src.utils.llm.get_vision_llm", return_value=_FakeVision()),
    ):
        out = _analyze_image_file("https://x/1.jpg", "general")
    assert out["success"] is True
    assert out["analysis"] == "视觉分析结果"
    assert out["title"] == "T"


# ── museum_search / wiki_lookup（mock get_json，不联网） ─────────
def test_museum_search_mocked():
    from src.tools.museum_search import museum_search

    def fake_get(url):
        if "/search" in url:
            return {"objectIDs": [1, 2]}
        return {
            "title": "Water Lilies", "artistDisplayName": "Claude Monet",
            "objectDate": "1906", "medium": "Oil", "department": "Europe",
            "primaryImage": "https://img/1.jpg", "objectURL": "https://met/1",
            "isPublicDomain": True,
        }

    with patch("src.tools.museum_search.get_json", side_effect=fake_get):
        out = museum_search.invoke({"artist": "Monet", "top_k": 2})
    assert out["success"] is True
    assert out["results"][0]["title"] == "Water Lilies"
    assert out["results"][0]["is_public_domain"] is True


def test_wiki_lookup_mocked():
    from src.tools.wiki_lookup import wiki_lookup

    def fake_get(url):
        assert "zh.wikipedia.org" in url
        return {
            "type": "standard",
            "title": "巴洛克",
            "description": "艺术风格",
            "extract": "巴洛克是一种艺术风格……" * 20,
            "content_urls": {"desktop": {"page": "https://zh.wikipedia.org/wiki/巴洛克"}},
        }

    with patch("src.tools.wiki_lookup.get_json", side_effect=fake_get):
        out = wiki_lookup.invoke({"entity": "巴洛克"})
    assert out["success"] is True
    assert out["lang"] == "zh"
    assert len(out["extract"]) <= 1600
