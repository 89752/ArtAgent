# tests/test_tools_unit.py
"""
工具层（非慢）集群合并单测：

- 新工具：color_analysis / aggregate_stats / compare_images /
  museum_search / wiki_lookup / image_lookup
- schema 驱动工具：exact_lookup / query_painter_knowledge
- 子智能体：delegate_task / run_tasks（结果契约、并发、超时、禁止嵌套、工具注册）

全程 mock 数据源 / 网络 / 视觉模型，不联网、不加载向量模型，秒级完成。
"""

import json
import os
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# 本地图片读取白名单扩展点：测试用临时目录生成的图片放行
os.environ.setdefault("ARTAGENT_IMAGE_ROOTS", tempfile.gettempdir())

import pandas as pd
from langchain_core.messages import AIMessage
from PIL import Image

from src.retrieval.structured_retriever import CORE_SCHEMA
from src.subagents.executor import (
    RESEARCH_TOOL_WHITELIST,
    _validate_result,
    run_tasks,
)
from src.tools.delegate import delegate_task
from src.tools.image_lookup import lookup_images
from src.tools.knowledge import query_painter_knowledge
from src.tools.registry import GENERAL_TOOLS, TOOL_BY_NAME
from src.tools.retrieval import _artwork_from_schema_row, exact_lookup


# ══════════════════════════════════════════════════════════════════
# 1. 新工具（color / stats / compare / museum / wiki）
# ══════════════════════════════════════════════════════════════════


def _tmp_png(color=(120, 40, 200)) -> str:
    p = Path(tempfile.mkdtemp()) / "test.png"
    Image.new("RGB", (64, 64), color).save(p)
    return str(p)


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), (120, 40, 200)).save(buf, format="PNG")
    return buf.getvalue()


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


def _patch_core():
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


# ══════════════════════════════════════════════════════════════════
# 2. schema 驱动工具（exact_lookup / knowledge / image_lookup）
# ══════════════════════════════════════════════════════════════════


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
    assert out["sample_work_images"] == [
        {"title": "Water Lilies", "image_file": "https://x/1.jpg"},
        {"title": "Impression, Sunrise", "image_file": "https://x/2.jpg"},
    ]


def test_image_lookup_uses_active_dataset_with_url():
    p1, p2 = _patch_retriever()
    with p1, p2:
        out = lookup_images(author="Monet", top_k=2)
    assert len(out) == 2
    assert out[0]["image_path"] == "https://x/1.jpg"   # URL 直通
    assert out[0]["image_file"] == "https://x/1.jpg"


# ══════════════════════════════════════════════════════════════════
# 3. 子智能体（delegate / run_tasks）
# ══════════════════════════════════════════════════════════════════


def test_no_nesting_and_no_memory_writes():
    assert "delegate_task" not in RESEARCH_TOOL_WHITELIST
    assert "remember" not in RESEARCH_TOOL_WHITELIST


def test_delegate_registered():
    assert "delegate_task" in TOOL_BY_NAME
    assert "delegate_task" in {t.name for t in GENERAL_TOOLS}


def test_validate_result_contract():
    assert (
        _validate_result('{"findings": "x", "sources": ["a"], "confidence": "high"}')
        is not None
    )
    assert _validate_result('{"findings": "x"}') is None
    assert _validate_result("不是 JSON") is None
    assert (
        _validate_result(
            '```json\n{"findings": "x", "sources": [], "confidence": "low"}\n```'
        )
        is not None
    )


def _fake_llm(sequence):
    class FakeLLM:
        def __init__(self):
            self.i = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            item = sequence[min(self.i, len(sequence) - 1)]
            self.i += 1
            return item(messages) if callable(item) else item

    return FakeLLM()


def test_run_tasks_completed():
    ok = json.dumps(
        {"findings": "莫奈", "sources": ["睡莲"], "confidence": "high"}
    )
    with patch(
        "src.subagents.executor.get_deterministic_llm",
        return_value=_fake_llm([AIMessage(content=ok)]),
    ):
        results = run_tasks(
            [
                {"task_id": "t1", "prompt": "研究莫奈"},
                {"task_id": "t2", "prompt": "研究梵高"},
            ]
        )
    assert len(results) == 2
    assert all(r.status == "completed" for r in results)
    assert results[0].result["findings"] == "莫奈"


def test_run_tasks_corrects_missing_fields():
    bad = AIMessage(content='{"findings": "x"}')
    good = AIMessage(
        content=json.dumps(
            {"findings": "x", "sources": ["a"], "confidence": "low"}
        )
    )
    with patch(
        "src.subagents.executor.get_deterministic_llm",
        return_value=_fake_llm([bad, good]),
    ):
        results = run_tasks([{"task_id": "t1", "prompt": "p"}])
    assert results[0].status == "completed"
    assert results[0].result["sources"] == ["a"]


def test_run_tasks_timeout(monkeypatch):
    monkeypatch.setenv("SUBAGENT_TIMEOUT_SEC", "0.2")

    def slow(messages):
        time.sleep(1)
        return AIMessage(
            content=json.dumps(
                {"findings": "x", "sources": ["a"], "confidence": "low"}
            )
        )

    with patch(
        "src.subagents.executor.get_deterministic_llm",
        return_value=_fake_llm([slow]),
    ):
        results = run_tasks([{"task_id": "t1", "prompt": "p"}])
    assert results[0].status == "timed_out"


def test_run_tasks_concurrency_cap(monkeypatch):
    monkeypatch.setenv("SUBAGENT_MAX_CONCURRENT", "1")

    def slow(messages):
        time.sleep(0.3)
        return AIMessage(
            content=json.dumps(
                {"findings": "x", "sources": ["a"], "confidence": "low"}
            )
        )

    with patch(
        "src.subagents.executor.get_deterministic_llm",
        return_value=_fake_llm([slow]),
    ):
        start = time.monotonic()
        results = run_tasks(
            [{"task_id": f"t{i}", "prompt": "p"} for i in range(2)]
        )
        elapsed = time.monotonic() - start
    assert elapsed >= 0.55  # 并发=1 时两个 0.3s 任务应串行
    assert all(r.status == "completed" for r in results)


def test_run_tasks_empty():
    results = run_tasks([])
    assert results[0].status == "failed"


def test_delegate_tool_payload():
    ok = json.dumps(
        {"findings": "x", "sources": ["a"], "confidence": "high"}
    )
    with patch(
        "src.subagents.executor.get_deterministic_llm",
        return_value=_fake_llm([AIMessage(content=ok)]),
    ):
        out = delegate_task.invoke(
            {"tasks": [{"description": "调研", "prompt": "p"}]}
        )
    payload = json.loads(out)
    assert payload["status"] == "completed"
    assert payload["results"][0]["findings"] == "x"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] tools_unit 集群全部 {len(fns)} 个单测通过")
