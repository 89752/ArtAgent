"""service 渲染层纯单测：转义、图片 URL、来源收集、僵尸任务重置。"""

import os
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="artagent_render_test_")
os.environ["INDEX_DIR"] = _TMP
os.environ["ARTAGENT_MEMORY_DIR"] = _TMP

from langchain_core.messages import ToolMessage

from web import service as svc
from src.data import documents_store


def test_thumb_url_variants():
    assert svc._thumb_url("") == ""
    assert svc._thumb_url("https://x/a.jpg") == "https://x/a.jpg"
    assert svc._thumb_url("28496-early05.jpg") == "/api/images/28496-early05.jpg"
    assert svc._thumb_url("../a.jpg") == "/api/images/a.jpg"


def test_answer_block_escapes_html():
    out = svc._answer_block("<script>alert(1)</script>")
    assert "&lt;script&gt;" in out
    assert "<script>" not in out


def test_chain_detail_escapes_user_content():
    out = svc._chain_detail("rewrite_split", {"user_query": "<img src=x onerror=1>"})
    assert "&lt;img" in out
    assert "<img" not in out


def test_artwork_grid_uses_url_and_escapes():
    html = svc._artwork_grid(
        [{"title": "<b>x</b>", "author": "A", "image_file": "28496-early05.jpg"}],
        True,
    )
    assert "/api/images/28496-early05.jpg" in html
    assert "&lt;b&gt;" in html
    assert '<img src="/api/images/' in html


def test_collect_sources_dedup_and_caps():
    msgs = [
        ToolMessage(content='[{"title": "A", "author": "X"}]', tool_call_id="1"),
        ToolMessage(content='[{"title": "A", "author": "X"}]', tool_call_id="2"),
        ToolMessage(
            content='[{"source": "user_pdf_text", "doc_name": "画册", "page": 3}]',
            tool_call_id="3",
        ),
        ToolMessage(
            content='[{"source": "user_table", "dataset_id": "t1"}]',
            tool_call_id="4",
        ),
    ]
    evidence = [
        {"source": "user_pdf_text", "doc_name": "笔记", "page": 7},
        {"url": "https://example.com/wiki/Monet"},           # web 兜底
    ]
    sources = svc._collect_sources(msgs, [{"title": "B", "author": "Y"}], evidence)
    labels = [s["label"] for s in sources]
    assert len(sources) == len(set(labels))          # 去重
    assert "《画册》第3页" in labels
    assert "表格《t1》" in labels
    assert any("《A》" in l for l in labels)
    assert any("《B》" in l for l in labels)
    assert "《笔记》第7页" in labels
    assert any("https://example.com" in l for l in labels)
    assert len(sources) <= 6


def test_init_db_resets_zombie_processing(tmp_path):
    documents_store._reset_for_tests(tmp_path / "docs.db")
    documents_store.init_db()
    documents_store.add_document(
        doc_id="zombie1", kind="pdf", doc_name="z.pdf", status="processing"
    )
    documents_store.add_document(
        doc_id="ok1", kind="table", doc_name="t.csv", status="pending_confirm"
    )
    # 模拟服务重启：再次 init_db
    documents_store.init_db()
    d1 = documents_store.get_document("zombie1")
    assert d1["status"] == "failed"
    assert "重启" in d1["error"]
    d2 = documents_store.get_document("ok1")
    assert d2["status"] == "pending_confirm"  # 等待用户确认的不能被重置
