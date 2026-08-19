"""Web/API 层统一集成测试：service 渲染、会话/记忆/反馈/任务/文档级联、限流与重生成。

TestClient + 临时 SQLite 隔离，离线运行（patch embedding / 视觉 / 引擎）。
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

import api
from src.analysis import engine as analysis_engine
from src.analysis import store as analysis_store
from src.data import db
from src.data import documents_store
from src.ingestion import pipeline as pipeline_mod
from src.ingestion import table_pipeline as tp
from src.memory import conversations as conv_mod
from src.memory import feedback as fb_mod
from src.memory import memory_items as mi_mod
from src.memory import summary as summary_mod
from src.observability import runs as runs_mod
from src.retrieval import hybrid as hybrid_mod
from src.tasks import store as tasks_mod
from src.platform import users as users_store
from web import service as svc

_TMP = Path(tempfile.mkdtemp(prefix="artagent_api_test_"))
os.environ["RATE_LIMIT_RPM"] = "0"

documents_store.DB_PATH = _TMP / "documents.db"
documents_store._LEGACY_STATUS_FILE = _TMP / "doc_status.json"
conv_mod._DB_PATH = _TMP / "conversations.db"
conv_mod._db_ready = False
summary_mod._DB_PATH = _TMP / "conversations.db"
summary_mod._db_ready = False
fb_mod._DB_PATH = _TMP / "feedback.db"
fb_mod._db_ready = False
runs_mod._DB_PATH = _TMP / "observability.db"
runs_mod._db_ready = False
tasks_mod._DB_PATH = _TMP / "tasks.db"
tasks_mod._db_ready = False
mi_mod._DB_PATH = _TMP / "agent_memory.db"
mi_mod._db_ready = False
db.close_all()
users_store._reset_for_tests(_TMP / "platform.db")
analysis_store.DB_PATH = _TMP / "user_images.db"
analysis_engine.USER_IMAGE_ROOT = _TMP / "uploads" / "user_images"


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _api_clean():
    api._rate_buckets.clear()
    # 重新断言各存储路径：pytest 命令行顺序可能让其他文件的 fixture
    # 改写本文件的模块全局 _DB_PATH，导致跨文件数据串扰
    documents_store.DB_PATH = _TMP / "documents.db"
    documents_store._LEGACY_STATUS_FILE = _TMP / "doc_status.json"
    conv_mod._DB_PATH = _TMP / "conversations.db"
    conv_mod._db_ready = False
    summary_mod._DB_PATH = _TMP / "conversations.db"
    summary_mod._db_ready = False
    fb_mod._DB_PATH = _TMP / "feedback.db"
    fb_mod._db_ready = False
    runs_mod._DB_PATH = _TMP / "observability.db"
    runs_mod._db_ready = False
    mi_mod._DB_PATH = _TMP / "agent_memory.db"
    mi_mod._db_ready = False
    db.close_all()
    tasks_mod._reset_for_tests(_TMP / "tasks.db")
    users_store._reset_for_tests(_TMP / "platform.db")
    analysis_store.DB_PATH = _TMP / "user_images.db"
    analysis_engine.USER_IMAGE_ROOT = _TMP / "uploads" / "user_images"
    documents_store.init_db()
    analysis_store.init_db()
    with patch("src.memory.memory_items._embed", return_value=None):
        try:
            yield
        finally:
            api._rate_buckets.clear()


# ══════════════ service 渲染层 ══════════════
def test_thumb_url_variants():
    assert svc._thumb_url("") == ""
    assert svc._thumb_url("https://x/a.jpg") == "https://x/a.jpg"
    assert svc._thumb_url("28496-early05.jpg") == "/api/images/28496-early05.jpg"
    assert svc._thumb_url("../a.jpg") == "/api/images/a.jpg"


def test_parse_artworks_from_knowledge_sample_images():
    import json

    msgs = [ToolMessage(content=json.dumps({
        "painter": "Monet",
        "matched_author": "Claude Monet",
        "sample_work_images": [
            {"title": "Water Lilies", "image_file": "https://x/1.jpg"},
            {"title": "Impression, Sunrise", "image_file": "https://x/2.jpg"},
        ],
    }), tool_call_id="k1")]
    out = svc._parse_artworks_from_messages(msgs)
    assert out == [
        {"title": "Water Lilies", "author": "Claude Monet",
         "date": "", "image_file": "https://x/1.jpg"},
        {"title": "Impression, Sunrise", "author": "Claude Monet",
         "date": "", "image_file": "https://x/2.jpg"},
    ]


def test_answer_block_escapes_html():
    out = svc._answer_block("<script>alert(1)</script>")
    assert "&lt;script&gt;" in out
    assert "<script>" not in out


def test_chain_detail_escapes_user_content():
    out = svc._chain_detail(
        "ask_user", {"pending_clarification": "<img src=x onerror=1>"}
    )
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
        {"url": "https://example.com/wiki/Monet"},
    ]
    sources = svc._collect_sources(msgs, [{"title": "B", "author": "Y"}], evidence)
    labels = [s["label"] for s in sources]
    assert len(sources) == len(set(labels))
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
    documents_store.init_db()
    assert documents_store.get_document("zombie1")["status"] == "failed"
    assert "重启" in documents_store.get_document("zombie1")["error"]
    assert documents_store.get_document("ok1")["status"] == "pending_confirm"


# ══════════════ Web API（会话/文档/图片） ══════════════
def test_chat_empty_message_streams_done(client):
    with client.stream(
        "POST", "/api/chat", json={"message": "", "session_id": "s-empty"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert '"type": "done"' in text
    assert '"sources"' in text


def test_chat_message_too_long_rejected(client):
    r = client.post("/api/chat", json={"message": "x" * 8001, "session_id": "s"})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert "参数错误" in body["error"]


def test_session_attachment_flow(client):
    documents_store.add_document(
        doc_id="web-test-doc",
        kind="table",
        doc_name="测试表格.csv",
        status="pending_confirm",
    )
    r = client.post("/api/sessions/s1/attachment", json={"doc_id": "nope"})
    assert r.status_code == 404
    assert r.json()["ok"] is False
    r = client.post("/api/sessions/s1/attachment", json={"doc_id": "web-test-doc"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    msgs = client.get("/api/sessions/s1").json()["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "attachment"
    assert msgs[0]["doc_id"] == "web-test-doc"
    r = client.get("/api/sessions?offset=0&limit=10")
    data = r.json()
    assert "items" in data and "total" in data and "has_more" in data
    assert any(s["session_id"] == "s1" for s in data["items"])
    r = client.patch("/api/sessions/s1", json={"title": "重命名后的会话"})
    assert r.status_code == 200
    s1 = next(s for s in client.get("/api/sessions?offset=0&limit=10").json()["items"]
              if s["session_id"] == "s1")
    assert s1["title"] == "重命名后的会话"
    r = client.delete("/api/documents/web-test-doc")
    assert r.status_code == 200
    assert client.get("/api/sessions/s1").json()["messages"] == []
    r = client.delete("/api/sessions/s1")
    assert r.status_code == 200


def test_upload_rejects_unsupported_type(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("evil.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "仅支持" in r.json()["error"]


def test_schema_validation(client):
    r = client.post("/api/documents/doc1/schema", json={"entity_col": ""})
    assert r.status_code == 422
    assert r.json()["ok"] is False


def test_image_route_serves_and_blocks_traversal(client):
    r = client.get("/api/images/..%2F..%2Fapi.py")
    assert r.status_code == 404
    r = client.get("/api/images/28496-early05.jpg")
    img_path = Path("data/core/images/28496-early05.jpg")
    if not img_path.exists():
        img_path = Path("SemArt/Images/28496-early05.jpg")
    if img_path.exists():
        assert r.status_code == 200
        assert r.headers.get("cache-control", "").startswith("public")
    else:
        assert r.status_code == 404


# ══════════════ 反馈 / 记忆 / 指标 / 限流 ══════════════
def test_feedback_api(client):
    r = client.post("/api/feedback", json={
        "session_id": "s-fb", "rating": 1, "reason": "", "comment": "很棒",
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True
    r = client.post("/api/feedback", json={
        "session_id": "s-fb", "rating": -1, "reason": "引用不充分", "comment": "",
    })
    assert r.status_code == 200
    r = client.post("/api/feedback", json={"session_id": "s", "rating": 0})
    assert r.status_code == 422
    assert client.get("/api/feedback").json()["total"] == 2


def test_memory_api_v2(client):
    mi_mod.add_memory("web_user", "用户喜欢莫奈睡莲", kind="preference",
                      entity="莫奈", source="user_explicit")
    mi_mod.add_memory("web_user", "用户住在上海", kind="fact",
                      entity="上海", source="extracted")
    summary_mod._save_summary("conv-x", "web_user", "上次聊了莫奈", 2)
    data = client.get("/api/memory").json()
    items = {i["content"]: i for i in data["items"]}
    assert "用户喜欢莫奈睡莲" in items
    assert items["用户住在上海"]["source"] == "extracted"
    assert items["用户住在上海"]["kind"] == "fact"
    r = client.delete(f"/api/memory/{items['用户喜欢莫奈睡莲']['id']}")
    assert r.status_code == 200
    assert r.json()["memory"] == 1
    r = client.delete("/api/memory/not-exist")
    assert r.status_code == 404
    r = client.delete("/api/memory")
    assert r.status_code == 200
    assert r.json()["memory"] == 0
    assert summary_mod.load_summary("conv-x") == ""


def test_memory_import_file_api(client):
    client.delete("/api/memory")  # 自包含：先清空再验证
    r = client.post(
        "/api/memory/import-file",
        files={
            "file": (
                "mem.txt",
                "用户喜欢莫奈\n用户住在上海\n".encode("utf-8"),
                "text/plain",
            )
        },
    )
    assert r.status_code == 200
    assert r.json()["stats"]["added"] == 2
    # 同内容再导 → 全部去重
    r2 = client.post(
        "/api/memory/import-file",
        files={
            "file": (
                "mem.txt",
                "用户喜欢莫奈\n用户住在上海\n".encode("utf-8"),
                "text/plain",
            )
        },
    )
    assert r2.status_code == 200
    assert r2.json()["stats"]["dup"] == 2
    # JSON 文件（带 kind/entity）
    r3 = client.post(
        "/api/memory/import-file",
        files={
            "file": (
                "mem.json",
                json.dumps([
                    {"content": "用户喜欢梵高", "kind": "fact", "entity": "梵高"},
                ]).encode("utf-8"),
                "application/json",
            )
        },
    )
    assert r3.status_code == 200
    assert r3.json()["stats"]["added"] == 1
    data = client.get("/api/memory").json()
    contents = {i["content"]: i for i in data["items"]}
    assert "用户喜欢莫奈" in contents
    assert contents["用户喜欢梵高"]["kind"] == "fact"
    assert contents["用户喜欢梵高"]["entity"] == "梵高"
    # 非法类型 / 空文件 → 400
    r4 = client.post(
        "/api/memory/import-file",
        files={"file": ("bad.xlsx", b"x", "application/octet-stream")},
    )
    assert r4.status_code == 400
    assert "仅支持" in r4.json()["error"]
    r5 = client.post(
        "/api/memory/import-file",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r5.status_code == 400
    assert "没有可导入" in r5.json()["error"]


def test_metrics_api(client):
    runs_mod.record_run(
        session_id="s-m", intent="general", tools=["web_search"],
        context_chars=1000, tool_rounds=1, latency_ms=300.0,
    )
    m = client.get("/api/metrics").json()
    assert m["count"] >= 1
    assert m["latency_ms"]["avg"] >= 300.0
    assert m["tool_calls"]["web_search"] >= 1


def test_rate_limit_429(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RPM", "60")
    monkeypatch.setenv("RATE_LIMIT_BURST", "2")
    api._rate_buckets.clear()
    assert client.get("/api/bootstrap").status_code == 200
    assert client.get("/api/bootstrap").status_code == 200
    r = client.get("/api/bootstrap")
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "30"
    monkeypatch.setenv("RATE_LIMIT_RPM", "0")
    api._rate_buckets.clear()


def test_request_id_passthrough(client):
    r = client.get("/api/bootstrap", headers={"X-Request-Id": "req-abc"})
    assert r.headers.get("X-Request-Id") == "req-abc"


# ══════════════ 任务化与重试 ══════════════
def test_tasks_lifecycle_and_payload():
    tid = tasks_mod.create_task("ingest_pdf", {"doc_id": "d1", "kind": "pdf"})
    assert tasks_mod.get_task(tid)["status"] == "pending"
    tasks_mod.update_task(tid, status="processing", progress=30)
    t = tasks_mod.get_task(tid)
    assert t["status"] == "processing"
    assert t["payload"]["doc_id"] == "d1"
    tasks_mod.update_task(tid, status="done", progress=100)
    assert tasks_mod.get_task(tid)["status"] == "done"
    assert tasks_mod.get_task(tid)["finished_at"]


def test_tasks_interrupted_recovery_and_retry():
    tid = tasks_mod.create_task("ingest_table", {"kind": "table"})
    tasks_mod.update_task(tid, status="processing")
    assert tasks_mod.mark_interrupted_on_startup() == 1
    assert tasks_mod.get_task(tid)["status"] == "interrupted"
    assert tasks_mod.reset_task(tid) is True
    assert tasks_mod.get_task(tid)["status"] == "pending"
    assert tasks_mod.reset_task(tid) is False


def test_tasks_invalid_status_rejected():
    tid = tasks_mod.create_task("x")
    with pytest.raises(ValueError):
        tasks_mod.update_task(tid, status="bogus")


def test_tasks_api_and_retry(client, monkeypatch):
    tid = tasks_mod.create_task("ingest_pdf", {"doc_id": "d1", "kind": "pdf"})
    tasks_mod.update_task(tid, status="failed", error="解析失败")
    r = client.get(f"/api/tasks/{tid}")
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "failed"
    calls = []

    def fake_ingest(
        doc_id, doc_name, pdf_path, kb_id,
        task_id=None, force_pdfplumber=False, user_id="web_user",
    ):
        calls.append((doc_id, task_id))

    monkeypatch.setattr(svc, "ingest_document", fake_ingest)
    r = client.post(f"/api/tasks/{tid}/retry")
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "pending"
    assert calls == [("d1", tid)]
    assert client.post(f"/api/tasks/{tid}/retry").status_code == 400
    assert client.post("/api/tasks/nope/retry").status_code == 404


# ══════════════ 文档生命周期级联 ══════════════
def _doc_isolate(tmp: Path):
    old_db = documents_store.DB_PATH
    old_uploads = pipeline_mod.UPLOADS_DIR
    old_legacy = documents_store._LEGACY_STATUS_FILE
    db_path = tmp / "documents.db"
    documents_store._reset_for_tests(db_path)
    documents_store._LEGACY_STATUS_FILE = Path("/nonexistent/doc_status.json")
    documents_store.init_db()
    pipeline_mod.UPLOADS_DIR = tmp / "uploads"
    tp.UPLOADS_DIR = pipeline_mod.UPLOADS_DIR
    return lambda: (
        setattr(documents_store, "DB_PATH", old_db),
        setattr(documents_store, "_LEGACY_STATUS_FILE", old_legacy),
        setattr(pipeline_mod, "UPLOADS_DIR", old_uploads),
        setattr(tp, "UPLOADS_DIR", old_uploads),
    )


def _fresh_hybrid():
    hybrid_mod._hybrid = None
    return hybrid_mod.get_hybrid_retriever()


def test_delete_pdf_cascades():
    tmp = Path(tempfile.mkdtemp(prefix="s6_pdf_"))
    restore = _doc_isolate(tmp)
    try:
        hybrid = _fresh_hybrid()
        doc_id = "pdf-abc"
        work_dir = pipeline_mod.UPLOADS_DIR / "default" / doc_id
        work_dir.mkdir(parents=True)
        pdf_path = work_dir / "document.pdf"
        pdf_path.write_text("fake pdf")
        documents_store.add_document(
            doc_id=doc_id, kind="pdf", doc_name="x.pdf",
            status="done", file_path=str(pdf_path),
            file_size=7, pages=1, text_chunks=2, image_pages=0,
        )
        with patch.object(pipeline_mod, "delete_pdf_vectors", return_value={"text": 2, "images": 0}) as mock_del:
            result = svc.delete_document(doc_id)
        assert result["doc_id"] == doc_id
        assert result["kind"] == "pdf"
        assert result["vectors"] == {"text": 2, "images": 0}
        assert result["files_removed"] is True
        assert result["db_deleted"] is True
        assert documents_store.get_document(doc_id) is None
        assert not work_dir.exists()
        mock_del.assert_called_once_with(doc_id)
    finally:
        restore()


def test_delete_table_unregisters_and_resets_active():
    tmp = Path(tempfile.mkdtemp(prefix="s6_tbl_"))
    restore = _doc_isolate(tmp)
    try:
        hybrid = _fresh_hybrid()
        doc_id = "tbl-xyz"
        dataset_id = tp.table_dataset_id(doc_id)
        documents_store.add_document(
            doc_id=doc_id, kind="table", doc_name="books.csv",
            status="active",
            metadata={
                "dataset_id": dataset_id,
                "table_path": str(tmp / "books.csv"),
                "confirmed_schema": {
                    "entity_col": "书名", "group_axis_col": None,
                    "description_col": "", "image_col": None,
                },
                "display_name": "书单",
                "supports_timeline": False,
                "supports_recommendation": False,
            },
        )
        retriever = tp.register_structured_dataset(
            dataset_id,
            tp.TableSchema(entity_col="书名", description_col=""),
            source="user_table",
            df=pd.DataFrame(columns=["书名"]),
        )
        hybrid.register(dataset_id, retriever)
        hybrid.active_dataset = dataset_id
        result = svc.delete_document(doc_id)
        assert result["doc_id"] == doc_id
        assert result["kind"] == "table"
        assert result["active_dataset_reset"] == "core"
        assert hybrid.active_dataset == "core"
        assert dataset_id not in hybrid.retrievers
        assert documents_store.get_document(doc_id) is None
    finally:
        restore()


def test_delete_nonexistent_raises():
    tmp = Path(tempfile.mkdtemp(prefix="s6_miss_"))
    restore = _doc_isolate(tmp)
    try:
        documents_store._reset_for_tests(tmp / "documents.db")
        documents_store._LEGACY_STATUS_FILE = Path("/nonexistent/doc_status.json")
        documents_store.init_db()
        try:
            svc.delete_document("no-such-doc")
            raise AssertionError("应抛出 KeyError")
        except KeyError as e:
            assert "no-such-doc" in str(e)
    finally:
        restore()


def test_pdf_ingest_preserves_non_default_user_status(monkeypatch):
    """真实登录用户的 PDF 完成后不能因默认用户查询而卡在 processing。"""
    tmp = Path(tempfile.mkdtemp(prefix="s6_owner_"))
    restore = _doc_isolate(tmp)
    try:
        doc_id = "owned-pdf"
        user_id = "user-42"
        work_dir = pipeline_mod.UPLOADS_DIR / "default" / doc_id
        work_dir.mkdir(parents=True)
        pdf_path = work_dir / "document.pdf"
        pdf_path.write_bytes(b"fake pdf")
        documents_store.add_document(
            doc_id=doc_id, kind="pdf", user_id=user_id, doc_name="large.pdf",
            status="processing", file_path=str(pdf_path), file_size=8,
        )

        plan = type("Plan", (), {"pages": [], "distribution": {}})()
        monkeypatch.setattr(pipeline_mod, "classify_document", lambda _path: plan)
        monkeypatch.setattr(pipeline_mod, "index_page_images", lambda *a, **k: 0)

        pipeline_mod.ingest_pdf(
            str(pdf_path), doc_id, doc_name="large.pdf", user_id=user_id,
        )

        owned = documents_store.get_document(doc_id, user_id)
        assert owned is not None
        assert owned["status"] == "done"
        assert documents_store.get_document(doc_id, "web_user") is None
    finally:
        restore()


def test_documents_repairs_legacy_stuck_pdf_from_done_task():
    doc_id = "legacy-stuck-pdf"
    user_id = "user-legacy"
    documents_store.add_document(
        doc_id=doc_id, kind="pdf", user_id=user_id, doc_name="large.pdf",
        status="processing",
    )
    tasks_mod.create_task(
        "ingest_pdf", {"doc_id": doc_id, "kind": "pdf", "user_id": user_id},
        task_id=doc_id,
    )
    tasks_mod.update_task(doc_id, status="done", progress=100)

    docs = svc.documents(user_id)

    assert docs[0]["status"] == "done"
    assert documents_store.get_document(doc_id, user_id)["status"] == "done"


# ══════════════ 重新生成 / 断开回归 ══════════════
class _FakeGraph:
    checkpointer = None

    def __init__(self, updates):
        self._updates = updates

    def stream(self, state, config=None, stream_mode="updates"):
        for u in self._updates:
            yield u


def _answers_graph(answer: str):
    return _FakeGraph([
        {"load_memory": {"user_preferences": {}}},
        {"ask_user": {"ask_user": "continue"}},
        {"general_agent": {"final_answer": answer}},
        {"reflection": {"reflection_notes": "PASS"}},
    ])


def test_stream_persists_user_turn_before_model_finishes(monkeypatch):
    """生成中刷新时，会话和首条用户消息也必须已经可恢复。"""
    sid = "s-refresh-during-stream"
    monkeypatch.setattr(svc, "graph", _answers_graph("稍后完成"))
    stream = svc.stream_answer("刷新也不能丢", sid)

    first = next(stream)

    assert first["type"] == "delta"
    saved = svc.conversation(sid)
    assert saved == [{"role": "user", "content": "刷新也不能丢"}]
    stream.close()


def test_regenerate_replaces_old_qa(client, monkeypatch):
    monkeypatch.setattr(svc, "graph", _answers_graph("第一版回答"))
    with client.stream("POST", "/api/chat", json={
        "message": "旧问题", "session_id": "s-regen",
    }) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert "第一版回答" in text
    monkeypatch.setattr(svc, "graph", _answers_graph("第二版回答"))
    with client.stream("POST", "/api/chat", json={
        "message": "新问题", "session_id": "s-regen", "regenerate": True,
    }) as r:
        text = "".join(r.iter_text())
    assert "第二版回答" in text
    msgs = client.get("/api/sessions/s-regen").json()["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    assert len(users) == 1
    assert users[0]["content"] == "新问题"


def test_disconnect_stops_and_saves_partial(client, monkeypatch):
    def infinite_stream(self, state, config=None, stream_mode="updates"):
        while True:
            yield {"general_agent": {"final_answer": "部分回答"}}
            time.sleep(0.02)

    monkeypatch.setattr(svc, "graph", _FakeGraph([]))
    svc.graph.stream = infinite_stream
    with client.stream("POST", "/api/chat", json={
        "message": "停一下", "session_id": "s-stop",
    }) as r:
        assert r.status_code == 200
        for _ in r.iter_text():
            break
    saved = None
    for _ in range(200):
        data = client.get("/api/sessions/s-stop").json()
        msgs = data.get("messages") or []
        if any(m.get("role") == "assistant" for m in msgs):
            saved = msgs
            break
        time.sleep(0.05)
    assert saved, "停止生成后会话应保存部分内容"


# ══════════════ 注册 / 修改密码 ══════════════
def test_register_and_change_password_api(client):
    r = client.post(
        "/api/auth/register",
        json={"username": "newbie_art", "password": "password123", "name": "新用户"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["token"] and body["user"]["username"] == "newbie_art"
    token = body["token"]

    # 重复注册 → 400
    r = client.post(
        "/api/auth/register",
        json={"username": "newbie_art", "password": "password123"},
    )
    assert r.status_code == 400
    assert "已存在" in r.json()["error"]

    # 弱密码 → 400
    r = client.post(
        "/api/auth/register",
        json={"username": "weak_user", "password": "123"},
    )
    assert r.status_code == 400

    # 未登录改密 → 401
    r = client.post(
        "/api/auth/change-password",
        json={"old_password": "password123", "new_password": "newpass456"},
    )
    assert r.status_code == 401

    # 旧密码错误 → 400
    r = client.post(
        "/api/auth/change-password",
        json={"old_password": "wrong-old", "new_password": "newpass456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "当前密码不正确" in r.json()["error"]

    # 正确改密 → 当前会话保持有效
    r = client.post(
        "/api/auth/change-password",
        json={"old_password": "password123", "new_password": "newpass456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    # 新密码可登录，旧密码失效
    assert client.post(
        "/api/auth/login",
        json={"username": "newbie_art", "password": "newpass456"},
    ).status_code == 200
    assert client.post(
        "/api/auth/login",
        json={"username": "newbie_art", "password": "password123"},
    ).status_code == 401
