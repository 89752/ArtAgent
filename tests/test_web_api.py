"""Web/API 层集成测试（FastAPI TestClient）。

运行前通过环境变量把 SQLite 数据隔离到临时目录，避免污染真实数据：
  - INDEX_DIR            → documents.db
  - ARTAGENT_MEMORY_DIR  → conversations.db / 偏好库
"""

import os
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="artagent_web_test_")
os.environ["INDEX_DIR"] = _TMP
os.environ["ARTAGENT_MEMORY_DIR"] = _TMP

import pytest
from fastapi.testclient import TestClient

import api
from src.data import documents_store


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_chat_empty_message_streams_done(client):
    with client.stream(
        "POST", "/api/chat",
        json={"message": "", "session_id": "s-empty"},
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

    # 未知文档 → 404
    r = client.post("/api/sessions/s1/attachment", json={"doc_id": "nope"})
    assert r.status_code == 404
    assert r.json()["ok"] is False

    # 记录附件 → 会话出现
    r = client.post("/api/sessions/s1/attachment", json={"doc_id": "web-test-doc"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.get("/api/sessions/s1")
    msgs = r.json()["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "attachment"
    assert msgs[0]["doc_id"] == "web-test-doc"

    # 列表分页形状 + 重命名
    r = client.get("/api/sessions?offset=0&limit=10")
    data = r.json()
    assert "items" in data and "total" in data and "has_more" in data
    assert any(s["session_id"] == "s1" for s in data["items"])

    r = client.patch("/api/sessions/s1", json={"title": "重命名后的会话"})
    assert r.status_code == 200
    r = client.get("/api/sessions?offset=0&limit=10")
    s1 = next(s for s in r.json()["items"] if s["session_id"] == "s1")
    assert s1["title"] == "重命名后的会话"

    # 删除文档 → 会话附件记录级联清理（仅剩附件记录的会话被移除）
    r = client.delete("/api/documents/web-test-doc")
    assert r.status_code == 200
    r = client.get("/api/sessions/s1")
    assert r.json()["messages"] == []

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


def test_dataset_unknown_rejected(client):
    r = client.post("/api/dataset/active", json={"dataset_id": "does-not-exist"})
    assert r.status_code == 404
    assert r.json()["ok"] is False


def test_image_route_serves_and_blocks_traversal(client):
    # 路径穿越 → 404
    r = client.get("/api/images/..%2F..%2Fapi.py")
    assert r.status_code == 404
    # 真实 SemArt 图片 → 200；CI 等无数据环境 → 404（SemArt/ 被 gitignore）
    r = client.get("/api/images/28496-early05.jpg")
    if Path("SemArt/Images/28496-early05.jpg").exists():
        assert r.status_code == 200
        assert r.headers.get("cache-control", "").startswith("public")
    else:
        assert r.status_code == 404
