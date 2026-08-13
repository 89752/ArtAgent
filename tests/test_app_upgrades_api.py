"""成熟应用升级 API 集成测试（一期体验闭环 + 二期可观测/任务化）。

通过 TestClient 覆盖：反馈、偏好面板、任务重试、指标、限流、request_id、
重新生成（fake graph）与断开停止回归。
"""

import os
import tempfile
import time
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="artagent_upgrade_api_")
os.environ["RATE_LIMIT_RPM"] = "0"  # 默认关限流，避免测试互相干扰

# 隔离方式：直接设置各存储模块的路径属性（不用 env）。
# pytest 在跑任何测试前会先导入全部测试模块，模块级改 env 会在 collection
# 阶段污染全局、且 fixture teardown 无法按模块恢复（2026-08-04 定位）。
from src.memory import conversations as _conv_module  # noqa: E402
from src.memory import summary as _summary_module  # noqa: E402
from src.memory import feedback as _feedback_module  # noqa: E402
from src.memory import memory_items as _mi_module  # noqa: E402
from src.data import documents_store as _docs_module  # noqa: E402

_conv_module._DB_PATH = Path(_TMP) / "conversations.db"
_conv_module._conn = None
_summary_module._DB_PATH = Path(_TMP) / "conversations.db"
_summary_module._conn = None
_feedback_module._DB_PATH = Path(_TMP) / "feedback.db"
_feedback_module._conn = None
_mi_module._DB_PATH = Path(_TMP) / "agent_memory.db"
_mi_module._conn = None
_docs_module.DB_PATH = Path(_TMP) / "documents.db"
_docs_module._LEGACY_STATUS_FILE = Path(_TMP) / "doc_status.json"

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from unittest.mock import patch

import api
from web import service
from src.memory import feedback
from src.tasks import store as tasks
from src.observability import runs

# 偏好接口现在写 memory_items：统一 patch 掉 embedding，避免加载 BGE 模型
patch("src.memory.memory_items._embed", return_value=None).start()


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_state():
    api._rate_buckets.clear()
    yield
    api._rate_buckets.clear()


# ── 反馈闭环 ──
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
    assert r.status_code == 422  # rating 仅允许 1/-1
    data = client.get("/api/feedback").json()
    assert data["total"] == 2


def test_memory_api_v2(client):
    """记忆面板 v2：全 kind 展示（含自动抽取来源）、按 id 删除、清空。"""
    from src.memory.memory_items import add_memory

    add_memory("web_user", "用户喜欢莫奈睡莲", kind="preference",
               entity="莫奈", source="user_explicit")
    add_memory("web_user", "用户住在上海", kind="fact",
               entity="上海", source="extracted")
    _summary_module._save_summary("conv-x", "web_user", "上次聊了莫奈", 2)

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
    assert _summary_module.load_summary("conv-x") == ""  # 滚动摘要一并清空


# ── 任务化与重试 ──
def test_tasks_api_and_retry(client, monkeypatch):
    tid = tasks.create_task("ingest_pdf", {"doc_id": "d1", "kind": "pdf"})
    tasks.update_task(tid, status="failed", error="解析失败")
    r = client.get(f"/api/tasks/{tid}")
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "failed"

    calls = []

    def fake_ingest(doc_id, doc_name, pdf_path, kb_id, task_id=None, force_pdfplumber=False):
        calls.append((doc_id, task_id))

    monkeypatch.setattr(service, "ingest_document", fake_ingest)
    r = client.post(f"/api/tasks/{tid}/retry")
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "pending"
    assert calls == [("d1", tid)]  # 后台任务已重新派发

    r = client.post(f"/api/tasks/{tid}/retry")
    assert r.status_code == 400  # pending 不可重复重试
    r = client.post("/api/tasks/nope/retry")
    assert r.status_code == 404


# ── 指标 ──
def test_metrics_api(client):
    runs.record_run(
        session_id="s-m", intent="general", tools=["web_search"],
        context_chars=1000, tool_rounds=1, latency_ms=300.0,
    )
    m = client.get("/api/metrics").json()
    assert m["count"] >= 1
    assert m["latency_ms"]["avg"] >= 300.0
    assert m["tool_calls"]["web_search"] >= 1


# ── 限流 ──
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


# ── 重新生成回归（fake graph） ──
class _FakeGraph:
    checkpointer = None

    def __init__(self, updates):
        self._updates = updates

    def stream(self, state, config=None, stream_mode="updates"):
        for u in self._updates:
            yield u


def _answers_graph(answer: str):
    return _FakeGraph([
        {"rewrite_split": {"user_query": "q"}},
        {"classify": {"intent": "general"}},
        {"general_agent": {"final_answer": answer}},
        {"reflection": {"reflection_notes": "PASS"}},
    ])


def test_regenerate_replaces_old_qa(client, monkeypatch):
    monkeypatch.setattr(service, "graph", _answers_graph("第一版回答"))
    with client.stream("POST", "/api/chat", json={
        "message": "旧问题", "session_id": "s-regen",
    }) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert "第一版回答" in text

    monkeypatch.setattr(service, "graph", _answers_graph("第二版回答"))
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

    monkeypatch.setattr(service, "graph", _FakeGraph([]))
    service.graph.stream = infinite_stream
    with client.stream("POST", "/api/chat", json={
        "message": "停一下", "session_id": "s-stop",
    }) as r:
        assert r.status_code == 200
        for _ in r.iter_text():
            break  # 读一条就断开
    # 生产者线程收尾：部分内容落库（轮询等待）
    saved = None
    for _ in range(200):
        data = client.get("/api/sessions/s-stop").json()
        msgs = data.get("messages") or []
        if any(m.get("role") == "assistant" for m in msgs):
            saved = msgs
            break
        time.sleep(0.05)
    assert saved, "停止生成后会话应保存部分内容"
