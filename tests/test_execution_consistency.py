"""Cross-boundary invariants for scoped execution; no model/network calls."""
import contextvars
import io
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.harness.context import RunContext, RunStopped, invoke_model, run_scope


@pytest.fixture
def conversations(tmp_path, monkeypatch):
    from src.memory import conversations as store
    monkeypatch.setattr(store, "_DB_PATH", tmp_path / "conversations.db")
    monkeypatch.setattr(store, "_DB_DIR", tmp_path)
    monkeypatch.setattr(store, "_LEGACY_DB_PATH", None)
    monkeypatch.setattr(store, "_db_ready", False)
    return store


def test_conversation_lease_blocks_other_process(conversations, tmp_path):
    store = conversations
    with store.conversation_run("s", "u"):
        store.save_conversation("s", "title", [{"role":"user", "content":"first"}], "u")
        script = '''import os,sys
os.environ["ARTAGENT_MEMORY_DIR"]=sys.argv[1]
from src.memory.conversations import conversation_run, ConversationBusy
try:
    with conversation_run("s", "u"): pass
except ConversationBusy:
    sys.exit(0)
sys.exit(1)
'''
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    with store.conversation_run("s", "u"):
        assert store.load_conversation("s", "u")[0]["content"] == "first"


def test_expired_conversation_writer_cannot_overwrite(conversations):
    store = conversations
    with store.conversation_run("s", "u"):
        with store._lock:
            conn = store._get_conn()
            conn.execute("UPDATE conversation_leases SET expires=0")
            conn.commit()
        def replacement():
            with store.conversation_run("s", "u"):
                store.save_conversation("s", "fresh", [{"content":"fresh"}], "u")
        contextvars.Context().run(replacement)
        with pytest.raises(store.ConversationBusy):
            store.save_conversation("s", "stale", [{"content":"stale"}], "u")
    assert store.load_conversation("s", "u") == [{"content":"fresh"}]


def test_unleased_edit_cannot_overwrite_live_conversation(conversations):
    with conversations.conversation_run("s", "u"):
        with pytest.raises(conversations.ConversationBusy):
            contextvars.Context().run(conversations.save_conversation, "s", "edit", [{"content":"edit"}], "u")


def test_selected_documents_filter_before_rerank():
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.base import RetrievalResult
    calls = []
    class Source:
        source = "user_pdf_text"
        def search(self, query, top_k=5, filters=None):
            calls.append(filters)
            return [RetrievalResult(source=self.source, content=doc, metadata={"doc_id":doc}) for doc in ("allowed", "other")]
    retriever = HybridRetriever()
    retriever.register("pdf", Source())
    retriever.register("core", SimpleNamespace(source="core", search=lambda *a, **k: pytest.fail("Unselected source queried")))
    with run_scope(RunContext("r", "u", document_ids={"allowed"})):
        results = retriever.search("query", rerank=False)
    assert [r.metadata["doc_id"] for r in results] == ["allowed"]
    assert calls == [{"doc_id":"allowed"}]


def test_model_reserves_output_and_counts_tool_schema(monkeypatch):
    calls = []
    class Model:
        kwargs = {"tools":[{"description":"schema" * 100}]}
        def bind(self, **kwargs):
            calls.append(kwargs)
            return self
        def invoke(self, messages):
            return AIMessage(content="ok")
    run = RunContext("r", "u", max_tokens=1000)
    with run_scope(run):
        invoke_model(Model(), [HumanMessage(content="q")])
    assert 0 < calls[0]["max_tokens"] < 1000
    assert 300 < run.usage["tokens"] <= 1000


def test_budget_exhaustion_stays_terminal_when_caught():
    run = RunContext("r", "u", max_model_calls=0)
    with run_scope(run):
        with pytest.raises(RunStopped):
            invoke_model(SimpleNamespace(invoke=lambda _: pytest.fail("Model started")), "q")
        with pytest.raises(RunStopped):
            run.check()


def test_slash_activation_routes_through_skill_tool():
    from src.agent.nodes.general import general_agent
    from src.agent.state import AgentState
    result = general_agent(AgentState(original_user_query="/artwork_deep_analysis test"))
    assert result["messages"][0].tool_calls[0]["name"] == "skill_artwork_deep_analysis"
    assert result["messages"][0].tool_calls[0]["args"] == {"task":"test"}


def test_pdf_page_limit_is_checked_before_persistence(monkeypatch):
    import fitz
    from src.ingestion.pdf_splitter import validate_pdf_stream
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        stream = io.BytesIO(doc.tobytes())
    monkeypatch.setenv("UPLOAD_MAX_PDF_PAGES", "1")
    with pytest.raises(ValueError, match="页数"):
        validate_pdf_stream(stream)
    assert stream.tell() == 0


def test_process_supervisor_stops_worker_before_reporting_cancelled(tmp_path, monkeypatch):
    from src.harness import process as supervisor
    from src.tasks import store
    store._reset_for_tests(tmp_path / "tasks.db")
    tid = store.create_agent_job("test", "u", ["step"])
    real_popen = subprocess.Popen
    children = []
    def launch(*args, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    timer = threading.Timer(.4, lambda: store.cancel_agent_job(tid))
    timer.start()
    try:
        supervisor.run_process(tid, "u")
    finally:
        timer.join()
        for child in children:
            supervisor.terminate_worker(child)
    assert children and children[0].poll() is not None
    assert store.get_task(tid)["status"] == "interrupted"
    assert store.get_task(tid)["lease_owner"] == ""


def test_legacy_password_upgrades_and_reset_revokes_sessions(tmp_path):
    import hashlib
    from src.platform import users
    users._reset_for_tests(tmp_path / "users.db")
    created = users.create_user_with_password("Test", "tester", "password123")
    uid = created["user"]["user_id"]
    legacy = "salt:" + hashlib.sha256(b"salt:password123").hexdigest()
    conn = users._get_conn()
    conn.execute("UPDATE users SET password_hash=? WHERE user_id=?", (legacy, uid))
    conn.commit()
    assert users.verify_login("tester", "wrong") is None
    assert users.verify_login("tester", "password123")
    assert users.get_user(uid)["password_hash"].startswith("pbkdf2_sha256$")
    token = users.issue_session_token(uid)
    assert users.get_user_by_api_key(token)
    assert users.reset_password(uid, "replacement123")
    assert users.get_user_by_api_key(token) is None
    assert users.get_user_by_api_key(created["api_key"])


def test_tokens_are_digested_and_sessions_expire(tmp_path):
    from src.platform import users
    users._reset_for_tests(tmp_path / "users.db")
    uid = users.create_user_with_password("Test", "tester", "password123")["user"]["user_id"]
    token = users.issue_session_token(uid)
    conn = users._get_conn()
    stored = conn.execute("SELECT key FROM api_keys WHERE label='session'").fetchone()[0]
    assert stored != token and stored.startswith("sha256:")
    assert users.get_user_by_api_key(stored) is None
    conn.execute("UPDATE api_keys SET created_at='2000-01-01T00:00:00+00:00' WHERE label='session'")
    conn.commit()
    assert users.get_user_by_api_key(token) is None


def test_upload_admission_is_user_scoped_and_releases_on_error(tmp_path, monkeypatch):
    from src.harness.admission import upload_slot, UploadQuotaExceeded
    from src.data import documents_store
    from src.tasks import store
    monkeypatch.setenv("ARTAGENT_HARNESS_DB_PATH", str(tmp_path / "harness.db"))
    monkeypatch.setenv("UPLOAD_USER_CONCURRENCY", "1")
    monkeypatch.setattr(documents_store, "list_documents", lambda user_id: [])
    store._reset_for_tests(tmp_path / "tasks.db")
    with upload_slot("u"):
        with pytest.raises(UploadQuotaExceeded):
            with upload_slot("u"):
                pass
        with upload_slot("other"):
            pass
    with pytest.raises(RuntimeError):
        with upload_slot("u"):
            raise RuntimeError("failed upload")
    with upload_slot("u"):
        pass


def test_pdf_split_reserves_all_parts_before_streaming_them(monkeypatch):
    import fitz
    from src.ingestion.pdf_splitter import save_split_upload
    from src.harness import admission
    from web import service
    events = []
    with fitz.open() as source:
        for number in range(3):
            source.new_page().insert_text((30,30), f"Page {number}")
        sizes = []
        for number in range(3):
            with fitz.open() as one:
                one.insert_pdf(source, from_page=number, to_page=number)
                sizes.append(len(one.tobytes(garbage=3, deflate=True)))
        stream = io.BytesIO(source.tobytes())
    limit = max(sizes) + 30
    monkeypatch.setattr(admission, "reserve_split_parts", lambda key,user,parts,size: events.append(("reserve", parts)))
    def save(name, data, **kwargs):
        blob = data.read()
        assert len(blob) <= limit
        events.append(("save", name))
        return {"doc_id":name}
    monkeypatch.setattr(service, "save_upload", save)
    results = save_split_upload(stream, limit, "book.pdf", "u", "slot")
    assert len(results) == 3
    assert events[0] == ("reserve", 3)
    assert len(events) == 4
