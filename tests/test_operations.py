"""Maintenance, recovery and cross-worker consistency without model calls."""
import json
import sqlite3
import time
from pathlib import Path

import pytest

from src.data import documents_store as docs


@pytest.fixture
def documents(tmp_path, monkeypatch):
    monkeypatch.setattr(docs, "DB_PATH", tmp_path / "documents.db")
    monkeypatch.setattr(docs, "_LEGACY_STATUS_FILE", tmp_path / "none.json")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER_ID", "u")
    docs.init_db()
    return tmp_path


def test_backup_restore_sqlite_and_binary(tmp_path):
    from src.ops.backup import create, restore, verify
    root = tmp_path / "data"
    root.mkdir()
    with sqlite3.connect(root / "tasks.db") as conn:
        conn.execute("CREATE TABLE tasks (value TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('durable')")
    (root / "image.bin").write_bytes(b"\x00\xffimage")
    snapshot = tmp_path / "snapshot"
    create({"data": root}, snapshot)
    assert len(verify(snapshot)["files"]) == 2
    restored = tmp_path / "restore"
    restore(snapshot, restored)
    with sqlite3.connect(restored / "data" / "tasks.db") as conn:
        assert conn.execute("SELECT value FROM tasks").fetchone()[0] == "durable"
    assert (restored / "data" / "image.bin").read_bytes() == b"\x00\xffimage"
    with pytest.raises(ValueError):
        restore(snapshot, restored)
    (snapshot / "data" / "image.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify(snapshot)


def test_restore_rejects_traversal_before_writing(tmp_path):
    from src.ops.backup import restore
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(json.dumps({"version": 1, "files": [
        {"path": "../outside", "bytes": 0, "sha256": "x"}]}))
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="unsafe"):
        restore(snapshot, destination)
    assert not destination.exists()


def test_version_gc_keeps_active_and_young_versions(documents, monkeypatch):
    from src.ingestion import maintenance, pipeline
    active, old, young = ["pdf-" + ch * 32 for ch in "abc"]
    docs.add_document("d", "pdf", user_id="u", status="done", metadata={
        "active_index_id": active, "index_versions": [active, old, young],
        "index_version_created": {active: 0, old: 0, young: 95}})
    monkeypatch.setattr(pipeline, "UPLOADS_DIR", documents / "uploads")
    removed = []
    monkeypatch.setattr(pipeline, "delete_pdf_vectors", lambda version, strict: removed.append(version))
    assert maintenance.collect_versions(now=100, grace_seconds=10) == 1
    assert removed == [old]
    assert docs.get_document("d", "u")["index_versions"] == [active, young]
    assert maintenance.collect_versions(now=100, grace_seconds=10) == 0


def test_version_gc_failure_is_retryable(documents, monkeypatch):
    from src.ingestion import maintenance, pipeline
    old = "pdf-" + "a" * 32
    docs.add_document("d", "pdf", user_id="u", status="failed", metadata={
        "index_versions": [old], "index_version_created": {old: 0}})
    def fail(*a, **k):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(pipeline, "delete_pdf_vectors", fail)
    with pytest.raises(RuntimeError):
        maintenance.collect_versions(now=100, grace_seconds=10)
    assert docs.get_document("d", "u")["index_versions"] == [old]


def test_table_workers_refresh_changed_schema_and_deletion(documents):
    from src.ingestion.table_pipeline import sync_active_tables
    from src.retrieval.hybrid import HybridRetriever
    docs.add_document("d", "table", user_id="u", status="active", metadata={
        "dataset_id": "table_u_d", "table_path": str(documents / "table.csv"),
        "confirmed_schema": {"entity_col": "name"}})
    first, second = HybridRetriever(), HybridRetriever()
    sync_active_tables(first)
    sync_active_tables(second)
    before = second.retrievers["table_u_d"]
    docs.update_document("d", metadata={"confirmed_schema": {"entity_col": "title"}})
    sync_active_tables(second)
    assert second.retrievers["table_u_d"] is not before
    docs.delete_document("d", "u")
    for worker in (first, second):
        worker.active_dataset = "table_u_d"
        sync_active_tables(worker)
        assert "table_u_d" not in worker.retrievers
        assert worker.active_dataset == "core"


def test_lexical_search_hides_staged_versions_before_ranking(documents, monkeypatch):
    from src.retrieval.lexical import PdfBm25Retriever
    docs.add_document("d", "pdf", user_id="u", status="done", metadata={"active_index_id": "old"})
    retriever = PdfBm25Retriever()
    monkeypatch.setattr(retriever, "_load_chunks", lambda: {"en": [
        ("Monet painting", {"doc_id": "old"}), ("Monet Monet Monet", {"doc_id": "staged"})]})
    monkeypatch.setattr("src.retrieval.lexical.translate_query", lambda q, lang: q)
    hits = retriever.search("Monet", top_k=1, filters={"doc_id": "d"})
    assert len(hits) == 1 and hits[0].content == "Monet painting"
    assert hits[0].metadata["doc_id"] == "d"


def test_stream_cancellation_kills_worker_and_releases_only_its_lease(tmp_path, monkeypatch):
    import sys
    import threading
    from src.harness.stream_process import stream
    from src.memory import conversations
    monkeypatch.setenv("ARTAGENT_MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(conversations, "_DB_PATH", tmp_path / "conversations.db")
    monkeypatch.setattr(conversations, "_LEGACY_DB_PATH", None)
    monkeypatch.setattr(conversations, "_db_ready", False)
    code = '''import json, sys, time
from src.memory.conversations import conversation_run
sys.stdin.readline()
with conversation_run("session", "u"):
    print(json.dumps({"type":"delta","html":"started"}), flush=True)
    time.sleep(60)
'''
    stopped = threading.Event()
    events = stream("chat", {"user_id": "u", "sid": "session"}, stopped,
                    command=[sys.executable, "-c", code], timeout=20)
    assert next(events)["type"] == "delta"
    stopped.set()
    assert list(events) == []
    with conversations.conversation_run("session", "u"):
        pass


def test_stream_deadline_stops_real_process(tmp_path, monkeypatch):
    import sys
    from src.harness.stream_process import stream
    from src.memory import conversations
    monkeypatch.setattr(conversations, "_DB_PATH", tmp_path / "conversations.db")
    monkeypatch.setattr(conversations, "_LEGACY_DB_PATH", None)
    monkeypatch.setattr(conversations, "_db_ready", False)
    code = "import sys,time; sys.stdin.readline(); time.sleep(60)"
    started = time.monotonic()
    events = list(stream("chat", {"user_id": "u", "sid": "s"}, timeout=.3,
                         command=[sys.executable, "-c", code]))
    assert events[-1]["code"] == "budget_exhausted"
    assert time.monotonic() - started < 10


def test_real_analysis_worker_reports_missing_image(documents, monkeypatch):
    from src.analysis import store
    from src.harness.stream_process import stream
    monkeypatch.setattr(store, "DB_PATH", documents / "user_images.db")
    store.init_db()
    events = list(stream("analysis", {"image_id": "missing"}, timeout=30))
    assert events[-1]["type"] == "error"
    assert "不存在" in events[-1]["message"]


def test_pdf_version_visibility_is_owner_scoped(documents):
    from src.ingestion.versions import visible_indexes
    docs.add_document("own", "pdf", user_id="u", status="done", metadata={"active_index_id": "own-v1"})
    docs.add_document("private", "pdf", user_id="other", status="done", metadata={"active_index_id": "private-v1"})
    assert visible_indexes() == {"own-v1": "own"}
    assert visible_indexes({"doc_id": "private"}) == {}


def test_many_claimants_have_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from src.tasks import store
    store._reset_for_tests(tmp_path / "tasks.db")
    tid = store.create_agent_job("concurrency fixture", "u", ["one"])
    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(lambda i: store.claim_job(tid, "u", str(i)), range(24)))
    assert sum(bool(attempt) for attempt in attempts) == 1
