"""Persistent parsing and concurrent conversation mutation regression cases."""
import contextvars
import os
import subprocess
import sys
import time

import pytest

from src.data import documents_store as docs
from src.tasks import store
from src.harness.ingestion import run_ingestion, document_lock
from src.harness.context import RunStopped


@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    monkeypatch.setenv("INDEX_DIR",str(tmp_path))
    monkeypatch.setenv("MEMORY_USER_ID", "u")
    monkeypatch.setenv("UPLOADS_DIR",str(tmp_path/"uploads"))
    monkeypatch.setattr(docs,"DB_PATH",tmp_path/"documents.db")
    monkeypatch.setattr(docs,"_LEGACY_STATUS_FILE",tmp_path/"absent.json")
    store._reset_for_tests(tmp_path/"tasks.db")
    docs.init_db()


def task(doc_id="d",kind="pdf",file_path="missing.pdf"):
    docs.add_document(doc_id,kind,user_id="u",status="processing")
    return store.create_task("ingest_"+kind,{"doc_id":doc_id,"user_id":"u","kind":kind,"file_path":file_path},task_id=doc_id)


def test_new_worker_does_not_interrupt_live_document():
    tid = task()
    attempt = store.claim_task(tid,"u","first")
    docs.init_db()
    assert store.mark_interrupted_on_startup() == 0
    assert docs.get_document("d","u")["status"] == "processing"
    assert store.claim_task(tid,"u","second") is None
    assert store.get_task(tid)["attempt_id"] == attempt


def test_pending_upload_survives_restart_but_expired_parser_needs_retry():
    tid = task()
    assert store.mark_interrupted_on_startup() == 0
    assert store.runnable_ingestion_tasks()[0]["task_id"] == tid
    store.claim_task(tid,"u","dead",ttl=-1)
    assert store.runnable_ingestion_tasks() == []
    assert store.get_task(tid)["status"] == "interrupted"
    assert store.reset_task(tid)
    assert store.runnable_ingestion_tasks()[0]["task_id"] == tid


def test_parser_claims_share_global_capacity(monkeypatch):
    monkeypatch.setenv("TASK_PARSE_CONCURRENCY","1")
    first,second = task("one"),task("two")
    attempt = store.claim_task(first,"u","first")
    assert store.claim_task(second,"u","second") is None
    assert store.finish_attempt(first,attempt,"done")
    assert store.claim_task(second,"u","second")


def test_empty_parser_result_does_not_finish():
    tid = task()
    run_ingestion(tid,"u",handler=lambda:{"status":"done","pages":1,"text_chunks":0,"image_pages":0})
    assert store.get_task(tid)["status"] == "failed"
    assert "证据" in store.get_task(tid)["error"]


def test_parser_success_commits_under_lease():
    tid = task()
    def parse():
        docs.update_document("d",status="done",pages=1,text_chunks=1)
        return {"status":"done","pages":1,"text_chunks":1}
    run_ingestion(tid,"u",handler=parse)
    assert store.get_task(tid)["status"] == "done"
    assert store.get_task(tid)["progress"] == 100
    assert store.get_task(tid)["lease_owner"] == ""


def test_cancelled_parser_cannot_write_document():
    tid = task()
    def parse():
        store.cancel_ingestion("d","u")
        with pytest.raises(RunStopped):
            docs.update_document("d",status="done")
        return {"status":"done","pages":1,"text_chunks":1}
    run_ingestion(tid,"u",handler=parse)
    assert store.get_task(tid)["status"] == "interrupted"
    assert docs.get_document("d","u")["status"] != "done"


def test_real_worker_failure_is_recorded_without_network(tmp_path):
    from src.harness.process import run_process
    tid = task(file_path=str(tmp_path/"missing.pdf"))
    run_process(tid,"u")
    job = store.get_task(tid)
    assert job["status"] == "failed",job
    assert job["error"]
    assert job["lease_owner"] == ""


def test_document_lock_excludes_another_process():
    script = '''import time
from src.harness.ingestion import document_lock
deadline=time.monotonic()+.3
def check():
    if time.monotonic()>deadline: raise TimeoutError()
try:
    with document_lock("u","d",check): pass
except TimeoutError:
    raise SystemExit(0)
raise SystemExit(1)
'''
    with document_lock("u","d",lambda:None):
        result=subprocess.run([sys.executable,"-c",script],capture_output=True,timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        assert result.returncode==0,result.stderr.decode(errors="replace")


def test_attachment_removal_revokes_stale_conversation_snapshot(tmp_path,monkeypatch):
    from src.memory import conversations as conv
    monkeypatch.setattr(conv,"_DB_PATH",tmp_path/"conversations.db")
    monkeypatch.setattr(conv,"_LEGACY_DB_PATH",None)
    monkeypatch.setattr(conv,"_db_ready",False)
    history=[{"role":"user","content":"question"},{"role":"attachment","doc_id":"d"}]
    conv.save_conversation("s","title",history,"u")
    with conv.conversation_run("s","u"):
        assert contextvars.Context().run(conv.remove_attachment_from_all,"d","u")==1
        with pytest.raises(conv.ConversationBusy):
            conv.save_conversation("s","title",history,"u")
    assert len(conv.load_conversation("s","u"))==1


def test_rename_cannot_race_live_answer(tmp_path,monkeypatch):
    from src.memory import conversations as conv
    monkeypatch.setattr(conv,"_DB_PATH",tmp_path/"conversations.db")
    monkeypatch.setattr(conv,"_LEGACY_DB_PATH",None)
    monkeypatch.setattr(conv,"_db_ready",False)
    with conv.conversation_run("s","u"):
        with pytest.raises(conv.ConversationBusy):
            contextvars.Context().run(conv.rename_conversation,"s","new title","u")


def test_schema_confirmation_rechecks_document_after_waiting(monkeypatch):
    from contextlib import contextmanager
    from src.ingestion import table_pipeline
    from src.harness import ingestion
    docs.add_document("d", "table", user_id="u", status="pending_confirm")

    @contextmanager
    def deletion_wins(*args):
        docs.delete_document("d", "u")
        yield

    monkeypatch.setattr(ingestion, "document_lock", deletion_wins)
    monkeypatch.setattr(table_pipeline, "_confirm_table_schema_locked",
                        lambda *args: pytest.fail("deleted document must not be registered"))
    with pytest.raises(KeyError):
        table_pipeline.confirm_table_schema("d", {}, "u")


def test_schema_confirmation_rejects_unfinished_parse(monkeypatch):
    from src.ingestion import table_pipeline
    task(kind="table")
    docs.update_document("d", status="pending_confirm")
    monkeypatch.setattr(table_pipeline, "_confirm_table_schema_locked",
                        lambda *args: pytest.fail("unfinished parse must not be registered"))
    with pytest.raises(ValueError):
        table_pipeline.confirm_table_schema("d", {}, "u")


def test_pdf_version_switch_only_after_success():
    from src.ingestion.versions import physical_id, visible_indexes
    tid = task()
    docs.update_document("d", metadata={"active_index_id": "old"})
    observed = []
    def parse():
        observed.append(physical_id("d"))
        assert visible_indexes() == {"old": "d"}
        return {"status": "done", "pages": 1, "text_chunks": 1}
    run_ingestion(tid, "u", handler=parse)
    assert observed[0] != "old"
    assert visible_indexes() == {observed[0]: "d"}
    assert store.get_task(tid)["status"] == "done"


def test_failed_pdf_version_preserves_old_index():
    from src.ingestion.versions import physical_id, visible_indexes
    tid = task()
    docs.update_document("d", metadata={"active_index_id": "old"})
    staged = []
    def parse():
        staged.append(physical_id("d"))
        raise RuntimeError("partial index write")
    run_ingestion(tid, "u", handler=parse)
    assert visible_indexes() == {"old": "d"}
    assert staged[0] in docs.get_document("d", "u")["index_versions"]
    assert store.get_task(tid)["status"] == "failed"


def test_first_failed_index_is_never_visible():
    from src.ingestion.versions import visible_indexes
    tid = task()
    run_ingestion(tid, "u", handler=lambda: {"status": "done", "pages": 0})
    assert visible_indexes() == {}


def test_unpublished_done_parser_result_is_not_visible():
    from src.ingestion.versions import visible_indexes
    task()
    docs.update_document("d", status="done", metadata={"index_versions": ["staged"]})
    assert visible_indexes() == {}


def test_text_search_queries_only_published_namespace(monkeypatch):
    from src.retrieval import userdoc_text_retriever as retrieval
    task()
    docs.update_document("d", metadata={"active_index_id": "published", "index_versions": ["published", "partial"]})
    queries = []
    class Empty:
        def count(self):
            return 0
    class Collection:
        def count(self):
            return 2
        def query(self, **kwargs):
            queries.append(kwargs["where"])
            return {"metadatas": [[{"doc_id": "published"}, {"doc_id": "partial"}]],
                    "distances": [[0.1, 0.0]], "documents": [["valid", "partial write"]]}
    monkeypatch.setattr(retrieval, "get_or_create_chroma_collection",
                        lambda name: Collection() if name == retrieval.FALLBACK_COLLECTION_NAME else Empty())
    results = retrieval.UserDocTextRetriever().search("query", filters={"doc_id": "d"})
    assert queries == [{"doc_id": {"$in": ["published"]}}]
    assert [item.content for item in results] == ["valid"]
    assert results[0].metadata["doc_id"] == "d"


def test_legacy_completed_document_remains_searchable():
    from src.ingestion.versions import visible_indexes
    task()
    docs.update_document("d", status="done")
    assert visible_indexes() == {"d": "d"}


def test_strict_vector_cleanup_propagates_backend_failure(monkeypatch):
    from src.ingestion import pipeline
    class Collection:
        def count(self):
            return 1
        def get(self, **kwargs):
            return {"ids": ["old-chunk"]}
        def delete(self, **kwargs):
            raise RuntimeError("delete failed")
    monkeypatch.setattr(pipeline, "get_or_create_chroma_collection", lambda name: Collection())
    with pytest.raises(RuntimeError, match="delete failed"):
        pipeline.delete_pdf_vectors("d", strict=True)


def test_page_render_replaces_previous_attempt_image(tmp_path):
    import fitz
    from src.ingestion.multimodal_indexer import render_page_image
    source = tmp_path / "source.pdf"
    with fitz.open() as pdf:
        pdf.new_page(width=100, height=100)
        pdf.save(source)
    pages = tmp_path / "pages"
    pages.mkdir()
    image = pages / "page-0.png"
    image.write_bytes(b"stale previous attempt")
    assert render_page_image(str(source), 0, pages) == str(image)
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
