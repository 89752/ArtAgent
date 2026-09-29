"""Leased document parsing with single-host OS locks across worker crashes."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
import time
import uuid

from src.harness.context import RunContext, RunStopped, run_scope
from src.tasks import store


class DocumentBusy(RuntimeError):
    pass


@contextmanager
def document_lock(user_id, doc_id, check):
    directory = Path(os.getenv("INDEX_DIR", "./data/index")) / "ingestion-locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{user_id}:{doc_id}".encode()).hexdigest()
    with (directory / (key + ".lock")).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        while True:
            check()
            try:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                time.sleep(.1)
        try:
            check()
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def run_ingestion(task_id, user_id, *, attempt="", handler=None):
    job = store.get_task(task_id)
    if not job or job["type"] not in {"ingest_pdf", "ingest_table"} or job["payload"].get("user_id") != user_id:
        return
    if attempt:
        if job["attempt_id"] != attempt or job["lease_until"] <= time.time():
            return
    else:
        attempt = store.claim_task(task_id,user_id,f"ingestion:{os.getpid()}:{uuid.uuid4().hex}",task_types=("ingest_pdf","ingest_table"))
    if not attempt:
        return
    from src.data import documents_store
    from src.memory.memory_items import set_active_user_id, clear_active_user_id
    started = time.monotonic()
    previous = float((job.get("usage") or {}).get("elapsed_seconds",0))
    stopped = threading.Event()
    lost = threading.Event()

    def cancelled():
        fresh = store.get_task(task_id)
        return lost.is_set() or not fresh or fresh["attempt_id"] != attempt or fresh["lease_until"] <= time.time() or fresh["cancel_requested"]

    def persist(usage):
        usage["elapsed_seconds"] = previous + time.monotonic()-started
        if not store.heartbeat(task_id,attempt,usage):
            lost.set()
            raise RunStopped("interrupted")

    context = RunContext(attempt,user_id,task_id=task_id,attempt_id=attempt,task_type="ingestion",
        deadline=started+max(0,int(os.getenv("TASK_PARSE_TIMEOUT_SEC","1800"))-previous),
        max_model_calls=10,max_tokens=40000,cancelled=cancelled,persist=persist,usage=job.get("usage") or {})
    def pulse():
        while not stopped.wait(15):
            try:
                with context._lock:
                    persist(dict(context.usage))
            except Exception:
                lost.set()
                return
    thread = threading.Thread(target=pulse,daemon=True,name="ingestion-lease")
    thread.start()
    status, error = "failed", ""
    payload = job["payload"]
    doc_id = payload.get("doc_id") or task_id
    try:
        documents_store.init_db()
        set_active_user_id(user_id)
        with run_scope(context), document_lock(user_id,doc_id,context.check):
            from src.ingestion.versions import index_version
            version = "pdf-" + uuid.uuid4().hex if job["type"] == "ingest_pdf" else ""
            if version:
                old = documents_store.get_document(doc_id, user_id) or {}
                versions = list(old.get("index_versions") or [])
                versions.append(version)
                created = dict(old.get("index_version_created") or {})
                # Legacy attempts without timestamps start their grace period now.
                for vid in versions:
                    created.setdefault(vid, time.time())
                metadata = {"index_versions": versions, "index_version_created": created}
                if not old.get("active_index_id") and old.get("status") == "done":
                    metadata["active_index_id"] = doc_id
                documents_store.update_document(doc_id, metadata=metadata)
            if handler is None:
                if job["type"] == "ingest_pdf":
                    from src.ingestion.pipeline import ingest_pdf
                    if not Path(payload["file_path"]).is_file():
                        raise FileNotFoundError(payload["file_path"])
                    handler = lambda: ingest_pdf(payload["file_path"],doc_id,doc_name=payload.get("doc_name",""),
                        kb_id=payload.get("kb_id","default"),force_pdfplumber=bool(payload.get("force_pdfplumber")),user_id=user_id)
                else:
                    from src.ingestion.table_pipeline import ingest_table
                    handler = lambda: ingest_table(payload["file_path"],doc_id,doc_name=payload.get("doc_name",""),
                        kb_id=payload.get("kb_id","default"),user_id=user_id)
            with index_version(version):
                result = handler()
            context.check()
            if not isinstance(result,dict) or result.get("status") not in {"done","pending_confirm"}:
                raise ValueError("解析器未返回有效完成状态")
            if job["type"] == "ingest_pdf" and (int(result.get("pages") or 0) < 1 or
                    int(result.get("text_chunks") or 0)+int(result.get("image_pages") or 0) < 1):
                raise ValueError("PDF 没有产生可检索文字或页面证据")
            if job["type"] == "ingest_table" and (int(result.get("rows") or 0) < 1 or int(result.get("cols") or 0) < 1):
                raise ValueError("表格没有可用数据")
            if version:
                documents_store.update_document(doc_id, metadata={"active_index_id": version})
            status = "done"
    except RunStopped as exc:
        status, error = ("interrupted" if exc.status == "cancelled" else exc.status), "解析执行已停止"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        stopped.set()
        thread.join(timeout=1)
        clear_active_user_id()
        usage = {**context.usage,"elapsed_seconds":previous+time.monotonic()-started}
        store.heartbeat(task_id,attempt,usage)
        store.finish_attempt(task_id,attempt,status,error)
