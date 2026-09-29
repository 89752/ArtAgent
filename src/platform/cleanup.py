"""Retryable account erasure. Identity is retained, disabled, until cleanup succeeds."""
import os
import shutil
import sqlite3
import time
from pathlib import Path


class CleanupPending(RuntimeError):
    pass


def purge_resources(user_id):
    from src.tasks import store as tasks
    from src.memory import conversations
    from src.harness.storage import connection
    from src.analysis import store as images
    from src.analysis.engine import USER_IMAGE_ROOT
    from src.data import documents_store
    from web.service import delete_document
    # Stop new work and ask supervisors to stop existing child processes.
    with tasks._lock:
        conn = tasks._get_conn()
        conn.execute("UPDATE tasks SET cancel_requested=1 WHERE json_extract(payload,'$.user_id')=?", (user_id,))
        conn.commit()
        active = conn.execute("SELECT COUNT(*) FROM tasks WHERE json_extract(payload,'$.user_id')=? AND lease_until>?", (user_id, time.time())).fetchone()[0]
    with conversations._lock:
        conn = conversations._get_conn()
        active += conn.execute("SELECT COUNT(*) FROM conversation_leases WHERE user_id=? AND expires>?", (user_id,time.time())).fetchone()[0]
    with connection() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "upload_slots" in tables:
            active += conn.execute("SELECT COUNT(*) FROM upload_slots WHERE user_id=? AND expires>?", (user_id,time.time())).fetchone()[0]
    images.init_db()
    with images._connect() as conn:
        owned_images = [dict(row) for row in conn.execute("SELECT * FROM user_images WHERE user_id=?", (user_id,))]
    active += sum(image.get("status") in {"processing", "analyzing"} for image in owned_images)
    if active:
        raise CleanupPending("账号已禁用，正在等待执行停止；请稍后重试删除")

    documents = documents_store.list_documents(user_id)
    for doc in documents:
        delete_document(doc["doc_id"], user_id)
    for image in owned_images:
        root = USER_IMAGE_ROOT.resolve()
        directory = (root / image["image_id"]).resolve()
        if directory == root or not directory.is_relative_to(root):
            raise ValueError("image path outside uploads")
        if directory.exists():
            shutil.rmtree(directory)
        images.delete_image(image["image_id"], user_id)

    path = Path(os.getenv("ARTAGENT_CHECKPOINT_DB_PATH", str(Path(__file__).resolve().parents[2]/"data"/"memory"/"checkpoints.db")))
    if path.exists():
        with sqlite3.connect(path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            prefix = user_id + ":"
            for table in ("writes", "checkpoints"):
                if table in tables:
                    conn.execute(f"DELETE FROM {table} WHERE substr(thread_id,1,?)=?", (len(prefix),prefix))
    from src.observability import runs
    from src.memory import collections, metrics
    for module, table in ((collections, "collections"), (metrics, "extraction_metrics")):
        with module._lock:
            conn = module._get_conn()
            conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
            conn.commit()
    with runs._lock:
        conn = runs._get_conn()
        for table in ("node_events", "model_calls", "tool_calls"):
            conn.execute(f"DELETE FROM {table} WHERE run_id IN (SELECT id FROM agent_runs WHERE user_id=?)", (user_id,))
        conn.execute("DELETE FROM agent_runs WHERE user_id=?", (user_id,))
        conn.commit()
    with connection() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("artifacts", "tool_receipts", "experiences", "learning_events", "upload_slots"):
            if table in tables:
                conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
    with tasks._lock:
        conn = tasks._get_conn()
        conn.execute("DELETE FROM tasks WHERE json_extract(payload,'$.user_id')=?", (user_id,))
        conn.commit()
    return {"documents": len(documents), "images": len(owned_images)}
