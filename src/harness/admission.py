"""Atomic per-user upload admission shared by API workers."""
from contextlib import contextmanager
import os
import time
import uuid

from src.harness.storage import connection


class UploadQuotaExceeded(RuntimeError):
    pass


@contextmanager
def upload_slot(user_id):
    from src.data.documents_store import list_documents
    from src.tasks import store
    key = uuid.uuid4().hex
    hard_bytes = max(1, int(os.getenv("UPLOAD_HARD_MAX_MB", "200"))) * 1024 * 1024
    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE IF NOT EXISTS upload_slots (id TEXT PRIMARY KEY,user_id TEXT,bytes INTEGER,expires REAL)")
        if "documents" not in {row[1] for row in conn.execute("PRAGMA table_info(upload_slots)")}:
            conn.execute("ALTER TABLE upload_slots ADD COLUMN documents INTEGER NOT NULL DEFAULT 1")
        conn.execute("DELETE FROM upload_slots WHERE expires<?", (time.time(),))
        count, reserved, document_slots = conn.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0),COALESCE(SUM(documents),0) FROM upload_slots WHERE user_id=?", (user_id,)).fetchone()
        if count >= max(1, int(os.getenv("UPLOAD_USER_CONCURRENCY", "2"))):
            raise UploadQuotaExceeded("同时上传的文件过多，请稍后重试")
        documents = list_documents(user_id)
        if len(documents) + document_slots >= max(1, int(os.getenv("UPLOAD_USER_MAX_DOCUMENTS", "200"))):
            raise UploadQuotaExceeded("资料数量已达到账号上限，请先清理不需要的资料")
        used = sum(int(doc.get("file_size") or 0) for doc in documents)
        if used + reserved + hard_bytes > max(1, int(os.getenv("UPLOAD_USER_STORAGE_MB", "2048"))) * 1024 * 1024:
            raise UploadQuotaExceeded("账号剩余空间不足以预留一次上传")
        with store._lock:
            pending = store._get_conn().execute("SELECT COUNT(*) FROM tasks WHERE type IN ('ingest_pdf','ingest_table') AND status IN ('pending','processing') AND json_extract(payload,'$.user_id')=?", (user_id,)).fetchone()[0]
        if pending + document_slots >= max(1, int(os.getenv("UPLOAD_USER_PENDING_LIMIT", "10"))):
            raise UploadQuotaExceeded("待解析资料过多，请等待已有任务完成")
        conn.execute("INSERT INTO upload_slots(id,user_id,bytes,expires) VALUES (?,?,?,?)", (key,user_id,hard_bytes,time.time()+900))
    try:
        yield key
    finally:
        with connection() as conn:
            conn.execute("DELETE FROM upload_slots WHERE id=?", (key,))


def reserve_split_parts(key, user_id, parts, size_bytes):
    """Reserve every output before creating documents, avoiding split quota bypass."""
    from src.data.documents_store import list_documents
    from src.tasks import store
    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        slot = conn.execute("SELECT * FROM upload_slots WHERE id=? AND user_id=? AND expires>?", (key,user_id,time.time())).fetchone()
        if slot is None:
            raise UploadQuotaExceeded("上传预留已过期，请重试")
        other_docs, other_bytes = conn.execute("SELECT COALESCE(SUM(documents),0),COALESCE(SUM(bytes),0) FROM upload_slots WHERE user_id=? AND id!=?", (user_id,key)).fetchone()
        docs = list_documents(user_id)
        with store._lock:
            pending = store._get_conn().execute("SELECT COUNT(*) FROM tasks WHERE type IN ('ingest_pdf','ingest_table') AND status IN ('pending','processing') AND json_extract(payload,'$.user_id')=?", (user_id,)).fetchone()[0]
        if len(docs)+other_docs+parts > max(1,int(os.getenv("UPLOAD_USER_MAX_DOCUMENTS","200"))) or pending+other_docs+parts > max(1,int(os.getenv("UPLOAD_USER_PENDING_LIMIT","10"))):
            raise UploadQuotaExceeded("拆分后的资料数量超过账号或待解析任务上限")
        reserved = max(size_bytes, slot["bytes"])
        if sum(int(d.get("file_size") or 0) for d in docs)+other_bytes+reserved > max(1,int(os.getenv("UPLOAD_USER_STORAGE_MB","2048")))*1024*1024:
            raise UploadQuotaExceeded("拆分后的资料超过账号存储上限")
        conn.execute("UPDATE upload_slots SET documents=?,bytes=? WHERE id=?", (parts,reserved,key))
