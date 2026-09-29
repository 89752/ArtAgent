"""Bounded, retryable cleanup; never remove the published PDF generation."""
import os
import re
import shutil
import time

from src.data import documents_store
from src.harness.ingestion import DocumentBusy, document_lock


def collect_versions(*, now=None, grace_seconds=None, limit=20):
    from src.ingestion.pipeline import UPLOADS_DIR, delete_pdf_vectors
    now = time.time() if now is None else now
    grace = max(3600, int(os.getenv("INDEX_VERSION_GRACE_SECONDS", "86400"))) if grace_seconds is None else grace_seconds
    removed = 0
    for item in documents_store.list_all_documents():
        if removed >= limit:
            break
        if item.get("kind") != "pdf":
            continue
        deadline = time.monotonic() + .1
        def check():
            if time.monotonic() >= deadline:
                raise DocumentBusy("document busy")
        try:
            with document_lock(item["user_id"], item["doc_id"], check):
                doc = documents_store.get_document(item["doc_id"], item["user_id"])
                if not doc:
                    continue
                versions = list(doc.get("index_versions") or [])
                created = dict(doc.get("index_version_created") or {})
                changed = False
                for version in list(versions):
                    if version == doc.get("active_index_id") or removed >= limit:
                        continue
                    if version not in created:
                        created[version] = now
                        changed = True
                        continue
                    if now - float(created[version]) < grace:
                        continue
                    if not re.fullmatch(r"pdf-[0-9a-f]{32}", version):
                        continue
                    root = UPLOADS_DIR.resolve()
                    directory = (root / doc["kb_id"] / doc["doc_id"] / "versions" / version).resolve()
                    if not directory.is_relative_to(root) or directory == root:
                        raise ValueError("version path outside uploads")
                    delete_pdf_vectors(version, strict=True)
                    if directory.exists():
                        shutil.rmtree(directory)
                    versions.remove(version)
                    created.pop(version, None)
                    changed = True
                    removed += 1
                if changed:
                    documents_store.update_document(doc["doc_id"], metadata={
                        "index_versions": versions, "index_version_created": created})
        except DocumentBusy:
            continue
    return removed


def mark_orphaned_documents(*, age_seconds=3600):
    from src.tasks import store
    count = 0
    for doc in documents_store.list_all_documents():
        if doc.get("status") != "processing" or store.get_task(doc["doc_id"]):
            continue
        try:
            started = time.mktime(time.strptime(doc.get("started_at") or "", "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if time.time() - started < age_seconds:
            continue
        # No destructive cleanup: keep original materials for explicit recovery.
        documents_store.update_document(doc["doc_id"], status="failed", error="历史解析缺少持久任务记录，请重新上传或由管理员恢复")
        count += 1
    return count
