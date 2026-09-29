"""Immutable physical index namespaces and atomic SQLite publication."""
from contextlib import contextmanager
from contextvars import ContextVar

_version = ContextVar("pdf_index_version", default="")


@contextmanager
def index_version(version):
    token = _version.set(version)
    try:
        yield
    finally:
        _version.reset(token)


def physical_id(doc_id):
    return _version.get() or doc_id


def visible_indexes(filters=None):
    from src.data import documents_store
    from src.harness.context import current_run
    from src.memory.memory_items import get_memory_user_id, DEFAULT_MEMORY_USER
    documents_store.init_db()
    filters = filters or {}
    run = current_run()
    user_id = run.user_id if run else get_memory_user_id()
    if user_id == DEFAULT_MEMORY_USER:
        user_id = "web_user"
    mapping = {}
    for doc in documents_store.list_documents(user_id):
        if doc.get("kind") != "pdf":
            continue
        if any(filters.get(key) and filters[key] != doc.get(key) for key in ("doc_id", "kb_id")):
            continue
        version = doc.get("active_index_id")
        if version or (not doc.get("index_versions") and doc.get("status") == "done"):
            mapping[version or doc["doc_id"]] = doc["doc_id"]
    return mapping
