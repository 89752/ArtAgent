# tests/test_documents_store.py
"""
documents_store 纯单测：SQLite CRUD、旧 JSON 迁移、metadata 合并、
状态字典形状兼容。不加载数据集、不调 LLM、不联网。
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import documents_store


def _reset():
    """每个测试用独立临时数据库。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    documents_store._reset_for_tests(path)
    # 阻止旧 JSON 迁移干扰测试
    old_legacy = documents_store._LEGACY_STATUS_FILE
    documents_store._LEGACY_STATUS_FILE = Path("/nonexistent/doc_status.json")
    try:
        documents_store.init_db()
    finally:
        documents_store._LEGACY_STATUS_FILE = old_legacy


def test_init_creates_table():
    _reset()
    assert documents_store.DB_PATH.exists()


def test_add_and_get_document():
    _reset()
    documents_store.add_document(
        doc_id="pdf-1", kind="pdf", doc_name="test.pdf",
        status="processing", file_path="uploads/default/pdf-1/document.pdf",
        file_size=1024,
    )
    doc = documents_store.get_document("pdf-1")
    assert doc["doc_id"] == "pdf-1"
    assert doc["kind"] == "pdf"
    assert doc["doc_name"] == "test.pdf"
    assert doc["file_size"] == 1024
    assert doc["status"] == "processing"


def test_upsert_document_updates_status_on_existing():
    _reset()
    documents_store.add_document(doc_id="pdf-1", kind="pdf", status="processing")
    documents_store.upsert_document("pdf-1", status="done", text_chunks=3)
    doc = documents_store.get_document("pdf-1")
    assert doc["status"] == "done"
    assert doc["text_chunks"] == 3


def test_update_document():
    _reset()
    documents_store.add_document(doc_id="pdf-2", kind="pdf")
    documents_store.update_document(
        "pdf-2", status="done", pages=10, text_chunks=5,
        metadata={"route_distribution": {"text": 8, "multimodal": 2}},
    )
    doc = documents_store.get_document("pdf-2")
    assert doc["status"] == "done"
    assert doc["pages"] == 10
    assert doc["text_chunks"] == 5
    assert doc["route_distribution"] == {"text": 8, "multimodal": 2}


def test_metadata_merge():
    """update_document 的 metadata 应与现有 metadata 合并，而非覆盖。"""
    _reset()
    documents_store.add_document(
        doc_id="tab-1", kind="table",
        metadata={"rows": 12, "dataset_id": "table_tab-1"},
    )
    documents_store.update_document(
        "tab-1", status="active",
        metadata={"confirmed_schema": {"entity_col": "name"}},
    )
    doc = documents_store.get_document("tab-1")
    assert doc["rows"] == 12
    assert doc["dataset_id"] == "table_tab-1"
    assert doc["confirmed_schema"]["entity_col"] == "name"


def test_list_documents_order():
    _reset()
    documents_store.add_document(doc_id="b", kind="pdf", started_at="2026-08-01 10:00:00")
    documents_store.add_document(doc_id="a", kind="pdf", started_at="2026-08-01 12:00:00")
    docs = documents_store.list_documents()
    assert [d["doc_id"] for d in docs] == ["a", "b"]


def test_delete_document():
    _reset()
    documents_store.add_document(doc_id="del", kind="pdf")
    assert documents_store.delete_document("del")
    assert documents_store.get_document("del") is None


def test_migrate_legacy_json():
    """从旧 doc_status.json 格式迁移到 SQLite。"""
    _reset()
    legacy = {
        "pdf-doc": {
            "doc_name": "legacy.pdf", "kb_id": "default", "status": "done",
            "started_at": "2026-08-01 10:00:00", "pages": 5,
            "route_distribution": {"text": 4, "dual": 1},
            "text_chunks": 10, "image_pages": 1, "elapsed_sec": 12.3,
        },
        "table-doc": {
            "doc_name": "legacy.csv", "kb_id": "default", "kind": "table",
            "status": "active", "started_at": "2026-08-01 11:00:00",
            "table_path": "uploads/default/table-doc/table.csv",
            "dataset_id": "table_table-doc", "rows": 20, "cols": 4,
            "columns": ["a", "b", "c", "d"],
            "confirmed_schema": {"entity_col": "a"},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        legacy_path = Path(tmp) / "doc_status.json"
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

        # 临时把迁移目标指到临时目录
        old_legacy = documents_store._LEGACY_STATUS_FILE
        documents_store._LEGACY_STATUS_FILE = legacy_path
        try:
            documents_store._migrate_legacy_json()
        finally:
            documents_store._LEGACY_STATUS_FILE = old_legacy

    pdf_doc = documents_store.get_document("pdf-doc")
    assert pdf_doc["kind"] == "pdf"
    assert pdf_doc["pages"] == 5
    assert pdf_doc["route_distribution"]["text"] == 4

    table_doc = documents_store.get_document("table-doc")
    assert table_doc["kind"] == "table"
    assert table_doc["rows"] == 20
    assert table_doc["dataset_id"] == "table_table-doc"
    assert table_doc["confirmed_schema"]["entity_col"] == "a"


def test_status_dict_shape():
    """_to_status_dict 应与旧 list_doc_status 返回的扁平形状兼容。"""
    _reset()
    documents_store.add_document(
        doc_id="shape", kind="table",
        metadata={
            "dataset_id": "table_shape", "rows": 7,
            "supports_timeline": True, "supports_recommendation": False,
        },
    )
    doc = documents_store.get_document("shape")
    assert doc["dataset_id"] == "table_shape"
    assert doc["rows"] == 7
    assert doc["supports_timeline"] is True
    assert doc["supports_recommendation"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] documents_store 全部 {len(fns)} 个单测通过！")
