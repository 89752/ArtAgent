# tests/test_stage6_lifecycle.py
"""
Stage 6 文档生命周期级联删除纯单测：
PDF 删除清向量/文件/SQLite；表格删除注销数据源并复位当前数据源。
不加载 SemArt、不调真实 LLM、不联网。
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data import documents_store
from src.ingestion import pipeline as pipeline_mod
from src.ingestion import table_pipeline as tp
from src.retrieval import hybrid as hybrid_mod
from src.retrieval.hybrid import HybridRetriever
from web import service as service_mod


def _isolate(tmp: Path):
    """把 SQLite 状态库与 uploads 指到临时目录，并返回恢复函数。"""
    old_db = documents_store.DB_PATH
    old_uploads = pipeline_mod.UPLOADS_DIR
    old_legacy = documents_store._LEGACY_STATUS_FILE
    db_path = tmp / "documents.db"
    documents_store._reset_for_tests(db_path)
    documents_store._LEGACY_STATUS_FILE = Path("/nonexistent/doc_status.json")
    documents_store.init_db()
    pipeline_mod.UPLOADS_DIR = tmp / "uploads"
    tp.UPLOADS_DIR = pipeline_mod.UPLOADS_DIR
    return lambda: (
        setattr(documents_store, "DB_PATH", old_db),
        setattr(documents_store, "_LEGACY_STATUS_FILE", old_legacy),
        setattr(pipeline_mod, "UPLOADS_DIR", old_uploads),
        setattr(tp, "UPLOADS_DIR", old_uploads),
    )


def _fresh_hybrid():
    """重置 Hybrid 单例，保留默认 core 数据源，避免测试间状态污染。"""
    hybrid_mod._hybrid = None
    return hybrid_mod.get_hybrid_retriever()


def test_delete_pdf_cascades():
    """PDF 删除应级联清理 Chroma 向量、上传文件与 SQLite 记录。"""
    tmp = Path(tempfile.mkdtemp(prefix="s6_pdf_"))
    restore = _isolate(tmp)
    try:
        hybrid = _fresh_hybrid()
        doc_id = "pdf-abc"
        work_dir = pipeline_mod.UPLOADS_DIR / "default" / doc_id
        work_dir.mkdir(parents=True)
        pdf_path = work_dir / "document.pdf"
        pdf_path.write_text("fake pdf")

        documents_store.add_document(
            doc_id=doc_id, kind="pdf", doc_name="x.pdf",
            status="done", file_path=str(pdf_path),
            file_size=7, pages=1, text_chunks=2, image_pages=0,
        )

        # 模拟 delete_pdf_vectors 的返回值，避免真查 Chroma
        with patch.object(pipeline_mod, "delete_pdf_vectors", return_value={"text": 2, "images": 0}) as mock_del:
            result = service_mod.delete_document(doc_id)

        assert result["doc_id"] == doc_id
        assert result["kind"] == "pdf"
        assert result["vectors"] == {"text": 2, "images": 0}
        assert result["files_removed"] is True
        assert result["db_deleted"] is True
        assert documents_store.get_document(doc_id) is None
        assert not work_dir.exists()
        mock_del.assert_called_once_with(doc_id)
    finally:
        restore()


def test_delete_table_unregisters_and_resets_active():
    """表格删除应注销数据源；若其为当前生效数据源，则复位为 core。"""
    tmp = Path(tempfile.mkdtemp(prefix="s6_tbl_"))
    restore = _isolate(tmp)
    try:
        hybrid = _fresh_hybrid()
        doc_id = "tbl-xyz"
        dataset_id = tp.table_dataset_id(doc_id)

        # 构造一个已激活表格数据源
        documents_store.add_document(
            doc_id=doc_id, kind="table", doc_name="books.csv",
            status="active",
            metadata={
                "dataset_id": dataset_id,
                "table_path": str(tmp / "books.csv"),
                "confirmed_schema": {
                    "entity_col": "书名", "group_axis_col": None,
                    "description_col": "", "image_col": None,
                },
                "display_name": "书单",
                "supports_timeline": False,
                "supports_recommendation": False,
            },
        )
        tp.register_structured_dataset(
            dataset_id,
            tp.TableSchema(entity_col="书名", description_col=""),
            source="user_table",
            df=pd.DataFrame(columns=["书名"]),
        )
        hybrid.register(dataset_id, hybrid.retrievers["core"])  # 占位 retriever
        hybrid.active_dataset = dataset_id

        result = service_mod.delete_document(doc_id)

        assert result["doc_id"] == doc_id
        assert result["kind"] == "table"
        assert result["active_dataset_reset"] == "core"
        assert hybrid.active_dataset == "core"
        assert dataset_id not in hybrid.retrievers
        assert documents_store.get_document(doc_id) is None
    finally:
        restore()


def test_delete_nonexistent_raises():
    """删除不存在的文档应抛出 KeyError。"""
    tmp = Path(tempfile.mkdtemp(prefix="s6_miss_"))
    restore = _isolate(tmp)
    try:
        documents_store._reset_for_tests(tmp / "documents.db")
        documents_store._LEGACY_STATUS_FILE = Path("/nonexistent/doc_status.json")
        documents_store.init_db()
        try:
            service_mod.delete_document("no-such-doc")
            raise AssertionError("应抛出 KeyError")
        except KeyError as e:
            assert "no-such-doc" in str(e)
    finally:
        restore()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] Stage 6 生命周期级联删除 {len(fns)} 个单测通过！")
