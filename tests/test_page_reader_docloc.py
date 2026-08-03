"""read_page_image 按文档名+页码定位整页图（不依赖语义检索命中）单测。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.tools.page_reader as pr


def _fake_docs(doc_name="莫奈手稿", doc_id="abc123", kb_id="default"):
    return [{"doc_name": doc_name, "doc_id": doc_id, "kb_id": kb_id, "pages": 16}]


def _setup_uploads():
    tmp = Path(tempfile.mkdtemp())
    pages = tmp / "default" / "abc123" / "pages"
    pages.mkdir(parents=True)
    (pages / "page-0.png").write_bytes(b"fake-png")
    (pages / "page-1.png").write_bytes(b"fake-png")
    return tmp


def test_resolve_page_path_by_doc_name_and_page():
    tmp = _setup_uploads()
    with patch.object(pr, "_UPLOADS_ROOT", tmp.resolve()), \
         patch("src.data.documents_store.list_documents", return_value=_fake_docs()):
        path, err = pr._resolve_page_path("莫奈手稿", 1)
    assert err is None
    assert path is not None and path.name == "page-0.png" and path.exists()


def test_resolve_page_missing_file_returns_error():
    tmp = _setup_uploads()
    with patch.object(pr, "_UPLOADS_ROOT", tmp.resolve()), \
         patch("src.data.documents_store.list_documents", return_value=_fake_docs()):
        path, err = pr._resolve_page_path("莫奈手稿", 99)
    assert err is not None and "不存在" in err


def test_resolve_unknown_doc_returns_error():
    tmp = _setup_uploads()
    with patch.object(pr, "_UPLOADS_ROOT", tmp.resolve()), \
         patch("src.data.documents_store.list_documents", return_value=_fake_docs()):
        path, err = pr._resolve_page_path("不存在的文档", 1)
    assert err is not None and "未找到" in err


def test_read_page_image_impl_requires_locator():
    out = pr.read_page_image_impl()
    assert out["success"] is False and "doc_name 与 page" in out["error"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] page_reader_docloc 全部 {len(fns)} 个单测通过")
