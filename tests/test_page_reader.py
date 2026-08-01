# tests/test_page_reader.py
"""
read_page_image（src/tools/page_reader.py）纯单测：
只测路径安全校验与错误分支（不调视觉模型、不联网），秒级完成。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.page_reader import _validate_image_path, read_page_image_impl

ROOT = Path(__file__).parent.parent.resolve()
UPLOADS = (ROOT / "data" / "uploads").resolve()


# ── 路径校验 ─────────────────────────────────────────────────────
def test_reject_empty_path():
    path, err = _validate_image_path("")
    assert path is None and "为空" in err


def test_reject_path_outside_uploads():
    path, err = _validate_image_path(str(ROOT / "SemArt" / "Images" / "x.jpg"))
    assert path is None and "允许范围" in err


def test_reject_path_traversal():
    path, err = _validate_image_path(str(UPLOADS / ".." / "SemArt" / "x.jpg"))
    assert path is None and "允许范围" in err


def _cleanup(*paths: Path) -> None:
    """容错清理测试临时文件（沙箱 safe-delete 可能拦截删除，残留无害）。"""
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except OSError:
            pass


def test_reject_non_image_suffix():
    p = UPLOADS / "default" / "fake" / "document.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-fake")
    try:
        path, err = _validate_image_path(str(p))
        assert path is None and "不支持的图片类型" in err
    finally:
        _cleanup(p, p.parent)


def test_reject_missing_file():
    path, err = _validate_image_path(str(UPLOADS / "default" / "nope" / "page-0.png"))
    assert path is None and "不存在" in err


def test_accept_valid_page_image(tmp_path=None):
    p = UPLOADS / "default" / "testdoc" / "pages" / "page-0.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    # 最小合法 PNG（1x1）
    p.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d4944415478da63fcffff3f030005fe02fea72d814d0000000049454e44ae426082"
        )
    )
    try:
        path, err = _validate_image_path(str(p))
        assert err is None and path == p
    finally:
        _cleanup(p, p.parent, p.parent.parent)


# ── impl 错误分支（不触发视觉调用） ──────────────────────────────
def test_impl_returns_error_for_bad_path():
    out = read_page_image_impl("")
    assert out["success"] is False and "error" in out


def test_impl_error_for_outside_path():
    out = read_page_image_impl(str(ROOT / "api.py"))
    assert out["success"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 page_reader 全部 {len(fns)} 个单测通过！")
