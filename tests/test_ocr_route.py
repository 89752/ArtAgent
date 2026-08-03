"""扫描页 OCR 路由单测：开关、降级、文字块过滤（mock MinerU，不联网）。"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.blocks import Block
from src.ingestion.pipeline import _parse_ocr_route


def _blocks():
    return [
        Block(type="text", content="手稿正文第一段", page_idx=0),
        Block(type="table", content="<table>…</table>", page_idx=0),
        Block(type="equation", content="x^2", page_idx=1),
        Block(type="image", content="page-1.png", page_idx=1),
    ]


def test_ocr_disabled_returns_empty():
    os.environ["MINERU_OCR"] = "0"
    try:
        out = _parse_ocr_route("x.pdf", [0, 1], None)
    finally:
        os.environ.pop("MINERU_OCR", None)
    assert out == []


def test_ocr_unavailable_returns_empty():
    with patch("src.ingestion.pipeline.mineru_available", return_value=False):
        out = _parse_ocr_route("x.pdf", [0], None)
    assert out == []


def test_ocr_parse_failure_falls_back():
    with patch("src.ingestion.pipeline.mineru_available", return_value=True), \
         patch("src.ingestion.pipeline.mineru_parse_pages", side_effect=RuntimeError("boom")):
        out = _parse_ocr_route("x.pdf", [0], None)
    assert out == []


def test_ocr_keeps_only_text_blocks():
    with patch("src.ingestion.pipeline.mineru_available", return_value=True), \
         patch("src.ingestion.pipeline.mineru_parse_pages", return_value=_blocks()):
        out = _parse_ocr_route("x.pdf", [0, 1], None)
    assert len(out) == 3  # text + table + equation（image 块不进文字层）
    assert all(b.type != "image" for b in out)


def test_ocr_empty_pages_returns_empty():
    with patch("src.ingestion.pipeline.mineru_available", return_value=True) as m:
        out = _parse_ocr_route("x.pdf", [], None)
    m.assert_not_called()
    assert out == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] ocr_route 全部 {len(fns)} 个单测通过")
