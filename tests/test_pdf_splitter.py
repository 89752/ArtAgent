"""超大 PDF 拆分单测：按页拆包、单页超限兜底、命名与页数正确。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz

from src.ingestion.pdf_splitter import split_pdf


def _make_pdf(pages: int) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1} content")
    data = doc.tobytes()
    doc.close()
    return data


def test_split_pdf_creates_multiple_parts():
    data = _make_pdf(5)
    parts = split_pdf(data, max_bytes=1200, filename="big.pdf")
    assert len(parts) >= 2
    assert all(name.startswith("big_part") and name.endswith(".pdf") for name, _ in parts)
    total_pages = 0
    for _, part_bytes in parts:
        doc = fitz.open(stream=part_bytes, filetype="pdf")
        total_pages += doc.page_count
        doc.close()
    assert total_pages == 5


def test_split_pdf_small_file_single_part():
    data = _make_pdf(2)
    parts = split_pdf(data, max_bytes=10 * 1024 * 1024, filename="small.pdf")
    assert len(parts) == 1
    assert parts[0][0] == "small_part1.pdf"


def test_split_pdf_single_oversized_page_still_returned():
    data = _make_pdf(1)
    parts = split_pdf(data, max_bytes=100, filename="one.pdf")
    assert len(parts) == 1  # 单页超限：仍作为独立部分返回，不丢页
