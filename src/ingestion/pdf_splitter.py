"""超大 PDF 拆分：按页拆成若干不超过 max_bytes 的独立 PDF。"""

from __future__ import annotations

from pathlib import Path


def split_pdf(
    data: bytes,
    max_bytes: int,
    filename: str = "document.pdf",
) -> list[tuple[str, bytes]]:
    """按页拆分 PDF，返回 [(part_name, part_bytes), ...]。

    规则：逐页累计，加入下一页会超过 max_bytes 时先封存当前部分；
    单页本身超过上限时仍作为独立部分返回（调用方自行提示质量取舍）。
    """
    import fitz  # PyMuPDF

    src = fitz.open(stream=data, filetype="pdf")
    stem = Path(filename).stem or "document"
    parts: list[tuple[str, bytes]] = []
    current = fitz.open()
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current.page_count:
            parts.append((f"{stem}_part{len(parts) + 1}.pdf", current.tobytes()))
            current = fitz.open()
            current_size = 0

    try:
        for i in range(src.page_count):
            page_doc = fitz.open()
            page_doc.insert_pdf(src, from_page=i, to_page=i)
            page_bytes = page_doc.tobytes()
            page_doc.close()
            if current.page_count and current_size + len(page_bytes) > max_bytes:
                flush()
            current.insert_pdf(src, from_page=i, to_page=i)
            current_size += len(page_bytes)
        flush()
    finally:
        src.close()
    return parts
