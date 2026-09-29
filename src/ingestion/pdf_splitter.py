"""超大 PDF 拆分：按页拆成若干不超过 max_bytes 的独立 PDF。"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def _open_stream(data):
    import fitz
    with tempfile.TemporaryDirectory(prefix="artagent-pdf-") as directory:
        path = Path(directory) / "input.pdf"
        data.seek(0)
        with path.open("wb") as target:
            shutil.copyfileobj(data, target, length=1024 * 1024)
        try:
            document = fitz.open(path)
        except Exception as exc:
            raise ValueError("无法读取 PDF 文件") from exc
        try:
            if document.needs_pass:
                raise ValueError("请先解密 PDF 后再上传")
            if not 0 < document.page_count <= max(1, int(os.getenv("UPLOAD_MAX_PDF_PAGES", "500"))):
                raise ValueError("PDF 页数超出允许范围")
            yield document, Path(directory)
        finally:
            document.close()
            data.seek(0)


def validate_pdf_stream(data):
    with _open_stream(data):
        pass


def save_split_upload(data, max_bytes, filename, user_id, reservation):
    """Prepare bounded parts on disk; reject oversized pages before persisting any."""
    import fitz
    from web.service import save_upload
    with _open_stream(data) as (source, directory):
        paths = []
        current = fitz.open()
        try:
            for index in range(source.page_count):
                current.insert_pdf(source, from_page=index, to_page=index)
                candidate = directory / "candidate.pdf"
                if candidate.exists():
                    candidate.unlink()
                current.save(candidate, garbage=3, deflate=True)
                if candidate.stat().st_size > max_bytes:
                    if current.page_count == 1:
                        raise ValueError("单页 PDF 超过拆分上限，请选择直接解析或压缩文件")
                    current.delete_page(current.page_count-1)
                    part = directory / f"part-{len(paths)+1}.pdf"
                    current.save(part, garbage=3, deflate=True)
                    paths.append(part)
                    current.close()
                    current = fitz.open()
                    current.insert_pdf(source, from_page=index, to_page=index)
                    candidate.unlink()
                    current.save(candidate, garbage=3, deflate=True)
                    if candidate.stat().st_size > max_bytes:
                        raise ValueError("单页 PDF 超过拆分上限，请选择直接解析或压缩文件")
                if len(paths) >= max(1, int(os.getenv("UPLOAD_MAX_SPLIT_PARTS", "20"))):
                    raise ValueError("拆分份数过多，请分批上传")
            if current.page_count:
                part = directory / f"part-{len(paths)+1}.pdf"
                current.save(part, garbage=3, deflate=True)
                paths.append(part)
        finally:
            current.close()
        from src.harness.admission import reserve_split_parts
        reserve_split_parts(reservation, user_id, len(paths), sum(path.stat().st_size for path in paths))
        saved = []
        for index, path in enumerate(paths):
            with path.open("rb") as stream:
                saved.append(save_upload(f"{Path(filename).stem}_part{index+1}.pdf", stream, user_id=user_id))
        return saved


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
