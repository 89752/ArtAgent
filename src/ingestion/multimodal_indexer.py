"""
多模态路线入库（Stage 3）：整页渲染 → DashScope 多模态向量化 → 独立 collection。

每页只产出一个向量（非 ColPali 类每页上千 patch 向量），存储/检索成本
与 SemArt 索引同量级。整页图渲染后落盘保存——检索命中时前端可展示、
后续 Stage 可喂 Qwen-VL 读图作答。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval.hybrid import get_or_create_chroma_collection
from src.retrieval.userdoc_image_retriever import COLLECTION_NAME, get_mm_embed_fn
from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("ingestion.multimodal")

RENDER_DPI = 150  # 整页渲染分辨率（兼顾清晰度与 <5MB 的 API 限制）


def render_page_image(pdf_path: str, page_no: int, out_dir: Path) -> str:
    """把 PDF 某一页整体渲染成图片，返回落盘路径。"""
    import fitz  # PyMuPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page-{page_no}.png"
    if out_path.exists():
        return str(out_path)
    with fitz.open(pdf_path) as doc:
        page = doc[page_no]
        zoom = RENDER_DPI / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(str(out_path))
    return str(out_path)


def embed_image_file(image_path: str) -> list[float]:
    """DashScope 多模态编码一张本地图片（base64 内联）。"""
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return get_mm_embed_fn()({"image": f"data:image/png;base64,{b64}"})


def index_page_images(
    pdf_path: str,
    doc_id: str,
    page_nos: list[int],
    *,
    doc_name: str = "",
    kb_id: str = "default",
    work_dir: Path,
) -> int:
    """整页渲染 + 多模态向量化 + 入库，返回入库页数。"""
    if not page_nos:
        return 0
    collection = get_or_create_chroma_collection(COLLECTION_NAME)
    pages_dir = work_dir / "pages"

    ids, embeddings, metadatas = [], [], []
    for page_no in page_nos:
        image_path = render_page_image(pdf_path, page_no, pages_dir)
        vector = embed_image_file(image_path)
        page_id = f"{doc_id}-p{page_no}"
        ids.append(f"{page_id}-img")
        embeddings.append(vector)
        metadatas.append(
            {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_id": page_id,
                "page": page_no + 1,
                "block_type": "page_image",
                "kb_id": kb_id,
                "image_path": image_path,
            }
        )
        log_event(logger, "mm_index", doc_id=doc_id, page=page_no + 1)

    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    logger.info("[mm_index] doc_id=%s 整页图入库 %d 页", doc_id, len(ids))
    return len(ids)
