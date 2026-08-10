"""
PDF 入库流水线编排。

流程：页级路由 → 文字路线解析分块入 BGE 库 / 多模态路线整页图入 DashScope 库
     → 写解析结果元数据（路由分布/chunk 数/耗时/状态）。

解析器选择：MinerU 精准解析 API 为主力（MINERU_TOKEN 配置时启用），
pdfplumber 兜底；公式密集页（force_mineru）在 MinerU 不可用/调用失败时
退到多模态整页图，不用 pdfplumber 硬解。

状态存储：统一走 src/data/documents_store（SQLite documents 表），
本模块不再维护独立的文档状态接口。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from src.data import documents_store
from src.ingestion.chunker import chunk_blocks
from src.ingestion.mineru_parser import mineru_available
from src.ingestion.mineru_parser import parse_pages as mineru_parse_pages
from src.ingestion.multimodal_indexer import index_page_images
from src.ingestion.page_classifier import DocRoutePlan, classify_document
from src.ingestion.pdfplumber_fallback import parse_pages as pdfplumber_parse_pages
from src.retrieval.hybrid import get_bge_m3_embed_batch, get_or_create_chroma_collection
from src.retrieval.userdoc_text_retriever import COLLECTION_NAME as TEXT_COLLECTION
from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("ingestion.pipeline")

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "./data/uploads"))


# ------------------------------------------------------------------ #
# 向量清理（删除文档时级联清理）                                       #
# ------------------------------------------------------------------ #


def delete_pdf_vectors(doc_id: str) -> dict:
    """删除该 doc_id 在 user_pdf_text / user_pdf_images 中的全部向量。"""
    from src.retrieval.userdoc_image_retriever import COLLECTION_NAME as IMAGE_COLLECTION

    deleted = {"text": 0, "images": 0}
    for name, col_name in (("text", TEXT_COLLECTION), ("images", IMAGE_COLLECTION)):
        collection = get_or_create_chroma_collection(col_name)
        if collection.count() == 0:
            continue
        try:
            hits = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
            if hits["ids"]:
                collection.delete(where={"doc_id": doc_id})
                deleted[name] = len(hits["ids"])
        except Exception as e:  # noqa: BLE001
            logger.warning("[pipeline] 清理 %s collection 失败 doc_id=%s: %s", col_name, doc_id, e)
    return deleted


# ------------------------------------------------------------------ #
# 解析器选择                                                            #
# ------------------------------------------------------------------ #


def _mineru_available() -> bool:
    """MinerU 精准解析 API：MINERU_TOKEN 配置即可用（调用失败走降级链）。"""
    return mineru_available()


def _parse_text_route(
    pdf_path: str, text_pages: list[int], plan: DocRoutePlan, work_dir: Path,
    force_pdfplumber: bool = False,
) -> tuple[list, set[int]]:
    """
    文字路线解析，返回 (blocks, 需转多模态整页图的页码集合)。

    MinerU 优先（MINERU_TOKEN 配置时）；不可用或调用失败时降级 pdfplumber，
    公式密集页（force_mineru）不接受 pdfplumber 硬解，转多模态整页图。
    """
    if not text_pages:
        return [], set()
    forced = {p.page_no for p in plan.pages if p.force_mineru} & set(text_pages)
    if not force_pdfplumber and _mineru_available():
        try:
            return mineru_parse_pages(pdf_path, text_pages, work_dir=work_dir), set()
        except Exception as e:  # noqa: BLE001 — 云解析失败不拖垮整个入库
            logger.warning("[ingest] MinerU 解析失败，降级 pdfplumber：%s", e)
    rest = [p for p in text_pages if p not in forced]
    blocks = pdfplumber_parse_pages(pdf_path, rest) if rest else []
    return blocks, forced


# ------------------------------------------------------------------ #
# 文字路线入库                                                          #
# ------------------------------------------------------------------ #


def _parse_ocr_route(
    pdf_path: str, mm_pages: list[int], work_dir, force_pdfplumber: bool = False
) -> list:
    """多模态/扫描页尝试 MinerU OCR：把扫描文字抽成可检索的 text blocks。

    这是"扫描件里的大段文字"进入文字层的正解（可语义检索，避免全靠视觉读图）。
    OCR 关闭 / MinerU 不可用 / 调用失败 / 无文字 → 返回空列表，保持纯图像路径。
    """
    if not mm_pages:
        return []
    if force_pdfplumber:
        return []  # 超大文件选择 pdfplumber 时，扫描页保持纯图像路线（不调 MinerU OCR）
    if os.getenv("MINERU_OCR", "1").strip().lower() in ("0", "false", "no"):
        return []
    if not mineru_available():
        return []
    try:
        blocks = mineru_parse_pages(pdf_path, mm_pages, work_dir=work_dir)
        text_blocks = [
            b for b in blocks
            if getattr(b, "type", "") in ("text", "table", "equation")
        ]
        log_event(
            logger, "ocr_route",
            pages=len(mm_pages), blocks=len(blocks), text_blocks=len(text_blocks),
        )
        return text_blocks
    except Exception as e:  # noqa: BLE001 — 失败降级纯图像路径
        logger.warning("[ingest] 扫描页 OCR 失败（保持纯图像路径）: %s", e)
        return []


def _context_header(doc_name: str, section: str) -> str:
    """上下文头：[文档 | 章节]——向量化与展示用，不写入原始 chunk。

    方案里的"实体"位无确定性来源（NER/LLM 抽取后续再做），
    先用文档名 + MinerU 标题层级（section）两档。
    """
    parts = [f"《{doc_name}》"] if doc_name else []
    if section:
        parts.append(section)
    return " | ".join(parts)


def index_text_chunks(chunks, doc_name: str = "") -> int:
    """BGE 批量编码 chunk 并写入 user_pdf_text collection。

    上下文头：向量化时拼接 [文档 | 章节] 头（只影响向量与展示，
    不改存储——documents 仍是原始 content，header 落 metadata 供展示复用；
    旧文档无 context_header 字段，展示端兼容缺省，重解析时覆盖）。
    """
    if not chunks:
        return 0
    collection = get_or_create_chroma_collection(TEXT_COLLECTION)
    embed_batch = get_bge_m3_embed_batch()
    headers = [_context_header(doc_name, c.section) for c in chunks]
    embed_inputs = [
        f"{h}\n{c.content}" if h else c.content for h, c in zip(headers, chunks)
    ]
    vectors = embed_batch(embed_inputs)
    collection.upsert(
        ids=[c.chroma_id() for c in chunks],
        embeddings=vectors,
        documents=[c.content for c in chunks],
        metadatas=[
            {**c.metadata(doc_name=doc_name), "context_header": h}
            for c, h in zip(chunks, headers)
        ],
    )
    logger.info("[text_index] doc_id=%s 文字 chunk 入库 %d 条", chunks[0].doc_id, len(chunks))
    return len(chunks)


# ------------------------------------------------------------------ #
# 主流水线                                                              #
# ------------------------------------------------------------------ #


def ingest_pdf(
    pdf_path: str,
    doc_id: str,
    doc_name: str = "",
    kb_id: str = "default",
    work_dir: Optional[Path] = None,
    force_pdfplumber: bool = False,
) -> dict:
    """
    PDF 入库主流程（同步执行；Web 层用 BackgroundTasks 包成后台任务）。

    返回入库摘要：页数/路由分布/文字 chunk 数/整页图数/耗时/状态。
    """
    t0 = time.time()
    doc_name = doc_name or Path(pdf_path).name
    work_dir = work_dir or (UPLOADS_DIR / kb_id / doc_id)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    file_path = str(work_dir / "document.pdf")
    file_size = Path(file_path).stat().st_size if Path(file_path).exists() else None

    documents_store.upsert_document(
        doc_id,
        kind="pdf",
        doc_name=doc_name,
        kb_id=kb_id,
        status="processing",
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        file_path=file_path,
        file_size=file_size,
    )

    try:
        # 1. 页级路由
        plan = classify_document(pdf_path)
        text_pages = [p.page_no for p in plan.pages if p.route in ("text", "dual")]
        mm_pages = [p.page_no for p in plan.pages if p.route in ("multimodal", "dual")]

        # 2. 文字路线：MinerU 优先、pdfplumber 兜底（公式密集页退多模态）
        blocks, extra_mm = _parse_text_route(
            pdf_path, text_pages, plan, work_dir, force_pdfplumber=force_pdfplumber
        )
        if extra_mm:
            mm_pages = sorted(set(mm_pages) | extra_mm)
            logger.info("[ingest] 公式密集页退多模态：%s", sorted(extra_mm))
        chunks = chunk_blocks(blocks, doc_id, kb_id=kb_id)
        n_chunks = index_text_chunks(chunks, doc_name=doc_name)

        # 3. 多模态路线：先 OCR 出文字层（扫描件可检索），再整页渲染入库。
        #    仅处理纯 multimodal 页；dual 页文字已被文字路线（MinerU 自带 OCR）抽取，
        #    再 OCR 会重复解析与重复 chunk。
        ocr_pages = [p.page_no for p in plan.pages if p.route == "multimodal"]
        ocr_blocks = _parse_ocr_route(
            pdf_path, ocr_pages, work_dir, force_pdfplumber=force_pdfplumber
        )
        if ocr_blocks:
            ocr_chunks = chunk_blocks(ocr_blocks, doc_id, kb_id=kb_id)
            n_chunks += index_text_chunks(ocr_chunks, doc_name=doc_name)
        n_images = index_page_images(
            pdf_path, doc_id, mm_pages,
            doc_name=doc_name, kb_id=kb_id, work_dir=work_dir,
        )

        summary = {
            "doc_name": doc_name,
            "pages": len(plan.pages),
            "text_chunks": n_chunks,
            "image_pages": n_images,
            "elapsed_sec": round(time.time() - t0, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "done",
            "error": "",
            "metadata": {"route_distribution": plan.distribution},
        }
        documents_store.upsert_document(doc_id, **summary)
        log_event(logger, "ingest_done", doc_id=doc_id, **summary)
        return {"doc_id": doc_id, **summary, "route_distribution": plan.distribution}

    except Exception as e:  # noqa: BLE001 — 失败也要落状态供前端轮询
        logger.exception("[ingest] 解析失败 doc_id=%s", doc_id)
        documents_store.upsert_document(
            doc_id, status="failed", error=str(e),
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        raise
