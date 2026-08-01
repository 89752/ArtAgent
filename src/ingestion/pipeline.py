"""
PDF 入库流水线编排（Stage 3）。

流程：页级路由 → 文字路线解析分块入 BGE 库 / 多模态路线整页图入 DashScope 库
     → 写解析结果元数据（路由分布/chunk 数/耗时/状态）。

解析器选择：MinerU 精准解析 API 为主力（MINERU_TOKEN 配置时启用），
pdfplumber 兜底；公式密集页（force_mineru）在 MinerU 不可用/调用失败时
退到多模态整页图，不用 pdfplumber 硬解。

状态存储：Phase 1 用 JSON 文件（data/index/doc_status.json）支撑上传进度
轮询；Stage 6 换 SQLite documents_store 时整体替换本模块的状态部分。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from src.ingestion.chunker import chunk_blocks
from src.ingestion.mineru_parser import mineru_available
from src.ingestion.mineru_parser import parse_pages as mineru_parse_pages
from src.ingestion.multimodal_indexer import index_page_images
from src.ingestion.page_classifier import DocRoutePlan, classify_document
from src.ingestion.pdfplumber_fallback import parse_pages as pdfplumber_parse_pages
from src.retrieval.hybrid import _get_bge_model, get_or_create_chroma_collection
from src.retrieval.userdoc_text_retriever import COLLECTION_NAME as TEXT_COLLECTION
from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("ingestion.pipeline")

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "./data/uploads"))
_STATUS_FILE = Path(os.getenv("INDEX_DIR", "./data/index")) / "doc_status.json"


# ------------------------------------------------------------------ #
# 状态存储（Stage 6 换 SQLite 前的轻量实现）                            #
# ------------------------------------------------------------------ #


def _load_status() -> dict:
    if not _STATUS_FILE.exists():
        return {}
    try:
        return json.loads(_STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_status(data: dict) -> None:
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_doc_status(doc_id: str, **fields) -> None:
    data = _load_status()
    data.setdefault(doc_id, {}).update(fields)
    _save_status(data)


def get_doc_status(doc_id: str) -> Optional[dict]:
    return _load_status().get(doc_id)


def list_doc_status() -> list[dict]:
    return [
        {"doc_id": doc_id, **info} for doc_id, info in _load_status().items()
    ]


# ------------------------------------------------------------------ #
# 解析器选择                                                            #
# ------------------------------------------------------------------ #


def _mineru_available() -> bool:
    """MinerU 精准解析 API：MINERU_TOKEN 配置即可用（调用失败走降级链）。"""
    return mineru_available()


def _parse_text_route(
    pdf_path: str, text_pages: list[int], plan: DocRoutePlan, work_dir: Path
) -> tuple[list, set[int]]:
    """
    文字路线解析，返回 (blocks, 需转多模态整页图的页码集合)。

    MinerU 优先（MINERU_TOKEN 配置时）；不可用或调用失败时降级 pdfplumber，
    公式密集页（force_mineru）不接受 pdfplumber 硬解，转多模态整页图。
    """
    if not text_pages:
        return [], set()
    forced = {p.page_no for p in plan.pages if p.force_mineru} & set(text_pages)
    if _mineru_available():
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


def _context_header(doc_name: str, section: str) -> str:
    """上下文头（Stage 4）：[文档 | 章节]——向量化与展示用，不写入原始 chunk。

    方案里的"实体"位无确定性来源（NER/LLM 抽取留给 Phase 2），
    Phase 1 用文档名 + MinerU 标题层级（section）两档。
    """
    parts = [f"《{doc_name}》"] if doc_name else []
    if section:
        parts.append(section)
    return " | ".join(parts)


def index_text_chunks(chunks, doc_name: str = "") -> int:
    """BGE 批量编码 chunk 并写入 user_pdf_text collection。

    Stage 4 上下文头：向量化时拼接 [文档 | 章节] 头（只影响向量与展示，
    不改存储——documents 仍是原始 content，header 落 metadata 供展示复用；
    旧文档无 context_header 字段，展示端兼容缺省，Stage 6 重解析时覆盖）。
    """
    if not chunks:
        return 0
    collection = get_or_create_chroma_collection(TEXT_COLLECTION)
    model = _get_bge_model()
    headers = [_context_header(doc_name, c.section) for c in chunks]
    embed_inputs = [
        f"{h}\n{c.content}" if h else c.content for h, c in zip(headers, chunks)
    ]
    vectors = model.encode(embed_inputs, normalize_embeddings=True).tolist()
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

    update_doc_status(
        doc_id,
        doc_name=doc_name,
        kb_id=kb_id,
        status="processing",
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    try:
        # 1. 页级路由
        plan = classify_document(pdf_path)
        text_pages = [p.page_no for p in plan.pages if p.route in ("text", "dual")]
        mm_pages = [p.page_no for p in plan.pages if p.route in ("multimodal", "dual")]

        # 2. 文字路线：MinerU 优先、pdfplumber 兜底（公式密集页退多模态）
        blocks, extra_mm = _parse_text_route(pdf_path, text_pages, plan, work_dir)
        if extra_mm:
            mm_pages = sorted(set(mm_pages) | extra_mm)
            logger.info("[ingest] 公式密集页退多模态：%s", sorted(extra_mm))
        chunks = chunk_blocks(blocks, doc_id, kb_id=kb_id)
        n_chunks = index_text_chunks(chunks, doc_name=doc_name)

        # 3. 多模态路线：整页渲染 → DashScope 入库
        n_images = index_page_images(
            pdf_path, doc_id, mm_pages,
            doc_name=doc_name, kb_id=kb_id, work_dir=work_dir,
        )

        summary = {
            "doc_name": doc_name,
            "pages": len(plan.pages),
            "route_distribution": plan.distribution,
            "text_chunks": n_chunks,
            "image_pages": n_images,
            "elapsed_sec": round(time.time() - t0, 1),
            "status": "done",
        }
        update_doc_status(doc_id, **summary)
        log_event(logger, "ingest_done", doc_id=doc_id, **summary)
        return {"doc_id": doc_id, **summary}

    except Exception as e:  # noqa: BLE001 — 失败也要落状态供前端轮询
        logger.exception("[ingest] 解析失败 doc_id=%s", doc_id)
        update_doc_status(doc_id, status="failed", error=str(e))
        raise
