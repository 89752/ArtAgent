"""
统一混合检索入口（Stage 2）：HybridRetriever。

架构定位：一个统一检索抽象层，底下并列多个实现 BaseRetriever 的数据源，
collection 级隔离、不做物理合并。当前实际存在两个异构向量空间：

  BGE 文本向量空间      SemArt + （Stage 3）PDF 文字 chunk + （Stage 5）表格描述列
  DashScope 多模态空间  （Stage 3）PDF 整页图——维度/语义分布不同，不可混库

search 流程:
  1. 扇出：各数据源独立检索（BGE 空间数据源各自用共享 BGE 编码 query；
     Stage 3 起多模态数据源走 DashScope 编码另查一路）
  2. RRF（Reciprocal Rank Fusion）按源内排名融合多路结果——
     跨数据源的 score 绝对值不可比，排名可比
  3. 按 page_id/doc_id 去重（应对 Stage 3 双路线页面被多路命中）；
     无 page_id/doc_id 的结果（如 SemArt 行）不参与去重
  4. （Stage 4）qwen3-rerank 精排：粗排 top 15–20 重排后取 top_k；
     整页图等非文本候选不参与精排、保持原槽位；精排失败降级粗排原序；
     RERANK_ENABLED=0 或 search(rerank=False) 可关闭（eval A/B 用）
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("retrieval.hybrid")

CHROMA_DIR = Path(os.getenv("INDEX_DIR", "./data/index")) / "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# RRF 平滑常数（常用经验值 60）
_RRF_K = 60

# Stage 4 精排：RRF 粗排后送 qwen3-rerank 重排的候选池大小。
# 实测（2026-08-01，n=25 基线口径）：pool=20 时池召回 68.0%、pool=40 时
# 76.0%，精排两次都 100% 兑现池内召回——瓶颈在池召回不在排序，取 40。
RERANK_POOL = 40

# 不参与文本精排的源：整页图 content 只是占位标签，精排打分无意义且会
# 把多模态高相似页错误压底——它们保持粗排原槽位
_NON_RERANK_SOURCES = {"user_pdf_image"}


# ------------------------------------------------------------------ #
# 共享单例：Chroma collection + BGE embedding（原 tools/retrieval.py 迁入）#
# ------------------------------------------------------------------ #


# Chroma PersistentClient 按线程隔离：SQLite 连接不能安全地跨线程共享，
# 尤其 Windows 下 FastAPI BackgroundTasks（线程池）与主事件循环共用一份缓存
# Collection 时会触发 "attempt to write a readonly database"。每个线程持有自己
# 的 PersistentClient/连接，由 SQLite 文件级锁协调并发。
_chroma_local = threading.local()


def _get_thread_local_chroma_client():
    client = getattr(_chroma_local, "client", None)
    if client is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_local.client = client
    return client


def get_chroma_collection(name: str):
    """按名加载持久化 Chroma collection（每线程单例，必须已存在）。"""
    return _get_thread_local_chroma_client().get_collection(name)


def get_or_create_chroma_collection(name: str):
    """按名获取或创建 Chroma collection（每线程单例；用户上传文档用）。"""
    return _get_thread_local_chroma_client().get_or_create_collection(name)


@lru_cache(maxsize=1)
def _get_bge_model():
    """加载 BGE embedding 模型（全局单例）。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def get_bge_embed_fn() -> Callable[[str], list[float]]:
    """返回 BGE 编码函数；模型本体在首次真正编码时才加载。"""

    def embed(text: str) -> list[float]:
        return _get_bge_model().encode(text, normalize_embeddings=True).tolist()

    return embed


# ------------------------------------------------------------------ #
# RRF 融合 + 去重                                                       #
# ------------------------------------------------------------------ #


def _rrf_fuse(per_source: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    """按源内排名做 RRF 融合排序（Python sort 稳定，单源时严格保持原顺序）。"""
    fused_scores: dict[int, float] = {}
    for hits in per_source:
        for rank, hit in enumerate(hits):
            fused_scores[id(hit)] = fused_scores.get(id(hit), 0.0) + 1.0 / (
                _RRF_K + rank + 1
            )
    all_hits = [hit for hits in per_source for hit in hits]
    all_hits.sort(key=lambda h: -fused_scores[id(h)])
    return all_hits


def _dedup(hits: list[RetrievalResult]) -> list[RetrievalResult]:
    """双路线页面去重：同页文字 chunk 与整页图同时命中时，丢弃整页图。

    Stage 3 双路线页面（文字 chunk 与整页图共享 page_id）若被多路命中，
    保留文字 chunk——它是更精确的证据，且当前 LLM 尚不能直接读图，
    整页图仅在页面无可用文字层时作为兜底证据。同页的多个文字 chunk
    内容不同、互不冲突，全部保留；无 page_id 的结果（SemArt 行）不参与。
    """
    text_pages = {
        h.metadata["page_id"]
        for h in hits
        if h.source == "user_pdf_text" and h.metadata.get("page_id")
    }
    out: list[RetrievalResult] = []
    for h in hits:
        if (
            h.source == "user_pdf_image"
            and h.metadata.get("page_id") in text_pages
        ):
            continue
        out.append(h)
    return out


# ------------------------------------------------------------------ #
# Stage 4：qwen3-rerank 精排                                            #
# ------------------------------------------------------------------ #


def _rerank_enabled(override: Optional[bool]) -> bool:
    """精排开关：search 参数显式覆盖 > env RERANK_ENABLED（默认开）。"""
    if override is not None:
        return override
    return os.getenv("RERANK_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def _rerank_fused(query: str, fused: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    """
    粗排结果送 qwen3-rerank 精排。

    文本候选（semart/user_pdf_text 等）重排，rerank_score 写入 metadata
    （原生 score 不动，跨源不可比的纪律不变）；非文本候选（整页图）保持
    原槽位——文本精排器读不懂图，占位标签只会被打低分错误压底。
    精排失败或候选不足时返回粗排原序。
    """
    pool = fused[:RERANK_POOL]
    if len(pool) <= top_k:
        return pool  # 无排可重，省一次 API 调用
    text_slots = [i for i, h in enumerate(pool) if h.source not in _NON_RERANK_SOURCES]
    if not text_slots:
        return pool
    from src.retrieval.reranker import rerank

    ranked = rerank(query, [pool[i].content for i in text_slots], top_n=len(text_slots))
    if ranked is None or len(ranked) != len(text_slots):
        # 降级：精排是增强不是依赖。部分响应按槽位重排会把被移动的文档
        # 在原槽位复制一份（同一文档占两位），宁可整体回退粗排原序
        return pool
    reordered = list(pool)
    for slot, (doc_idx, score) in zip(text_slots, ranked):
        hit = pool[text_slots[doc_idx]]
        hit.metadata["rerank_score"] = round(score, 4)
        reordered[slot] = hit
    log_event(logger, "hybrid_rerank", pool=len(pool), text=len(text_slots))
    return reordered


# ------------------------------------------------------------------ #
# HybridRetriever                                                     #
# ------------------------------------------------------------------ #


class HybridRetriever:
    """多数据源统一检索：注册 → 扇出 → RRF 融合 → 去重。"""

    def __init__(self) -> None:
        self._retrievers: dict[str, BaseRetriever] = {}
        # Stage 5：当前生效的结构化数据源（用户在前端切换）。
        # semantic_search 工具无状态，检索时以它为 dataset_id 过滤——
        # 选 semart 时用户表格不参与，选用户表格时 SemArt 不参与；
        # 无 dataset_id 属性的检索器（用户 PDF 两路）不受切换影响，始终参与。
        self.active_dataset: str = "semart"

    def set_active_dataset(self, dataset_id: str) -> None:
        """切换当前生效数据源；只允许 semart 或已注册的用户表格。"""
        if dataset_id != "semart" and dataset_id not in self._retrievers:
            raise KeyError(f"未注册的数据源：{dataset_id}")
        self.active_dataset = dataset_id
        logger.info("[hybrid] 切换生效数据源 → %s", dataset_id)

    def register(self, source: str, retriever: BaseRetriever) -> None:
        """按 source 标签注册一个数据源检索器。"""
        self._retrievers[source] = retriever

    @property
    def retrievers(self) -> dict[str, BaseRetriever]:
        return dict(self._retrievers)

    def search(
        self,
        query: str,
        top_k: int = 5,
        sources: Optional[list[str]] = None,
        dataset_id: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> list[RetrievalResult]:
        """
        统一检索入口。

        sources:    只查指定 source 标签的数据源；None 表示全部已注册源。
        dataset_id: 限定当前生效的结构化数据源（Stage 5 用户表格接入后，
                    同一 source 标签下按 dataset_id 选具体表）；None 不限。
        rerank:     Stage 4 精排开关；None 取 env RERANK_ENABLED（默认开），
                    显式 False 跳过精排（eval A/B 用）。
        """
        use_rerank = _rerank_enabled(rerank)
        # 精排开启时各源多取候选（池化后再重排）；否则保持原样按需取
        fetch_k = RERANK_POOL if use_rerank else top_k

        per_source: list[list[RetrievalResult]] = []
        for name, retriever in self._retrievers.items():
            if sources is not None and name not in sources:
                continue
            r_dataset = getattr(retriever, "dataset_id", None)
            if dataset_id is not None and r_dataset is not None and r_dataset != dataset_id:
                continue
            try:
                hits = retriever.search(query, top_k=fetch_k, filters=None)
            except Exception as e:  # 单源失败不拖垮整体检索
                logger.warning("[hybrid] source=%s 检索失败：%s", name, e)
                hits = []
            per_source.append(hits)
            log_event(logger, "hybrid_source", source=name, hits=len(hits))

        fused = _dedup(_rrf_fuse(per_source))
        if use_rerank:
            fused = _rerank_fused(query, fused, top_k)
        return fused[:top_k]


# ------------------------------------------------------------------ #
# 全局单例：自动注册 SemArt 作为第一个数据源                             #
# ------------------------------------------------------------------ #

_hybrid: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """返回全局 HybridRetriever 单例（首次调用时注册全部已上线数据源）。"""
    global _hybrid
    if _hybrid is None:
        from src.retrieval.structured_retriever import get_structured_retriever
        from src.retrieval.userdoc_image_retriever import UserDocImageRetriever
        from src.retrieval.userdoc_text_retriever import UserDocTextRetriever

        hybrid = HybridRetriever()
        hybrid.register("semart", get_structured_retriever("semart"))
        # Stage 3：用户 PDF 两路检索器（collection 为空时自动返回空列表）
        hybrid.register("user_pdf_text", UserDocTextRetriever())
        hybrid.register("user_pdf_image", UserDocImageRetriever())
        _hybrid = hybrid
    return _hybrid
