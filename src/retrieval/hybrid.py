"""
统一混合检索入口（Stage 2）：HybridRetriever。

架构定位：一个统一检索抽象层，底下并列多个实现 BaseRetriever 的数据源，
collection 级隔离、不做物理合并。当前实际存在两个异构向量空间：

  BGE 文本向量空间      SemArt + （Stage 3）PDF 文字 chunk + （Stage 5）表格描述列
  DashScope 多模态空间  （Stage 3）PDF 整页图——维度/语义分布不同，不可混库

search 流程：
  1. 扇出：各数据源独立检索（BGE 空间数据源各自用共享 BGE 编码 query；
     Stage 3 起多模态数据源走 DashScope 编码另查一路）
  2. RRF（Reciprocal Rank Fusion）按源内排名融合多路结果——
     跨数据源的 score 绝对值不可比，排名可比
  3. 按 page_id/doc_id 去重（应对 Stage 3 双路线页面被多路命中）；
     无 page_id/doc_id 的结果（如 SemArt 行）不参与去重
"""

from __future__ import annotations

import os
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


# ------------------------------------------------------------------ #
# 共享单例：Chroma collection + BGE embedding（原 tools/retrieval.py 迁入）#
# ------------------------------------------------------------------ #


@lru_cache(maxsize=8)
def get_chroma_collection(name: str):
    """按名加载持久化 Chroma collection（每名单例）。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name)


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
    """按 page_id/doc_id 去重：同键只保留排名最高的一条。

    Stage 3 双路线页面（文字 chunk + 整页图共享 page_id）依赖此逻辑避免
    重复引用；metadata 中两键皆无的结果（SemArt 行）原样保留。
    """
    seen: set = set()
    out: list[RetrievalResult] = []
    for h in hits:
        key = h.metadata.get("page_id") or h.metadata.get("doc_id")
        if key is None:
            out.append(h)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


# ------------------------------------------------------------------ #
# HybridRetriever                                                     #
# ------------------------------------------------------------------ #


class HybridRetriever:
    """多数据源统一检索：注册 → 扇出 → RRF 融合 → 去重。"""

    def __init__(self) -> None:
        self._retrievers: dict[str, BaseRetriever] = {}

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
    ) -> list[RetrievalResult]:
        """
        统一检索入口。

        sources:    只查指定 source 标签的数据源；None 表示全部已注册源。
        dataset_id: 限定当前生效的结构化数据源（Stage 5 用户表格接入后，
                    同一 source 标签下按 dataset_id 选具体表）；None 不限。
        """
        per_source: list[list[RetrievalResult]] = []
        for name, retriever in self._retrievers.items():
            if sources is not None and name not in sources:
                continue
            r_dataset = getattr(retriever, "dataset_id", None)
            if dataset_id is not None and r_dataset is not None and r_dataset != dataset_id:
                continue
            try:
                hits = retriever.search(query, top_k=top_k, filters=None)
            except Exception as e:  # 单源失败不拖垮整体检索
                logger.warning("[hybrid] source=%s 检索失败：%s", name, e)
                hits = []
            per_source.append(hits)
            log_event(logger, "hybrid_source", source=name, hits=len(hits))

        fused = _dedup(_rrf_fuse(per_source))
        return fused[:top_k]


# ------------------------------------------------------------------ #
# 全局单例：自动注册 SemArt 作为第一个数据源                             #
# ------------------------------------------------------------------ #

_hybrid: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """返回全局 HybridRetriever 单例（首次调用时注册 SemArt 数据源）。"""
    global _hybrid
    if _hybrid is None:
        from src.retrieval.structured_retriever import get_structured_retriever

        hybrid = HybridRetriever()
        hybrid.register("semart", get_structured_retriever("semart"))
        _hybrid = hybrid
    return _hybrid
