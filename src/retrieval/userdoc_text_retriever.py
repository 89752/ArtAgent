"""
用户 PDF 文字路线检索器：BGE 向量空间。

与 SemArt 同一 BGE embedding 模型、同一向量语义空间，天然可联合检索；
独立 collection（user_pdf_text），collection 级隔离不做物理合并。
"""

from __future__ import annotations

import hashlib
import math

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import get_bge_m3_embed_fn, get_or_create_chroma_collection
from src.utils.logging_config import get_logger

logger = get_logger("retrieval.userdoc_text")

COLLECTION_NAME = "user_pdf_text"
# The fallback must not share a collection with BGE: Chroma collections have a
# fixed vector dimension.  It keeps uploads usable in a fresh/offline Docker
# deployment without polluting the production BGE vector space.
FALLBACK_COLLECTION_NAME = "user_pdf_text_fallback"
_FALLBACK_DIM = 256
_bge_available: bool | None = None


def _fallback_embed(text: str) -> list[float]:
    """Small deterministic character n-gram embedding for offline user docs.

    This is deliberately a continuity fallback, not a replacement for BGE-M3:
    it gives exact/near-keyword recall for newly uploaded documents while a
    model-less sandbox or first deployment is brought online.
    """
    vector = [0.0] * _FALLBACK_DIM
    normalized = "".join(str(text or "").casefold().split())
    tokens = [normalized[i:i + 2] for i in range(max(0, len(normalized) - 1))]
    if not tokens and normalized:
        tokens = [normalized]
    for token in tokens:
        slot = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big"
        ) % _FALLBACK_DIM
        vector[slot] += 1.0
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector] if magnitude else vector


def get_userdoc_text_indexer() -> tuple[str, callable]:
    """Return the compatible collection and batch encoder for user documents.

    The BGE model is loaded lazily.  If it is unavailable locally, index only
    user uploads into a separate fallback collection instead of failing the
    whole ingestion job.  Core data remains strictly in its BGE space.
    """
    global _bge_available
    if _bge_available is not False:
        try:
            # Probe once here so an ingestion job can choose a compatible
            # collection before any vectors are written.
            get_bge_m3_embed_fn()("ArtAgent embedding readiness probe")
            _bge_available = True

            def bge_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
                from src.retrieval.hybrid import get_bge_m3_embed_batch

                return get_bge_m3_embed_batch()(texts, batch_size=batch_size)

            return COLLECTION_NAME, bge_batch
        except RuntimeError as exc:
            _bge_available = False
            logger.warning(
                "[userdoc_text] BGE-M3 unavailable; using isolated local fallback for user documents: %s",
                exc,
            )

    def fallback_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
        del batch_size
        return [_fallback_embed(text) for text in texts]

    return FALLBACK_COLLECTION_NAME, fallback_batch


class UserDocTextRetriever:
    """用户上传 PDF 文字 chunk 的检索器（BaseRetriever 协议实现）。"""

    source = "user_pdf_text"
    dataset_id = None  # 跨用户文档检索；结构化数据源的 dataset_id 过滤不作用于本类

    def search(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        # filters 支持 doc_id / kb_id 等值过滤（映射 Chroma where）
        where = None
        if filters:
            conds = {
                k: v for k, v in filters.items() if k in ("doc_id", "kb_id") and v
            }
            if len(conds) == 1:
                where = conds
            elif len(conds) > 1:
                where = {"$and": [{k: v} for k, v in conds.items()]}

        out: list[RetrievalResult] = []
        primary = get_or_create_chroma_collection(COLLECTION_NAME)
        fallback = get_or_create_chroma_collection(FALLBACK_COLLECTION_NAME)
        backends: list[tuple[object, list[float]]] = []
        if primary.count():
            try:
                backends.append((primary, get_bge_m3_embed_fn()(query)))
            except RuntimeError as exc:
                logger.warning("[userdoc_text] BGE 查询跳过：%s", exc)
        if fallback.count():
            backends.append((fallback, _fallback_embed(query)))

        for collection, query_vector in backends:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, collection.count()),
                include=["metadatas", "distances", "documents"],
                **({"where": where} if where else {}),
            )
            for meta, dist, doc in zip(
                results["metadatas"][0], results["distances"][0], results["documents"][0]
            ):
                out.append(
                    RetrievalResult(
                        content=doc or "",
                        source="user_pdf_text",
                        score=1 - dist,
                        metadata=dict(meta),
                    )
                )
        out.sort(key=lambda item: item.score, reverse=True)
        return out[:top_k]
