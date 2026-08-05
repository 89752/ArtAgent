"""
用户 PDF 文字路线检索器：BGE 向量空间。

与 SemArt 同一 BGE embedding 模型、同一向量语义空间，天然可联合检索；
独立 collection（user_pdf_text），collection 级隔离不做物理合并。
"""

from __future__ import annotations

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import get_bge_m3_embed_fn, get_or_create_chroma_collection
from src.utils.logging_config import get_logger

logger = get_logger("retrieval.userdoc_text")

COLLECTION_NAME = "user_pdf_text"


class UserDocTextRetriever:
    """用户上传 PDF 文字 chunk 的检索器（BaseRetriever 协议实现）。"""

    source = "user_pdf_text"
    dataset_id = None  # 跨用户文档检索；结构化数据源的 dataset_id 过滤不作用于本类

    def search(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        collection = get_or_create_chroma_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return []

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

        results = collection.query(
            query_embeddings=[get_bge_m3_embed_fn()(query)],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances", "documents"],
            **({"where": where} if where else {}),
        )

        out: list[RetrievalResult] = []
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
        return out
