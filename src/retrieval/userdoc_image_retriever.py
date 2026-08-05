"""
用户 PDF 多模态路线检索器：DashScope 多模态向量空间。

整页图片用 tongyi-embedding-vision-plus 编码（1152 维），与 BGE 文本空间
维度/语义分布不同，独立 collection（user_pdf_images）不混库。
查询时用同一模型编码文本 query，实现"文搜图"跨模态检索。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import get_or_create_chroma_collection
from src.utils.logging_config import get_logger

load_dotenv()

logger = get_logger("retrieval.userdoc_image")

COLLECTION_NAME = "user_pdf_images"
MM_EMBED_MODEL = "tongyi-embedding-vision-plus"  # 1152 维；qwen3-vl-embedding 为对照组


def get_mm_embed_fn():
    """返回 DashScope 多模态编码函数（文本/图片同空间）。"""

    def embed(item: dict) -> list[float]:
        """item: {"text": ...} 或 {"image": "data:image/...;base64,..."}"""
        import dashscope
        from dashscope import MultiModalEmbedding

        dashscope.api_key = os.getenv("LLM_API_KEY")
        resp = MultiModalEmbedding.call(model=MM_EMBED_MODEL, input=[item])
        if resp.status_code != 200:
            raise RuntimeError(f"多模态编码失败：{resp.code} {resp.message}")
        return resp.output["embeddings"][0]["embedding"]

    return embed


class UserDocImageRetriever:
    """用户上传 PDF 整页图的跨模态检索器（BaseRetriever 协议实现）。"""

    source = "user_pdf_image"
    dataset_id = None

    def search(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        collection = get_or_create_chroma_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return []  # 空库时跳过 DashScope 调用，零成本

        where = None
        if filters:
            conds = {
                k: v for k, v in filters.items() if k in ("doc_id", "kb_id") and v
            }
            if len(conds) == 1:
                where = conds
            elif len(conds) > 1:
                where = {"$and": [{k: v} for k, v in conds.items()]}

        try:
            query_vec = get_mm_embed_fn()({"text": query})
        except Exception as e:
            logger.warning("[userdoc_image] 查询编码失败：%s", e)
            return []

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
            **({"where": where} if where else {}),
        )

        out: list[RetrievalResult] = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            meta = dict(meta)
            out.append(
                RetrievalResult(
                    content=f"[整页图]《{meta.get('doc_name', '')}》第 {meta.get('page', '?')} 页",
                    source="user_pdf_image",
                    score=1 - dist,
                    metadata=meta,
                    image_refs=[meta["image_path"]] if meta.get("image_path") else [],
                )
            )
        return out
