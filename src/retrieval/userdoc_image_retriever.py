"""用户 PDF 多模态路线检索器（嵌入提供商可配置）。

整页图片用多模态嵌入模型编码（默认 DashScope tongyi-embedding-vision-plus，
1152 维），与 BGE 文本空间维度/语义分布不同，独立 collection
（user_pdf_images）不混库。查询时用同一模型编码文本 query，实现
"文搜图"跨模态检索。

提供商经 config.yaml 的 retrieval.pdf_image_embed_* 配置：
  - provider: dashscope（默认）或 openai（OpenAI 兼容 /embeddings 端点）；
  - model / api_key / base_url 均可覆盖，api_key / base_url 缺省回落对话模型配置。

注意：切换嵌入模型会改变向量空间，旧向量不可比，需重建 user_pdf_images
collection 后重新入库。
"""

from __future__ import annotations

from dotenv import load_dotenv

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import get_or_create_chroma_collection
from src.utils.config import get
from src.utils.logging_config import get_logger

load_dotenv()

logger = get_logger("retrieval.userdoc_image")

COLLECTION_NAME = "user_pdf_images"


def _dashscope_embed_fn(model: str, api_key: str):
    """DashScope 多模态编码（文本/图片同空间）。"""

    def embed(item: dict) -> list[float]:
        """item: {"text": ...} 或 {"image": "data:image/...;base64,..."}"""
        import dashscope
        from dashscope import MultiModalEmbedding

        dashscope.api_key = api_key
        resp = MultiModalEmbedding.call(model=model, input=[item])
        if resp.status_code != 200:
            raise RuntimeError(f"多模态编码失败：{resp.code} {resp.message}")
        return resp.output["embeddings"][0]["embedding"]

    return embed


def _to_openai_input(item: dict):
    """把 DashScope 风格 item（{"text": ...} / {"image": "data:..."}）
    转成 OpenAI 兼容 embeddings 接受的字符串输入。"""
    if "text" in item:
        return item["text"]
    if "image" in item:
        return item["image"]
    return item


def _openai_embed_fn(model: str, api_key: str, base_url: str):
    """OpenAI 兼容 /embeddings 调用（多模态能力取决于端点实现）。"""

    def embed(item: dict) -> list[float]:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.embeddings.create(
            model=model,
            input=[_to_openai_input(item)],
            encoding_format="float",
        )
        return resp.data[0].embedding

    return embed


def get_mm_embed_fn():
    """返回多模态编码函数（按 config 选择提供商）。"""
    provider = str(
        get("retrieval.pdf_image_embed_provider", "dashscope")
    ).strip().lower()
    model = str(get("retrieval.pdf_image_embed_model", "tongyi-embedding-vision-plus"))
    api_key = get("retrieval.pdf_image_embed_api_key") or get("models.llm_api_key")
    base_url = get("retrieval.pdf_image_embed_base_url") or get("models.llm_base_url")

    if provider == "dashscope":
        if not api_key:
            raise ValueError(
                "缺少多模态嵌入 API Key"
                "（retrieval.pdf_image_embed_api_key 或 LLM_API_KEY）"
            )
        return _dashscope_embed_fn(model, api_key)
    if provider == "openai":
        if not api_key or not base_url:
            raise ValueError(
                "openai 嵌入提供商需要 api_key 与 base_url"
                "（缺省回落对话模型配置）"
            )
        return _openai_embed_fn(model, api_key, base_url)
    raise ValueError(
        f"未知的 PDF 图片嵌入提供商：{provider}（支持 dashscope / openai）"
    )


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
