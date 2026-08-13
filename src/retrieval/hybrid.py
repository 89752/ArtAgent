"""
统一混合检索入口：HybridRetriever。

架构定位：一个统一检索抽象层，底下并列多个实现 BaseRetriever 的数据源，
collection 级隔离、不做物理合并。当前实际存在两个异构向量空间：

  BGE 文本向量空间      core 核心库 + PDF 文字 chunk + 表格描述列（均 bge-m3）
  DashScope 多模态空间  PDF 整页图——维度/语义分布不同，不可混库

search 流程:
  1. 扇出：各数据源独立检索（BGE 空间数据源各自用共享 BGE 编码 query；
     多模态数据源走 DashScope 编码另查一路）
  2. RRF（Reciprocal Rank Fusion）按源内排名融合多路结果——
     跨数据源的 score 绝对值不可比，排名可比
  3. 按 page_id/doc_id 去重（应对双路线页面被多路命中）；
     无 page_id/doc_id 的结果（如核心库行）不参与去重
  4. Jina Reranker v3.5 API 精排：粗排 top 40 重排后取 top_k；
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

def _chroma_dir() -> Path:
    """惰性解析 Chroma 目录（首次使用时），避免测试在 import 期修改 INDEX_DIR
    导致全局索引路径被污染（2026-08-04 全量套件串扰修复）。"""
    return Path(os.getenv("INDEX_DIR", "./data/index")) / "chroma"
# 核心库用多语言模型（中文提问 + 英文描述）：1024 维，~2.2GB，本地缓存后全离线
EMBEDDING_MODEL_M3 = "BAAI/bge-m3"
# bge-m3 默认 max_seq_length=8192：长文本 padding 会让批量编码极慢。
# 作品描述 p99≈2472 字符（约 620 token），1024 token 截断对质量几乎无损，
# 但能把最坏批次的算力降 8 倍（2026-08-02 实测：8192+fp32 下 39k 条需数小时）。
BGE_M3_MAX_SEQ_LENGTH = 1024

# RRF 平滑常数（常用经验值 60）
_RRF_K = 60

# ── 加权 RRF：通道权重（2026-08-02，Ragent 加权 RRF 借鉴） ────────
# score(hit) = Σ_weight_source / (k + rank)。新接入/可信度低的通道降权，
# 防止靠名次抢前排（如整页图是兜底证据、实时 API 只有元数据）。
# 每个通道可用 env `CHANNEL_WEIGHT_<SOURCE大写>` 覆盖（如 CHANNEL_WEIGHT_MET_MUSEUM=0.3），
# 未知源默认 1.0。core/表格等主力通道保持 1.0，与旧等权行为一致。
CHANNEL_WEIGHTS: dict[str, float] = {
    "core": 1.0,
    "extended": 1.0,
    "user_table": 1.0,
    "user_pdf_text": 1.0,
    "user_pdf_image": 0.5,  # 整页图是兜底证据（同页有文字已被 _dedup 丢弃），降权防干扰画作检索
    "met_museum": 0.5,      # 实时 API 预留：只有元数据，低权重
    "rijksmuseum": 0.5,
}


def _channel_weight(source: str) -> float:
    """返回通道权重；env 可覆盖（CHANNEL_WEIGHT_<SOURCE>），未知源默认 1.0。"""
    env_key = "CHANNEL_WEIGHT_" + source.upper()
    env_val = os.getenv(env_key)
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return CHANNEL_WEIGHTS.get(source, 1.0)


# 精排：RRF 粗排后送 reranker 重排的候选池大小。
# 实测（2026-08-01，n=25 基线口径）：pool=20 时池召回 68.0%、pool=40 时
# 76.0%，精排两次都 100% 兑现池内召回——瓶颈在池召回不在排序，默认取 40。
# 池子越大召回越高、单次精排耗时越长，可按延迟预算调小（如 15）。
try:
    RERANK_POOL = int(os.getenv("RERANK_POOL", "40"))
except ValueError:
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
    path = str(_chroma_dir())
    if client is None or getattr(_chroma_local, "path", None) != path:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        _chroma_local.client = client
        _chroma_local.path = path
    return client


def get_or_create_chroma_collection(name: str):
    """按名获取或创建 Chroma collection（每线程单例；用户上传文档用）。"""
    return _get_thread_local_chroma_client().get_or_create_collection(name)


@lru_cache(maxsize=1)
def _get_bge_m3_model():
    """加载 bge-m3 多语言模型（全局单例；核心库专用）。"""
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(EMBEDDING_MODEL_M3, trust_remote_code=True)
    model.max_seq_length = BGE_M3_MAX_SEQ_LENGTH
    # GPU 可用时用 fp16：显存减半、吞吐翻倍以上；fp16/fp32 的余弦相似度差异可忽略
    if torch.cuda.is_available():
        model.half()
    return model


def get_bge_m3_embed_fn() -> Callable[[str], list[float]]:
    """返回 bge-m3 编码函数（核心库与用户文档共享的多语言 embedding 空间）。"""

    def embed(text: str) -> list[float]:
        return _get_bge_m3_model().encode(text, normalize_embeddings=True).tolist()

    return embed


def get_bge_m3_embed_batch() -> Callable[[list[str], int], list[list[float]]]:
    """返回 bge-m3 批量编码函数（索引用：内部批处理，CPU 上远快于逐条调用）。"""

    def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return _get_bge_m3_model().encode(
            texts, normalize_embeddings=True, batch_size=batch_size
        ).tolist()

    return embed_batch


# ------------------------------------------------------------------ #
# RRF 融合 + 去重                                                       #
# ------------------------------------------------------------------ #


def _rrf_fuse(per_source: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    """按源内排名做加权 RRF 融合排序（默认等权时与旧行为逐位一致）。"""
    fused_scores: dict[int, float] = {}
    for hits in per_source:
        weight = _channel_weight(hits[0].source) if hits else 1.0
        for rank, hit in enumerate(hits):
            fused_scores[id(hit)] = fused_scores.get(id(hit), 0.0) + weight / (
                _RRF_K + rank + 1
            )
    all_hits = [hit for hits in per_source for hit in hits]
    all_hits.sort(key=lambda h: -fused_scores[id(h)])
    return all_hits


def _dedup(hits: list[RetrievalResult]) -> list[RetrievalResult]:
    """去重：向量与词法命中同一作品/同一 chunk 时保留 RRF 分高者；
    同页文字 chunk 与整页图同时命中时，丢弃整页图。

    双路线页面（文字 chunk 与整页图共享 page_id）若被多路命中，
    保留文字 chunk——它是更精确的证据，且当前 LLM 尚不能直接读图，
    整页图仅在页面无可用文字层时作为兜底证据。同页的多个文字 chunk
    内容不同、互不冲突，全部保留；无 page_id 的结果（核心库行）不参与。
    """
    text_pages = {
        h.metadata["page_id"]
        for h in hits
        if h.source == "user_pdf_text" and h.metadata.get("page_id")
    }
    seen: set[tuple] = set()
    out: list[RetrievalResult] = []
    for h in hits:
        key = None
        if h.source == "core":
            key = (
                "core",
                h.metadata.get("dedup_key")
                or (h.metadata.get("title"), h.metadata.get("artist")),
            )
        elif h.source == "user_pdf_text":
            key = ("pdf_text", h.content)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        if (
            h.source == "user_pdf_image"
            and h.metadata.get("page_id") in text_pages
        ):
            continue
        out.append(h)
    return out


# ------------------------------------------------------------------ #
# Jina Reranker v3.5 精排                                              #
# ------------------------------------------------------------------ #


def _rerank_enabled(override: Optional[bool]) -> bool:
    """精排开关：search 参数显式覆盖 > env RERANK_ENABLED（默认开）。"""
    if override is not None:
        return override
    return os.getenv("RERANK_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def _rerank_fused(query: str, fused: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    """
    粗排结果送 Jina Reranker v3.5 API 精排。

    文本候选（core/user_pdf_text 等）重排，rerank_score 写入 metadata
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
        # 当前生效的结构化数据源（用户在前端切换）。
        # semantic_search 工具无状态，检索时以它为 dataset_id 过滤——
        # 选 core 时用户表格不参与，选用户表格时核心库不参与；
        # 无 dataset_id 属性的检索器（用户 PDF 两路）不受切换影响，始终参与。
        self.active_dataset: str = "core"

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
        filters: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """
        统一检索入口。

        sources:    只查指定 source 标签的数据源；None 表示全部已注册源。
        dataset_id: 限定当前生效的结构化数据源（用户表格接入后，
                    同一 source 标签下按 dataset_id 选具体表）；None 不限。
        rerank:     精排开关；None 取 env RERANK_ENABLED（默认开），
                    显式 False 跳过精排（eval A/B 用）。
        filters:    结构化过滤条件（{字段: 值}），透传给各数据源实现；
                    与向量检索的 metadata 过滤语义一致。
        """
        use_rerank = _rerank_enabled(rerank)
        # 精排开启时各源多取候选（池化后再重排）；否则保持原样按需取
        fetch_k = RERANK_POOL if use_rerank else top_k

        per_source: list[list[RetrievalResult]] = []
        for name, retriever in self._retrievers.items():
            src_label = getattr(retriever, "source", name)
            if sources is not None and name not in sources and src_label not in sources:
                continue
            r_dataset = getattr(retriever, "dataset_id", None)
            if dataset_id is not None and r_dataset is not None and r_dataset != dataset_id:
                continue
            try:
                hits = retriever.search(query, top_k=fetch_k, filters=filters)
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
# 全局单例：自动注册 core 等数据源                                        #
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
        if os.getenv("LEXICAL_ENABLED", "1").strip().lower() not in ("0", "false", "no"):
            from src.retrieval.lexical import CoreLexicalRetriever, PdfBm25Retriever

            hybrid.register("core_lexical", CoreLexicalRetriever())
            hybrid.register("user_pdf_text_lexical", PdfBm25Retriever())
        # 用户 PDF 两路检索器（collection 为空时自动返回空列表）
        hybrid.register("user_pdf_text", UserDocTextRetriever())
        hybrid.register("user_pdf_image", UserDocImageRetriever())
        # M3：核心库注册（缺 CSV 时跳过，不打断启动）
        try:
            hybrid.register("core", get_structured_retriever("core"))
        except KeyError:
            pass
        # 默认数据源开关（DEFAULT_DATASET 可覆盖，默认 core）
        if os.getenv("DEFAULT_DATASET") == "core" and "core" in hybrid._retrievers:
            hybrid.active_dataset = "core"
        _hybrid = hybrid
    return _hybrid
