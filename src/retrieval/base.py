"""
检索层基础抽象（Stage 2）。

RetrievalResult：跨数据源统一的检索结果载体，source 标签保证可追溯性
（回答里能区分"来自 SemArt 结构化库 / 用户 PDF 第 N 页 / 博物馆实时 API"）。

BaseRetriever：数据源接入协议。新增数据源只需实现 search()；
nearby_venues / wiki_lookup / color_analysis 这类与"向量相似度"没有
可比性的外部工具不实现本协议，不参与 HybridRetriever 自动融合。
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# 数据源标签：met_museum / rijksmuseum 为 Stage 7 预留（接口先定、暂不实现）
RetrievalSource = Literal[
    "semart",
    "user_table",
    "user_pdf_text",
    "user_pdf_image",
    "met_museum",
    "rijksmuseum",
]


class RetrievalResult(BaseModel):
    """一条跨数据源统一的检索结果。"""

    # 给 LLM 看的文本内容（SemArt 为描述文本；PDF 为 chunk / 整页图说明）
    content: str
    source: RetrievalSource
    # 数据源原生的相似度分数（SemArt 向量检索为 1 - cosine distance）。
    # 注意：跨数据源的 score 不可直接比较，跨源排序由 HybridRetriever 的
    # RRF 按"源内排名"融合，不依赖 score 的绝对值。
    score: float = 0.0
    # doc_id / page_id / block_type / dataset_id 等；去重与过滤的载体
    metadata: dict = Field(default_factory=dict)
    # 关联图片引用（SemArt 的 IMAGE_FILE、PDF 整页图路径等），供前端展示
    image_refs: list[str] = Field(default_factory=list)


@runtime_checkable
class BaseRetriever(Protocol):
    """数据源检索协议：新增数据源实现本协议即可接入 HybridRetriever。"""

    # 该数据源产出的 RetrievalResult.source 值
    source: RetrievalSource

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """按 query 检索，返回源内按相关性降序的结果列表。

        filters 为数据源相关的结构化过滤条件（保留参数，Stage 2 各实现可忽略）。
        """
        ...
