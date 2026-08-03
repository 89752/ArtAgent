"""
结构化表检索器（Stage 2）：TableSchema + StructuredTableRetriever。

核心思想：timeline / recommendation 依赖的不是某个具体数据集名字，
而是"当前数据源的 schema 声明了它有实体列 / 分组轴列 / 描述列"这个抽象。
内置核心库（dataset_id="core"）与 Stage 5 用户上传的 CSV/Excel
都以同样方式注册复用同一套能力。

懒加载设计：注册时只落 schema 与若干 loader，不真正读 CSV / 开 Chroma /
加载 BGE 模型——路由层能力开关只查 schema，不能让每次意图分类都付出
数据集加载代价。各资源在首次实际使用时才解析并缓存。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel

from src.data.access import fuzzy_match
from src.retrieval.base import RetrievalResult, RetrievalSource
from src.utils.logging_config import get_logger

logger = get_logger("retrieval.structured")


# ------------------------------------------------------------------ #
# TableSchema：结构化表的"角色声明"                                     #
# ------------------------------------------------------------------ #


class TableSchema(BaseModel):
    """声明一张结构化表中各列扮演的角色（SemArt 的 AUTHOR/TIMEFRAME/... 搬进配置）。"""

    entity_col: str  # 实体名列，如 SemArt 的 AUTHOR
    group_axis_col: Optional[str] = None  # 分组/时间轴列，如 TIMEFRAME；没有则 None
    description_col: str  # 描述文本列，如 DESCRIPTION
    image_col: Optional[str] = None  # 图片引用列，如 IMAGE_FILE
    # 检索工具（exact_lookup / query_painter_knowledge / image_lookup）按角色取列，
    # 不再硬编码 SemArt 列名（2026-08-02 核心库切换改造）
    title_col: Optional[str] = None
    date_col: Optional[str] = None
    technique_col: Optional[str] = None
    school_col: Optional[str] = None

    @property
    def supports_timeline(self) -> bool:
        """是否有分组轴可支撑 timeline 管线（由 group_axis_col 是否存在推出）。"""
        return bool(self.group_axis_col)

    @property
    def supports_recommendation(self) -> bool:
        """是否有实体列+描述列可支撑 recommendation 管线。"""
        return bool(self.entity_col and self.description_col)


# 核心库（从零构建）schema：列名自由，角色对齐 TableSchema
CORE_SCHEMA = TableSchema(
    entity_col="artist",
    group_axis_col="year_bucket",
    description_col="description",
    image_col="image_url",
    title_col="title",
    date_col="year_display",
    technique_col="material",
    school_col="movement",
)

# 核心库归一化后的作品表（normalize_core.py 产出；不存在则不注册 core）
CORE_DATA_PATH = Path(os.getenv("CORE_DATA_PATH", "./data/core/artworks_core.csv"))


def _entity_tokens(names: list[str]) -> list[str]:
    """实体排除名单的分词：长度 > 2 的词转小写（与 Stage 1 前 recommendation
    节点内联的排除逻辑完全一致，"Van Gogh" → ["van", "gogh"]）。"""
    tokens: list[str] = []
    for name in names:
        tokens.extend(t.lower() for t in str(name).split() if len(t) > 2)
    return tokens


# ------------------------------------------------------------------ #
# StructuredTableRetriever                                            #
# ------------------------------------------------------------------ #


class StructuredTableRetriever:
    """一张结构化表的统一访问入口：结构化操作 + 可选向量语义检索。

    - group_by_axis / exclude_by_entity：给确定性管线（timeline/recommendation）用
    - search：BaseRetriever 协议实现，接入 HybridRetriever。
      挂了向量集合（SemArt）时走 BGE 向量检索；否则退化走 access.fuzzy_match
      （Stage 5 无索引表格的兜底路径）。
    """

    def __init__(
        self,
        dataset_id: str,
        schema: TableSchema,
        *,
        source: RetrievalSource = "user_table",
        df: Optional[pd.DataFrame] = None,
        df_loader: Optional[Callable[[], pd.DataFrame]] = None,
        collection_loader: Optional[Callable[[], Any]] = None,
        embed_fn_loader: Optional[Callable[[], Callable[[str], list[float]]]] = None,
    ):
        self.dataset_id = dataset_id
        self.schema = schema
        self.source = source
        self._df = df
        self._df_loader = df_loader
        self._collection: Any = None
        self._collection_loader = collection_loader
        self._embed_fn: Optional[Callable[[str], list[float]]] = None
        self._embed_fn_loader = embed_fn_loader

    # ── 懒加载资源 ────────────────────────────────────────────────
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if self._df_loader is None:
                raise RuntimeError(f"数据源 {self.dataset_id} 未提供 DataFrame")
            self._df = self._df_loader()
        return self._df

    def _get_collection(self) -> Any:
        if self._collection is None and self._collection_loader is not None:
            self._collection = self._collection_loader()
        return self._collection

    def _get_embed_fn(self) -> Optional[Callable[[str], list[float]]]:
        if self._embed_fn is None and self._embed_fn_loader is not None:
            self._embed_fn = self._embed_fn_loader()
        return self._embed_fn

    # ── 结构化操作：timeline ──────────────────────────────────────
    def group_by_axis(self, entity: str) -> dict[str, pd.DataFrame]:
        """按分组轴列对某个实体的记录分组（给 timeline 用）。

        返回 {分组值: 子 DataFrame}，分组值按字符串升序（SemArt 的
        "1851-1900" 形式天然可按时间排序）。分组轴为空的记录归入 "Unknown"：
        存在真实分组时 Unknown 组被丢弃；完全没有真实分组时返回
        {"Unknown": 全部}。行为与 Stage 1 前 timeline 节点内联逻辑一致。
        """
        if not self.schema.group_axis_col:
            return {}
        works = fuzzy_match(self.df, self.schema.entity_col, entity)
        if works.empty:
            return {}

        works = works.copy()
        axis = self.schema.group_axis_col
        works["_AXIS"] = works[axis].fillna("").map(lambda v: v if v else "Unknown")
        keys = sorted(k for k in works["_AXIS"].unique() if k and k != "Unknown")
        if not keys:
            keys = ["Unknown"]
        return {k: works[works["_AXIS"] == k] for k in keys}

    # ── 结构化操作：recommendation ────────────────────────────────
    def entity_matches(self, value: str, names: list[str]) -> bool:
        """判断某实体字段值是否命中排除名单（分词包含，忽略大小写）。"""
        value_lower = (value or "").lower()
        return any(tok in value_lower for tok in _entity_tokens(names))

    def exclude_by_entity(self, names: list[str]) -> pd.DataFrame:
        """返回排除命中实体后的 DataFrame（实体列分词包含匹配）。"""
        tokens = _entity_tokens(names)
        if not tokens or not self.schema.entity_col:
            return self.df
        col = self.schema.entity_col
        mask = self.df[col].astype(str).str.lower().map(
            lambda v: not any(tok in v for tok in tokens)
        )
        return self.df[mask]

    def exclude_from_results(
        self, results: list[dict], names: list[str]
    ) -> list[dict]:
        """从检索结果字典列表中排除命中实体的条目（recommendation 实际使用）。

        结果字典来自 row_to_artwork_dict 归一化形状（小写 key），实体字段
        按 schema.entity_col 小写定位（SemArt：AUTHOR → author）。
        """
        if not names:
            return list(results)
        key = self.schema.entity_col.lower()
        # 兼容：core 结果经 _format_result 输出为 author 键（而非 artist）
        return [
            r for r in results
            if not self.entity_matches(r.get(key) or r.get("author") or "", names)
        ]

    # ── BaseRetriever 协议：HybridRetriever 融合入口 ───────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """语义检索：挂了向量集合走 BGE 向量检索，否则走 fuzzy_match 兜底。"""
        collection = self._get_collection()
        embed_fn = self._get_embed_fn()
        if collection is not None and embed_fn is not None:
            if collection.count() == 0:
                return []  # 空集合短路（core 未索引时），避免 n_results=0 报错
            return self._vector_search(collection, embed_fn, query, top_k)
        return self._fuzzy_search(query, top_k, filters)

    def _vector_search(
        self, collection: Any, embed_fn: Callable[[str], list[float]],
        query: str, top_k: int,
    ) -> list[RetrievalResult]:
        """BGE 向量空间检索（SemArt 路径，与原 semantic_search 行为一致）。"""
        results = collection.query(
            query_embeddings=[embed_fn(query)],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )
        out: list[RetrievalResult] = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            meta = dict(meta)
            meta["dataset_id"] = self.dataset_id
            out.append(
                RetrievalResult(
                    content=str(meta.get("description") or meta.get("title") or ""),
                    source=self.source,
                    score=1 - dist,
                    metadata=meta,
                    image_refs=[meta["file"]] if meta.get("file") else [],
                )
            )
        return out

    def _fuzzy_search(
        self, query: str, top_k: int, filters: dict | None
    ) -> list[RetrievalResult]:
        """无向量索引表的兜底检索（三级递进）：

        1. 实体列 fuzzy_match（短查询/指名查询主路径）
        2. 描述列整串包含（短语级查询）
        3. 词重叠打分（长查询兜底——Stage 5 实测 recommendation 的
           extracted_features 是 30–60 词特征描述，整串包含必空：
           按内容词在"实体列+描述列"中的命中数打分排序，确定性、无模型）

        filters（可选）：{列名: 值} 等值过滤，作用于命中的 DataFrame。
        用户表的 entity_col/description_col 可能为空（负样本表），
        空角色列直接跳过对应路径，两路皆空则返回空结果而非抛 KeyError。
        """
        scored: list[tuple[float, int, pd.Series]] | None = None
        hits = pd.DataFrame()
        if self.schema.entity_col and self.schema.entity_col in self.df.columns:
            hits = fuzzy_match(self.df, self.schema.entity_col, query)
        if hits.empty:
            desc_col = self.schema.description_col
            if desc_col and desc_col in self.df.columns:
                mask = (
                    self.df[desc_col]
                    .astype(str)
                    .str.lower()
                    .str.contains(str(query).lower(), na=False, regex=False)
                )
                hits = self.df[mask]
        if hits.empty:
            scored = self._word_overlap_rows(query)

        if scored is not None:
            rows_iter = [(row, score) for score, _, row in scored[:top_k]]
        else:
            if not hits.empty and filters:
                for col, value in filters.items():
                    if col in hits.columns:
                        hits = hits[hits[col].astype(str) == str(value)]
            rows_iter = [(row, 1.0) for _, row in hits.head(top_k).iterrows()]

        out: list[RetrievalResult] = []
        for row, score in rows_iter:
            meta = {
                str(c).lower(): ("" if pd.isna(row[c]) else str(row[c]))
                for c in self.df.columns
            }
            meta["dataset_id"] = self.dataset_id
            image = ""
            if self.schema.image_col and self.schema.image_col in self.df.columns:
                image = str(row.get(self.schema.image_col) or "")
            content = ""
            if self.schema.description_col and self.schema.description_col in self.df.columns:
                content = str(row.get(self.schema.description_col) or "")
            out.append(
                RetrievalResult(
                    content=content,
                    source=self.source,
                    score=score,
                    metadata=meta,
                    image_refs=[image] if image else [],
                )
            )
        return out

    def _word_overlap_rows(
        self, query: str, max_tokens: int = 20
    ) -> list[tuple[float, int, pd.Series]]:
        """第三级兜底：词重叠打分，返回 [(命中率, 原行序, row)] 降序。

        命中文本 = 实体列 + 描述列拼接（小写）；查询取长度 >3 的内容词
        （去重、上限 max_tokens）。命中率为 0 的行不返回。
        """
        text_cols = [
            c
            for c in (self.schema.entity_col, self.schema.description_col)
            if c and c in self.df.columns
        ]
        if not text_cols:
            return []
        tokens = list(dict.fromkeys(
            w for w in re.findall(r"[a-zA-Z]{4,}", str(query).lower())
        ))[:max_tokens]
        if not tokens:
            return []
        corpus = (
            self.df[text_cols]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        scored: list[tuple[float, int, pd.Series]] = []
        for idx, text in corpus.items():
            n = sum(1 for t in tokens if t in text)
            if n:
                scored.append((n / len(tokens), idx, self.df.loc[idx]))
        # 命中率降序；同分保持原行序（确定性）
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored


# ------------------------------------------------------------------ #
# 数据源注册表（SemArt 为第一个注册实例）                                #
# ------------------------------------------------------------------ #

_REGISTRY: dict[str, StructuredTableRetriever] = {}


def register_structured_dataset(
    dataset_id: str,
    schema: TableSchema,
    **kwargs: Any,
) -> StructuredTableRetriever:
    """注册一个结构化数据源，返回其 StructuredTableRetriever 实例。"""
    retriever = StructuredTableRetriever(dataset_id, schema, **kwargs)
    _REGISTRY[dataset_id] = retriever
    logger.info(
        "[register] dataset_id=%s source=%s timeline=%s recommendation=%s",
        dataset_id, retriever.source,
        schema.supports_timeline, schema.supports_recommendation,
    )
    return retriever


def _register_core() -> None:
    """把核心库注册为内置结构化数据源（df / Chroma / BGE 全部懒加载）。

    数据未就绪（CORE_DATA_PATH 不存在）时抛 KeyError——调用方
    （能力开关/注册表）会捕获并降级 general，不打断启动。
    """
    path = CORE_DATA_PATH
    if not Path(path).exists():
        raise KeyError(f"核心库数据未就绪：{path}（先跑 normalize_core.py）")

    def df_loader():
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        # 核心库 CSV 列名 → 角色列/展示键 归一化（2026-08-02）：
        # CSV 用 artist_name/year_display/material/movement/year_bucket/image_url，
        # schema 与 row_to_artwork_dict 统一用 artist/author/date/technique/
        # school/timeframe/image_file——不加别名会 KeyError 或产出空证据。
        if "artist" not in df.columns and "artist_name" in df.columns:
            df["artist"] = df["artist_name"]
        # CSV 无 year_display（index_core 入库时才算）：这里按同一规则补，
        # 保证结构化工具与 Chroma metadata 的日期显示一致
        if "year_display" not in df.columns:
            inc = (
                df["inception"].astype(str).str.strip()
                if "inception" in df.columns
                else ""
            )
            yr = df["year"].astype(str).str.strip() if "year" in df.columns else ""
            df["year_display"] = inc.where(inc != "", yr)
        alias_map = {
            "year_display": "date",
            "material": "technique",
            "movement": "school",
            "year_bucket": "timeframe",
            "image_url": "image_file",
        }
        for src, dst in alias_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]
        if "author" not in df.columns:
            df["author"] = df["artist"]
        return df

    from src.retrieval.hybrid import get_bge_m3_embed_fn, get_or_create_chroma_collection

    register_structured_dataset(
        "core",
        CORE_SCHEMA,
        source="core",
        df_loader=df_loader,
        collection_loader=lambda: get_or_create_chroma_collection("core"),
        embed_fn_loader=get_bge_m3_embed_fn,  # 核心库中文查询 → 多语言向量空间
    )


# 内置数据集注册表：首次访问懒注册
_BUILTIN_REGISTRARS = {"core": _register_core}


def get_structured_retriever(dataset_id: str = "core") -> StructuredTableRetriever:
    """按 dataset_id 取已注册的结构化检索器；内置数据集首次访问懒注册。"""
    if dataset_id not in _REGISTRY:
        registrar = _BUILTIN_REGISTRARS.get(dataset_id)
        if registrar is None:
            raise KeyError(f"未注册的数据源：{dataset_id}")
        registrar()
    return _REGISTRY[dataset_id]
