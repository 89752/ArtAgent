"""P1-2 aggregate_stats：结构化统计工具（零 LLM，本地确定性计算）。

按当前数据源的 schema 角色列做分组计数，回答"哪个时期作品最多"
"哪种技法最常见"类问题，替代 semantic_search 硬查。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain_core.tools import tool


def _column_for(schema, group_by: str) -> Optional[str]:
    """把语义分组名映射到当前 schema 的列名。"""
    mapping = {
        "school": schema.school_col,
        "timeframe": schema.group_axis_col,
        "technique": schema.technique_col,
        "author": schema.entity_col,
    }
    col = mapping.get(group_by)
    if col is None and group_by in schema.model_fields:
        col = getattr(schema, group_by)
    return col


def _apply_filters(df: pd.DataFrame, schema, filters: Optional[dict]) -> pd.DataFrame:
    """按 author/school/timeframe 做大小写不敏感包含过滤。"""
    if not filters:
        return df
    alias = {
        "author": schema.entity_col,
        "school": schema.school_col,
        "timeframe": schema.group_axis_col,
    }
    for key, value in filters.items():
        if not value:
            continue
        col = alias.get(key)
        if col and col in df.columns:
            df = df[df[col].astype(str).str.lower().str.contains(
                str(value).lower(), na=False, regex=False
            )]
    return df


@tool
def aggregate_stats(
    group_by: str,
    filters: Optional[dict] = None,
    top_k: int = 10,
) -> dict:
    """按当前数据源做分组统计（作品数量/占比/代表作品）。

    适用场景：用户问"印象派哪个时期作品最多""藏画中哪种技法最常见"
    "某画家各时期的作品分布"等统计类问题。

    Args:
        group_by: 分组维度："school" 流派 / "timeframe" 时期 / "technique" 技法 /
                  "author" 画家，或当前数据源自定义列名
        filters:  可选过滤 {author/school/timeframe: 值}（包含匹配）
        top_k:    最多返回多少组（默认10）

    Returns:
        {group_by, total, groups: [{value, count, ratio, sample_titles[]}],
         note}
    """
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.structured_retriever import get_structured_retriever

    dataset_id = get_hybrid_retriever().active_dataset
    retriever = get_structured_retriever(dataset_id)
    schema = retriever.schema
    df = _apply_filters(retriever.df, schema, filters)

    col = _column_for(schema, group_by)
    if col is None or col not in df.columns:
        return {
            "group_by": group_by,
            "total": 0,
            "groups": [],
            "note": f"当前数据源 {dataset_id} 不支持按 {group_by} 分组",
        }

    total = len(df)
    if total == 0:
        return {"group_by": group_by, "total": 0, "groups": [], "note": "无匹配记录"}

    counts = (
        df[col]
        .fillna("")
        .astype(str)
        .map(lambda v: v if v else "(未知)")
        .value_counts()
    )
    title_col = schema.title_col
    groups: list[dict] = []
    for value, count in counts.head(top_k).items():
        subset = df[df[col].fillna("").astype(str).map(
            lambda v: v if v else "(未知)"
        ) == value]
        samples = (
            subset[title_col].head(3).tolist()
            if title_col and title_col in subset.columns
            else []
        )
        groups.append(
            {
                "value": value,
                "count": int(count),
                "ratio": round(float(count) / total, 3),
                "sample_titles": [str(s) for s in samples],
            }
        )
    return {"group_by": group_by, "total": total, "groups": groups, "note": ""}
