"""
统一能力工具：原子子管线逻辑下沉（2026-08-02）。

comparison / timeline / recommendation 三个图分支的"确定性部分"
（分组、排除、证据收集）下沉为工具，交给 agent（ReAct）统一编排：
- compare_subjects：         按对象分组收集对比证据（保留"每个对象都检索"的保证）
- timeline_by_periods：      按数据集年代分组轴收集时期证据与配图（保留"不编年"的保证）
- recommend_with_exclusions：提炼风格特征 → 检索 → 排除用户已喜欢画家（保留"不重复推荐"的保证）

综合叙述仍由 agent 完成（工具只产出结构化证据）。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool


def group_by_artist(candidates: list[dict], per_artist: int = 2) -> dict[str, list[str]]:
    """把候选画作按画家分组，每人保留 top 作品标题（供"推荐画"粒度使用）。"""
    out: dict[str, list[str]] = {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        author = str(c.get("author") or "").strip()
        title = str(c.get("title") or "").strip()
        if not author or not title:
            continue
        titles = out.setdefault(author, [])
        if title not in titles and len(titles) < per_artist:
            titles.append(title)
    return out


@tool
def compare_subjects(
    subjects: list[str],
    dimensions: Optional[list[str]] = None,
) -> list[dict]:
    """按维度对比两个或多个画家/画作：对每个对象分别检索评论证据，按对象分组返回。

    适用场景：用户要求对比两位画家、两幅画或两种风格的差异。

    Args:
        subjects: 要对比的对象（画家或画作名，英文优先）
        dimensions: 对比维度关键词，如 ["color use", "brushwork", "composition"]

    Returns:
        每个对象的检索证据分组：{subject, query, evidence[]}
    """
    from src.retrieval.relevance import llm_relevance_filter
    from src.tools.retrieval import semantic_search

    dims = dimensions or ["style", "color", "technique"]
    dim_str = " ".join(dims)
    out: list[dict] = []
    for subject in subjects:
        query = f"{subject} {dim_str} painting style characteristics"
        try:
            results = semantic_search.invoke({"query": query, "top_k": 4})
        except Exception:
            results = []
        docs = llm_relevance_filter(query, results, min_keep=2)
        out.append({"subject": subject, "query": query, "evidence": docs})
    return out


@tool
def timeline_by_periods(subject: str) -> dict:
    """按数据集的年代分组轴梳理某画家/流派的时期与代表作品。

    适用场景：用户要求梳理某画家或流派随时间演变的风格。
    注意：同一姓名可能命中多个不同画家（如 Turner 混入 William Turner
    Dannat）——每个时期返回 artists 列表与全局 identity_note，agent 必须
    核对 subject 身份，发现异名作品时说明并排除，不得静默混用。

    Args:
        subject: 要梳理的画家/流派名称（英文优先）

    Returns:
        {subject, identity_note, periods: [{period, artists[], evidence[], images[]}], images[]}
    """
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.structured_retriever import get_structured_retriever
    from src.data.access import row_to_artwork_dict
    from src.tools.image_lookup import lookup_images

    dataset_id = get_hybrid_retriever().active_dataset
    retriever = get_structured_retriever(dataset_id)
    groups = retriever.group_by_axis(subject)

    periods: list[dict] = []
    images: list[dict] = []
    seen_artists: set[str] = set()
    for period, subset in list(groups.items())[:6]:
        artists = [
            str(v).strip()
            for v in subset[retriever.schema.entity_col].unique()
            if str(v).strip()
        ]
        seen_artists.update(artists)
        evidence = [
            row_to_artwork_dict(row)
            for _, row in subset.head(2).iterrows()
        ]
        imgs = lookup_images(author=subject, timeframe=period, top_k=1)
        images.extend(imgs)
        periods.append({
            "period": period,
            "artists": artists,
            "evidence": evidence,
            "images": imgs,
        })
    identity_note = ""
    if len(seen_artists) > 1:
        identity_note = (
            f"同名异人提示：检索到的作品归属多个不同的画家全名（{', '.join(sorted(seen_artists))}），"
            "回答前请核对 subject 身份并排除不匹配者的作品。"
        )
    return {
        "subject": subject,
        "identity_note": identity_note,
        "periods": periods,
        "images": images,
    }


@tool
def recommend_with_exclusions(preference: str, exclude_artists: list[str]) -> dict:
    """基于用户偏好推荐画家/作品：内部提炼风格特征 → 语义检索 → 排除用户已喜欢的画家。

    适用场景：用户表达喜欢某种风格/某位画家，希望推荐类似的画家或作品。

    Args:
        preference: 用户表达的偏好描述（可用中文）
        exclude_artists: 要排除的画家（用户已喜欢/已提到过的）

    Returns:
        {features, liked_artists, candidates: [{author, title, description_snippet}],
         by_artist: {author: [top titles]}}
    """
    from src.agent.nodes.common import parse_json
    from src.agent.prompts import RECOMMENDATION_FEATURE_PROMPT
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.structured_retriever import get_structured_retriever
    from src.tools.retrieval import semantic_search
    from src.utils.llm import get_llm

    # 1) 主观偏好 → 结构化风格特征（项目核心亮点：检索 query 是 Agent 推理产物）
    prompt = RECOMMENDATION_FEATURE_PROMPT.format(
        user_query=preference,
        preference_context="",
    )
    raw = get_llm(0.3).invoke(prompt).content
    parsed = parse_json(raw) or {}
    liked = parsed.get("liked_artists", []) if isinstance(parsed, dict) else []
    features = parsed.get("features", "") if isinstance(parsed, dict) else ""
    if not features:
        features = raw.strip()

    # 2) 特征检索 + 排除（用户明确排除的 + 特征提炼出的已喜欢画家）
    exclude = list(exclude_artists) + [str(a) for a in liked]
    try:
        results = semantic_search.invoke({"query": features, "top_k": 12})
    except Exception:
        results = []
    dataset_id = get_hybrid_retriever().active_dataset
    retriever = get_structured_retriever(dataset_id)
    filtered = retriever.exclude_from_results(results, exclude)

    return {
        "features": features,
        "liked_artists": liked,
        "candidates": filtered[:10],
        "by_artist": group_by_artist(filtered[:10]),
    }
