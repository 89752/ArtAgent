"""
场景3：基于偏好的链式推荐子管线（项目核心亮点）。

extract_features → feature_search → relevance_filter → synthesize
  - extract_features: LLM 把"浓烈奔放"等主观偏好推理成结构化风格特征
                      （检索 query 是 Agent 生成的中间产物，而非用户原话）
  - feature_search:   用提炼的特征做语义检索，排除用户已喜欢的画家
  - relevance_filter: LLM 判定候选画家是否真的匹配特征
  - synthesize:       组织推荐 + "为什么推荐"依据
"""

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.prompts import (
    RECOMMENDATION_FEATURE_PROMPT,
    RECOMMENDATION_FILTER_PROMPT,
    RECOMMENDATION_SYNTHESIZE_PROMPT,
)
from src.agent.nodes.common import parse_json, collect_artworks
from src.data.access import format_evidence_block
from src.utils.llm import get_llm, get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("recommend")


def recommendation_extract_features(state: AgentState) -> dict:
    """把主观偏好推理成结构化风格特征 + 抽取用户已喜欢的画家。"""
    # 若有历史偏好记忆，注入上下文（S5 与 S3 联动）
    pref_context = ""
    prefs = state.user_preferences or {}
    if prefs.get("artists") or prefs.get("styles"):
        pref_context = (
            f"\n（该用户历史偏好记忆：喜欢的画家 {prefs.get('artists', [])}，"
            f"偏好风格 {prefs.get('styles', [])}，可作为参考）"
        )

    prompt = RECOMMENDATION_FEATURE_PROMPT.format(
        user_query=state.user_query,
        preference_context=pref_context,
    )
    raw = get_llm(0.3).invoke(prompt).content
    parsed = parse_json(raw) or {}

    liked = parsed.get("liked_artists", []) if isinstance(parsed, dict) else []
    features = parsed.get("features", "") if isinstance(parsed, dict) else ""

    # 兜底：解析失败时把整段输出当作特征
    if not features:
        features = raw.strip()

    # 核心亮点可观测：主观偏好 → 推理出的结构化风格特征
    log_event(logger, "extract_features", liked=liked, features=features)
    return {
        "subjects": liked,
        "extracted_features": features,
        "current_step": "recommendation_extract_features",
    }


def recommendation_feature_search(state: AgentState) -> dict:
    """用提炼的特征检索，排除用户已喜欢的画家。"""
    from src.tools.retrieval import semantic_search

    try:
        results = semantic_search.invoke(
            {"query": state.extracted_features, "top_k": 12}
        )
    except Exception as e:
        logger.warning("[feature_search] semantic_search failed: %s", e)
        results = []

    # 排除用户已喜欢的画家本人的作品
    exclude_tokens = []
    for a in state.subjects:
        exclude_tokens.extend(t.lower() for t in a.split() if len(t) > 2)

    filtered = []
    for r in results:
        author_lower = r.get("author", "").lower()
        if any(tok in author_lower for tok in exclude_tokens):
            continue
        filtered.append(r)

    log_event(
        logger, "feature_search",
        raw_hits=len(results), after_exclude=len(filtered), excluded=state.subjects,
    )
    return {
        "retrieved_docs": {"candidates": filtered},
        "artworks": collect_artworks({"candidates": filtered}),
        "current_step": "recommendation_feature_search",
    }


def recommendation_relevance_filter(state: AgentState) -> dict:
    """LLM 判定候选画家是否真的匹配偏好特征。"""
    candidates = state.retrieved_docs.get("candidates", [])
    # 证据格式化统一走数据访问层
    cand_text = format_evidence_block(
        candidates, "- {author} | {title}: {description_snippet}"
    ) or "(无候选)"

    prompt = RECOMMENDATION_FILTER_PROMPT.format(
        extracted_features=state.extracted_features,
        exclude_artists="、".join(state.subjects) or "(无)",
        candidates=cand_text,
    )
    raw = get_deterministic_llm().invoke(prompt).content
    parsed = parse_json(raw)

    recommendations = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("author"):
                recommendations.append(
                    {"author": item["author"], "reason": item.get("reason", "")}
                )

    log_event(
        logger, "relevance_filter",
        candidates_in=len(candidates),
        recommended=[r["author"] for r in recommendations[:4]],
    )
    return {
        "candidates": recommendations[:4],
        "current_step": "recommendation_relevance_filter",
    }


def recommendation_synthesize(state: AgentState) -> dict:
    """组织最终推荐。"""
    if not state.candidates:
        rec_text = "(未能筛选出匹配的画家，可基于风格特征给出通用建议)"
    else:
        rec_text = "\n".join(
            f"- {c['author']}：{c['reason']}" for c in state.candidates
        )

    prompt = RECOMMENDATION_SYNTHESIZE_PROMPT.format(
        user_query=state.user_query,
        extracted_features=state.extracted_features,
        recommendations=rec_text,
    )
    answer = get_llm(0.5).invoke(prompt).content
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "current_step": "recommendation_synthesize",
    }
