"""
检索结果相关性校正（Stage 4）：通用轻量 LLM 过滤步骤。

定位：recommendation 的 rec_filter 思路的通用化。向量检索（即便经
qwen3-rerank 精排）仍会混入"形似而答非所问"的噪声候选；确定性管线把
证据直接拼进合成 prompt，没有自救机会。本步骤在 HybridRetriever 之后、
证据消费之前，用一次轻量确定性 LLM 调用剔掉不相关候选。

适用范围（为什么不是所有分支都挂）：
  - comparison   每个对象的检索结果进合成前过滤（本模块主要受益方）
  - general      在 general_tools 节点对 semantic_search 的 ToolMessage
                 结果过滤（图节点层，不进工具——工具内不藏 LLM 调用，
                 eval 与工具单测保持确定性）
  - recommendation 保留其专用 rec_filter：它做特征匹配 + 排除 + 理由生成，
                 不止相关性判断，通用过滤器不替代
  - timeline     无向量检索（结构化 group_by_axis），不适用

纪律（与 reranker 相同）：过滤是增强不是依赖——任何失败返回原列表；
只删不重排（顺序已由 RRF + 精排决定）；永不返回空证据（LLM 过度过滤时
按原序兜底补足 min_keep 条，避免合成端拿到"无检索结果"）。
开关：RELEVANCE_FILTER_ENABLED=0 或参数 enabled=False 关闭（A/B 与排障用）。
"""

from __future__ import annotations

import os

from src.agent.prompts import RELEVANCE_FILTER_PROMPT
from src.utils.logging_config import get_logger, log_event

logger = get_logger("retrieval.relevance")


def _filter_enabled(override: bool | None) -> bool:
    """过滤开关：显式参数 > env RELEVANCE_FILTER_ENABLED（默认开）。"""
    if override is not None:
        return override
    return os.getenv("RELEVANCE_FILTER_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _candidate_line(i: int, item: dict) -> str:
    title = str(item.get("title") or "(无标题)")
    snippet = str(item.get("description_snippet") or item.get("content") or "")
    return f"[{i}] {title}: {snippet[:200]}"


def llm_relevance_filter(
    query: str,
    items: list[dict],
    *,
    min_keep: int = 2,
    max_candidates: int = 12,
    llm=None,
    enabled: bool | None = None,
) -> list[dict]:
    """
    用 LLM 判断候选与 query 的相关性，剔除不相关项。

    items: semantic_search 返回形状的画作/文档片段字典（需有 title 与
           description_snippet 键，缺失时退化为 content 截断）。
    min_keep: 兜底保留条数——LLM 全否定/输出残缺时按原序补足，永不返回空。
    max_candidates: 超过此数的尾部候选不参与过滤、原样透传（控制调用体量）。
    llm: 可注入的模型实例（测试用）；None 时用 get_deterministic_llm()。
    返回：原列表的子序列（保持原相对顺序）；未过滤/失败时返回原列表本身。
    """
    if not _filter_enabled(enabled):
        return items
    if len(items) <= min_keep:
        return items  # 无可过滤，省一次 LLM 调用

    candidates = items[:max_candidates]
    rest = items[max_candidates:]
    numbered = "\n".join(_candidate_line(i, c) for i, c in enumerate(candidates))
    prompt = RELEVANCE_FILTER_PROMPT.format(query=query, candidates=numbered)

    try:
        from src.agent.nodes.common import parse_json  # 延迟导入，避免模块级重依赖

        model = llm
        if model is None:
            from src.utils.llm import get_deterministic_llm

            model = get_deterministic_llm()
        raw = model.invoke(prompt).content
        parsed = parse_json(raw)
    except Exception as e:  # noqa: BLE001 — 任何失败都降级原列表
        logger.warning("[relevance] 过滤调用失败，保留原列表：%s", e)
        return items

    if not isinstance(parsed, list):
        logger.warning("[relevance] 输出非数组，保留原列表")
        return items

    keep = {
        i
        for i in parsed
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(candidates)
    }
    # 兜底：LLM 过度过滤（全否定/有效编号不足）时按原序补足 min_keep
    for i in range(len(candidates)):
        if len(keep) >= min_keep:
            break
        keep.add(i)

    filtered = [c for i, c in enumerate(candidates) if i in keep]
    log_event(
        logger, "relevance_filter",
        in_count=len(items), kept=len(filtered), dropped=len(items) - len(filtered) - len(rest),
    )
    return filtered + rest
