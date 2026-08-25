"""
检索结果相关性校正：通用轻量 LLM 过滤步骤。

定位：向量检索（即便经
Jina Reranker v3.5 精排）仍会混入"形似而答非所问"的噪声候选；确定性管线把
证据直接拼进合成 prompt，没有自救机会。本步骤在 HybridRetriever 之后、
证据消费之前，用一次轻量确定性 LLM 调用剔掉不相关候选。

适用范围（为什么不是所有分支都挂）：
  - comparison   每个对象的检索结果进合成前过滤（本模块主要受益方）
  - general      在 general_tools 节点对 semantic_search 的 ToolMessage
                 结果过滤（图节点层，不进工具——工具内不藏 LLM 调用，
                 eval 与工具单测保持确定性）
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


def _is_image_result(item: dict) -> bool:
    """整页图结果不能靠文本 snippet 判断相关性，需透传给 read_page_image。"""
    return item.get("source") == "user_pdf_image"


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

    注意：source="user_pdf_image" 的整页图结果只能由视觉模型真正读取内容，
    文本 snippet 只是占位标题，因此不参与 LLM 相关性判断、始终保留。
    """
    if not _filter_enabled(enabled):
        return items

    # 整页图结果始终保留，只对其余候选做文本相关性过滤
    image_indices = {i for i, it in enumerate(items) if _is_image_result(it)}
    text_indices = [i for i in range(len(items)) if i not in image_indices]
    text_items = [items[i] for i in text_indices]

    if len(text_items) <= min_keep:
        return items  # 文本候选无可过滤，省一次 LLM 调用

    candidates = text_items[:max_candidates]
    rest = text_items[max_candidates:]
    numbered = "\n".join(_candidate_line(i, c) for i, c in enumerate(candidates))
    prompt = RELEVANCE_FILTER_PROMPT.format(query=query, candidates=numbered)

    try:
        from src.utils.json_utils import parse_json  # 延迟导入，避免模块级重依赖

        model = llm
        if model is None:
            from src.utils.llm import get_cheap_llm

            model = get_cheap_llm()
        raw = model.invoke(prompt).content
        parsed = parse_json(raw)
    except Exception as e:  # noqa: BLE001 — 任何失败都降级原列表
        logger.warning("[relevance] 过滤调用失败，保留原列表：%s", e)
        return items

    if not isinstance(parsed, list):
        logger.warning("[relevance] 输出非数组，保留原列表")
        return items

    keep_in_text = {
        i
        for i in parsed
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(candidates)
    }
    # 兜底：LLM 过度过滤（全否定/有效编号不足）时按原序补足 min_keep
    for i in range(len(candidates)):
        if len(keep_in_text) >= min_keep:
            break
        keep_in_text.add(i)

    # 把在 text_items 中的下标映射回原列表下标
    kept_original_indices = {text_indices[i] for i in keep_in_text}
    # 尾部未参与过滤的文本候选原样保留
    kept_original_indices.update(text_indices[max_candidates:])

    result = [items[i] for i in range(len(items)) if i in image_indices or i in kept_original_indices]

    log_event(
        logger, "relevance_filter",
        in_count=len(items), kept=len(result), dropped=len(items) - len(result),
    )
    return result
