"""
qwen3-rerank 精排（Stage 4）：RRF 粗排后的第二次排序。

定位：HybridRetriever 的扇出 + RRF 融合解决"多源召回"，精排解决
"top_k 内谁更相关"——BGE 向量相似度是粗粒度信号（整库单 embedding
模型），reranker 用交叉注意力对 query×候选逐对打分，把粗排 top 15–20
重排后再取 top_k。这是 Stage 4 Recall@5 提升的主要杠杆。

端点（实施方案 §5 已核实）：DashScope OpenAI 兼容
  POST https://dashscope.aliyuncs.com/compatible-api/v1/reranks
  model=qwen3-rerank，单次 ≤500 文档、单文档 ≤4000 token，支持 instruct。
  gte-rerank-v2 已下线，勿用。

工程纪律（§1.5 事故教训）：显式超时 + 有限重试；任何失败都返回 None
由调用方降级回 RRF 原序——精排是增强不是依赖，检索绝不因 reranker
挂掉而失败。成本控制：候选 ≤ top_k 时调用方应跳过（无排可重）。
"""

from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("retrieval.reranker")

RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
RERANK_MODEL = "qwen3-rerank"
REQUEST_TIMEOUT = 30  # 秒（§1.5 教训：任何外部 API 必须显式超时）
MAX_RETRIES = 2  # 首次之后的重试次数
DOC_CHAR_LIMIT = 3000  # 单文档 ≤4000 token 的字符级保守截断


def rerank_available() -> bool:
    """精排依赖 DashScope key（与对话/视觉同一把 DEEPSEEK_API_KEY）。"""
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    instruct: str | None = None,
) -> list[tuple[int, float]] | None:
    """
    对候选文档按与 query 的相关性精排。

    返回 [(原始下标, 相关性分数)] 按分数降序；任何失败返回 None
    （调用方降级回粗排原序）。空候选返回空列表。
    """
    if not documents:
        return []
    if not rerank_available():
        logger.warning("[rerank] DEEPSEEK_API_KEY 未配置，降级原序")
        return None

    payload: dict = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [(d or "")[:DOC_CHAR_LIMIT] for d in documents],
        "top_n": min(top_n or len(documents), len(documents)),
        "return_documents": False,
    }
    if instruct:
        payload["instruct"] = instruct
    headers = {
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY'].strip()}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = requests.post(
                RERANK_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            ranked = [
                (int(r["index"]), float(r.get("relevance_score", 0.0)))
                for r in results
            ]
            ranked.sort(key=lambda t: -t[1])
            log_event(
                logger, "rerank",
                docs=len(documents), top_n=payload["top_n"], attempt=attempt,
            )
            return ranked
        except Exception as e:  # noqa: BLE001 — 任何失败都走降级
            last_err = e
            logger.warning("[rerank] 第 %d 次调用失败：%s", attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(min(2**attempt, 5))

    logger.warning("[rerank] 重试耗尽，降级原序：%s", last_err)
    return None
