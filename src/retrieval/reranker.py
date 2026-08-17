"""精排：RRF 粗排后的第二次排序——Jina Reranker v3.5 API。

定位：HybridRetriever 的扇出 + RRF 融合解决"多源召回"，精排解决
"top_k 内谁更相关"——BGE 向量相似度是粗粒度信号（整库单 embedding
模型），reranker 用交叉注意力对 query×候选逐对打分，把粗排 top 40
重排后再取 top_k。

端点：https://api.jina.ai/v1/rerank，模型默认 jina-reranker-v3.5
（可用 RERANK_MODEL 覆盖）；API Key 通过 RERANK_API_KEY 配置，
未配置时精排降级回粗排原序。本机无法直连 Jina 时，可配
RERANK_PROXY（如 http://127.0.0.1:7890）只让 Jina 请求走代理，
不影响 LLM 等其他调用。

工程纪律：显式超时 + 有限重试；失败返回 None 由调用方降级回 RRF 原序
——精排是增强不是依赖，检索绝不因 reranker 挂掉而失败。
成本控制：候选 ≤ top_k 时调用方应跳过（无排可重）。
"""

from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("retrieval.reranker")

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
# 模型 ID 槽位（与 LLM_MODEL 同原则）：改 env 即换模型，无需改代码
RERANK_MODEL = os.getenv("RERANK_MODEL", "jina-reranker-v3.5")

REQUEST_TIMEOUT = 30  # 秒：外部 API 必须显式超时
MAX_RETRIES = 2       # 首次之后的额外重试次数
DOC_CHAR_LIMIT = 3000  # 单文档字符级保守截断


def _active_api_key() -> str:
    """Jina rerank API Key（RERANK_API_KEY，调用时动态读取）。"""
    return os.getenv("RERANK_API_KEY", "").strip()


def _active_proxy() -> str:
    """Jina 专用代理（RERANK_PROXY，调用时动态读取）；空串表示直连。"""
    return os.getenv("RERANK_PROXY", "").strip()


def rerank_available() -> bool:
    """Jina API 依赖 RERANK_API_KEY。"""
    return bool(_active_api_key())


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_active_api_key()}",
        "Content-Type": "application/json",
    }


def _call_jina(
    query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]]:
    """调用 Jina Reranker v3.5 API，返回 [(原始下标, 分数)]。"""
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    proxy = _active_proxy()
    resp = requests.post(
        JINA_RERANK_URL,
        headers=_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
        proxies={"http": proxy, "https": proxy} if proxy else None,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return [(int(r["index"]), float(r.get("relevance_score", 0.0))) for r in results]


def _call_with_retries(
    query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]] | None:
    """带重试调用；失败返回 None（调用方降级粗排原序）。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            ranked = _call_jina(query, documents, top_n)
            ranked.sort(key=lambda t: -t[1])
            log_event(
                logger, "rerank", model=RERANK_MODEL,
                docs=len(documents), top_n=min(top_n, len(documents)),
                attempt=attempt,
            )
            return ranked
        except Exception as e:  # noqa: BLE001 — 任何失败都走降级
            last_err = e
            logger.warning("[rerank] 第 %d 次调用失败：%s", attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(min(2**attempt, 5))
    logger.warning("[rerank] 重试耗尽：%s", last_err)
    return None


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[tuple[int, float]] | None:
    """对候选文档按与 query 的相关性精排。

    返回 [(原始下标, 相关性分数)] 按分数降序；API 失败返回 None
    （调用方降级回粗排原序）。空候选返回空列表。
    """
    if not documents:
        return []
    if not rerank_available():
        logger.warning("[rerank] RERANK_API_KEY 未配置，降级原序")
        return None

    docs = [(d or "")[:DOC_CHAR_LIMIT] for d in documents]
    n = min(top_n or len(docs), len(docs))
    return _call_with_retries(query, docs, n)
