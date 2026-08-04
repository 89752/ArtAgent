"""
精排（Stage 4）：RRF 粗排后的第二次排序——双端点 + 免费额度接力。

定位：HybridRetriever 的扇出 + RRF 融合解决"多源召回"，精排解决
"top_k 内谁更相关"——BGE 向量相似度是粗粒度信号（整库单 embedding
模型），reranker 用交叉注意力对 query×候选逐对打分，把粗排 top 40
重排后再取 top_k。这是 Stage 4 Recall@5 提升的主要杠杆。

端点与模型（2026-08-01 实测，修正"gte-rerank-v2 已下线"的旧记载）：
  兼容端点 POST /compatible-api/v1/reranks
    · qwen3-rerank（主）：唯一兼容端点文本精排；单次 ≤500 文档、
      单文档 ≤4000 token，支持 instruct
  原生端点 POST /api/v1/services/rerank/text-rerank/text-rerank
    · gte-rerank-v2（默认后备）：报文不同（input/parameters 包裹），
      未下线、免费额度与 qwen3-rerank 各自独立（控制台实测满额 100 万）
    · qwen3-vl-rerank：多模态（文本也兼容），Phase 2 整页图精排预留
  切换只改 env：RERANK_MODEL / RERANK_FALLBACK_MODEL；模型按名字自动
  路由到对应端点（_NATIVE_MODELS 判定），无需改代码。

工程纪律（§1.5 事故教训）：显式超时 + 有限重试；主模型失败自动接力
后备模型，双模型都失败才返回 None 由调用方降级回 RRF 原序——精排是
增强不是依赖，检索绝不因 reranker 挂掉而失败。
成本控制：候选 ≤ top_k 时调用方应跳过（无排可重）。

额度耗尽方案（2026-08-03）：
  1. RERANK_ENABLED=0            → 关闭精排，粗排原序（Recall@5 85%，零成本）
  2. RERANK_BACKEND=local        → 本地精排（Jina Reranker v3.5 listwise 默认，
                                   bge-reranker-v2-m3 cross-encoder 兼容），
                                   一次性下载后永久离线，不再依赖任何额度
  3. RERANK_API_URL/API_KEY      → 指向其他 OpenAI 兼容 rerank 服务
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv

from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("retrieval.reranker")

COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
NATIVE_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

# 模型 ID 槽位（与 DEEPSEEK_MODEL 同原则）：改 env 即换模型，无需改代码。
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
RERANK_FALLBACK_MODEL = os.getenv("RERANK_FALLBACK_MODEL", "gte-rerank-v2")
# 精排后端：api（DashScope/兼容服务）或 local（本地 cross-encoder，零额度）
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "api").strip().lower()
RERANK_LOCAL_MODEL = os.getenv("RERANK_LOCAL_MODEL", "jinaai/jina-reranker-v3.5")
# 本地精排设备：cpu（默认，避免与 Ollama 抢显存）| cuda（快，但需空闲显存 ~1.5GB）
RERANK_LOCAL_DEVICE = os.getenv("RERANK_LOCAL_DEVICE", "cpu").strip().lower()
# 本地精排单文档字符上限（Jina listwise 对总序列长度敏感，调小可大幅提速；
# 描述 p99≈2472 字符，1200 截断对重排质量影响有限）
RERANK_LOCAL_DOC_CHAR_LIMIT = int(os.getenv("RERANK_LOCAL_DOC_CHAR_LIMIT", "3000"))
# 兼容端点可整体切换（额度耗尽时指向其他 OpenAI 兼容 rerank 服务）；
# API Key 独立可配，默认沿用 DEEPSEEK_API_KEY。
RERANK_API_URL = os.getenv("RERANK_API_URL", COMPAT_URL)
# DashScope 兼容端点接受 return_documents；第三方端点（如 Jina）可能拒绝
# 未知字段，非 DashScope 时省略该参数。
_USE_RETURN_DOCUMENTS = RERANK_API_URL.startswith("https://dashscope.aliyuncs.com")

# 走原生端点（报文不同）的模型集合；其余按兼容端点处理
_NATIVE_MODELS = {"gte-rerank-v2", "qwen3-vl-rerank"}

REQUEST_TIMEOUT = 30  # 秒（§1.5 教训：任何外部 API 必须显式超时）
MAX_RETRIES = 2  # 每个模型首次之后的重试次数
DOC_CHAR_LIMIT = 3000  # 单文档 ≤4000 token 的字符级保守截断


def _active_api_key() -> str:
    """rerank 专用 key 优先，默认沿用 DEEPSEEK_API_KEY（调用时动态读取）。"""
    return (
        os.getenv("RERANK_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
    )


def rerank_available() -> bool:
    """本地后端恒可用；API 后端依赖 rerank key（默认 DEEPSEEK_API_KEY）。"""
    if RERANK_BACKEND == "local":
        return True
    return bool(_active_api_key())


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_active_api_key()}",
        "Content-Type": "application/json",
    }


def _call_compat(
    model: str, query: str, documents: list[str], top_n: int, instruct: str | None
) -> list[tuple[int, float]]:
    """兼容端点（OpenAI 风格报文）：qwen3-rerank 专用路径。"""
    payload: dict = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    if _USE_RETURN_DOCUMENTS:
        payload["return_documents"] = False
        if instruct:  # instruct 是 DashScope qwen3-rerank 专属字段
            payload["instruct"] = instruct
    resp = requests.post(RERANK_API_URL, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return [(int(r["index"]), float(r.get("relevance_score", 0.0))) for r in results]


def _call_native(
    model: str, query: str, documents: list[str], top_n: int, instruct: str | None
) -> list[tuple[int, float]]:
    """原生端点（input/parameters 报文）：gte-rerank-v2 / qwen3-vl-rerank。

    不支持 instruct（原生参数面不含），调用方传入时静默忽略。
    """
    payload = {
        "model": model,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": min(top_n, len(documents)), "return_documents": False},
    }
    resp = requests.post(NATIVE_URL, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    results = (resp.json().get("output") or {}).get("results") or []
    return [(int(r["index"]), float(r.get("relevance_score", 0.0))) for r in results]


def _endpoint_for(model: str):
    return _call_native if model in _NATIVE_MODELS else _call_compat


def _call_with_retries(
    model: str, query: str, documents: list[str], top_n: int, instruct: str | None
) -> list[tuple[int, float]] | None:
    """单模型带重试调用；耗尽返回 None。"""
    fn = _endpoint_for(model)
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            ranked = fn(model, query, documents, top_n, instruct)
            ranked.sort(key=lambda t: -t[1])
            log_event(logger, "rerank", model=model,
                      docs=len(documents), top_n=min(top_n, len(documents)), attempt=attempt)
            return ranked
        except Exception as e:  # noqa: BLE001 — 任何失败都走接力/降级
            last_err = e
            logger.warning("[rerank] %s 第 %d 次调用失败：%s", model, attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(min(2**attempt, 5))
    logger.warning("[rerank] %s 重试耗尽：%s", model, last_err)
    return None


@lru_cache(maxsize=1)
def _get_local_reranker(model_name: str):
    """加载本地精排模型（一次性下载，之后永久离线）。
    Jina v3.5（默认）：0.6B listwise，一次前向排整批候选（BEIR 63.20）；
    bge-reranker-v2-m3（旧兼容）：cross-encoder 两两打分。"""
    if "jina" in model_name.lower():
        from transformers import AutoModel

        logger.info("[rerank] 加载本地 listwise 精排模型 %s（首次需下载，约 1–2GB）", model_name)
        model = AutoModel.from_pretrained(model_name, dtype="auto", trust_remote_code=True)
        model.eval()
        if RERANK_LOCAL_DEVICE == "cuda":
            import torch

            if torch.cuda.is_available():
                model = model.half().to("cuda")
        return model
    from sentence_transformers import CrossEncoder

    logger.info("[rerank] 加载本地 cross-encoder 精排模型 %s（首次需下载，约 1–2GB）", model_name)
    return CrossEncoder(model_name)


def _rerank_local(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[tuple[int, float]] | None:
    """本地精排：零 API 额度，失败降级粗排原序。"""
    try:
        model = _get_local_reranker(RERANK_LOCAL_MODEL)
        if "jina" in RERANK_LOCAL_MODEL.lower():
            # Jina listwise 接口：返回已按分数降序的 [{index, relevance_score, document}]
            results = model.rerank(query, documents, top_n=top_n)
            ranked = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        else:
            pairs = [[query, d] for d in documents]
            scores = model.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip(range(len(documents)), scores.tolist()),
                key=lambda t: -float(t[1]),
            )[:top_n]
        log_event(logger, "rerank", model=f"local:{RERANK_LOCAL_MODEL}",
                  docs=len(documents), top_n=top_n)
        return ranked
    except Exception as e:  # noqa: BLE001 —— 本地精排失败不阻断检索
        logger.warning("[rerank] 本地精排失败，降级原序：%s", e)
        return None


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    instruct: str | None = None,
) -> list[tuple[int, float]] | None:
    """
    对候选文档按与 query 的相关性精排。

    返回 [(原始下标, 相关性分数)] 按分数降序；主模型失败自动接力后备模型，
    双模型均失败返回 None（调用方降级回粗排原序）。空候选返回空列表。
    """
    if not documents:
        return []
    if RERANK_BACKEND == "local":
        docs = [(d or "")[:RERANK_LOCAL_DOC_CHAR_LIMIT] for d in documents]
        return _rerank_local(query, docs, min(top_n or len(docs), len(docs)))
    if not rerank_available():
        logger.warning("[rerank] RERANK_API_KEY 未配置，降级原序")
        return None

    docs = [(d or "")[:DOC_CHAR_LIMIT] for d in documents]
    n = min(top_n or len(docs), len(docs))

    ranked = _call_with_retries(RERANK_MODEL, query, docs, n, instruct)
    if ranked is not None:
        return ranked
    if RERANK_FALLBACK_MODEL and RERANK_FALLBACK_MODEL != RERANK_MODEL:
        logger.warning("[rerank] 主模型 %s 不可用，接力后备 %s",
                       RERANK_MODEL, RERANK_FALLBACK_MODEL)
        ranked = _call_with_retries(RERANK_FALLBACK_MODEL, query, docs, n, instruct)
        if ranked is not None:
            return ranked
    logger.warning("[rerank] 主备模型均失败，降级原序")
    return None
