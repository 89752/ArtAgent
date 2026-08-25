"""ArtAgent 统一配置加载器

优先级：环境变量 > config.yaml > 内置默认值。
- config.yaml 里的 `$ENV_NAME` 会被替换成环境变量值（如 $LLM_API_KEY）
- 环境变量可通过 _ENV_MAP 覆盖对应配置项（如 TOOL_TIMEOUT_SEC → governance.tool_timeout_sec）
- 所有读取都走带类型和边界的访问器，非法值回落默认值
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("ARTAGENT_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml")))

# 环境变量 → 配置点路径（点分）。
# 注意：memory.* / project.* / logging.* / ingestion.* / cost.* 以及
# retrieval 下的 RERANK_* / LEXICAL_* 等开关由各模块直接读取同名环境变量，
# 不走 config.yaml，故不在此注册；retrieval.pdf_image_embed_* 是例外
# （经 config 消费，用于 PDF 整页图嵌入提供商）。
_ENV_MAP: dict[str, str] = {
    "LLM_API_KEY": "models.llm_api_key",
    "LLM_BASE_URL": "models.llm_base_url",
    "LLM_MODEL": "models.llm_model",
    "VISION_API_KEY": "models.vision_api_key",
    "VISION_BASE_URL": "models.vision_base_url",
    "VISION_MODEL": "models.vision_model",
    "JUDGE_MODEL": "models.judge_model",
    "JUDGE_API_KEY": "models.judge_api_key",
    "JUDGE_BASE_URL": "models.judge_base_url",
    "CHEAP_MODEL": "models.cheap_model",
    "CHEAP_API_KEY": "models.cheap_api_key",
    "CHEAP_BASE_URL": "models.cheap_base_url",
    "REASONING_MODEL": "models.reasoning_model",
    "REASONING_API_KEY": "models.reasoning_api_key",
    "REASONING_BASE_URL": "models.reasoning_base_url",
    "MODEL_ROUTING_ENABLED": "models.routing_enabled",
    "AGENTIC_RAG_ENABLED": "retrieval.agentic_enabled",
    "AGENTIC_RAG_MIN_EVIDENCE": "retrieval.agentic_min_evidence",
    "PDF_IMAGE_EMBED_PROVIDER": "retrieval.pdf_image_embed_provider",
    "PDF_IMAGE_EMBED_MODEL": "retrieval.pdf_image_embed_model",
    "PDF_IMAGE_EMBED_API_KEY": "retrieval.pdf_image_embed_api_key",
    "PDF_IMAGE_EMBED_BASE_URL": "retrieval.pdf_image_embed_base_url",
    "TOOL_TIMEOUT_SEC": "governance.tool_timeout_sec",
    "TOOL_RETRIES": "governance.tool_retries",
    "TOOL_OUTPUT_MAX_CHARS": "governance.tool_output_max_chars",
    "SUBAGENT_MAX_CONCURRENT": "subagents.max_concurrent",
    "SUBAGENT_MAX_TOTAL_PER_RUN": "subagents.max_total_per_run",
    "SUBAGENT_TIMEOUT_SEC": "subagents.timeout_sec",
    "SUBAGENT_MAX_TURNS": "subagents.max_turns",
}


def _defaults() -> dict[str, Any]:
    """内置默认值（与 config.yaml 保持一致，保证无文件也能跑）。"""
    return {
        "models": {
            "llm_api_key": None,
            "llm_base_url": None,
            "llm_model": None,
            "vision_api_key": None,
            "vision_base_url": None,
            "vision_model": None,
            "judge_model": None,
            "judge_api_key": None,
            "judge_base_url": None,
            "cheap_model": None,
            "cheap_api_key": None,
            "cheap_base_url": None,
            "reasoning_model": None,
            "reasoning_api_key": None,
            "reasoning_base_url": None,
            "routing_enabled": True,
            "request_timeout_sec": 180,
            "max_retries": 2,
        },
        "retrieval": {
            "pdf_image_embed_provider": "dashscope",
            "pdf_image_embed_model": "tongyi-embedding-vision-plus",
            "pdf_image_embed_api_key": None,
            "pdf_image_embed_base_url": None,
            "agentic_enabled": True,
            "agentic_min_evidence": 3,
        },
        "governance": {
            "tool_timeout_sec": 60,
            "tool_retries": 1,
            "tool_output_max_chars": 2000,
        },
        "subagents": {
            "max_concurrent": 3,
            "max_total_per_run": 6,
            "timeout_sec": 300,
            "max_turns": 15,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 里的值覆盖 base；None 表示"未设置"，保留 base。"""
    out = dict(base)
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_env_refs(value: Any) -> Any:
    """把字符串里的 $ENV_NAME 替换成环境变量值；未设置则替换为空。

    缺失的环境变量绝不能保留字面量（如 "$LLM_API_KEY"），否则会被当成
    有值传给下游，造成"假密钥静默运行"。整体替换为空时返回 None。
    """
    if not isinstance(value, str):
        return value

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        return os.getenv(name, "")

    resolved = re.sub(r"\$([A-Z_][A-Z0-9_]*)", _sub, value)
    return resolved if resolved else None


def _set_path(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _get_path(data: dict, dotted: str, default: Any = None) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@lru_cache(maxsize=1)
def _base_config() -> dict[str, Any]:
    """只缓存"默认值 + config.yaml"（不含环境变量）。

    环境变量是最高优先级且必须实时生效（测试、运行时切换都要立刻可见），
    所以 env 覆盖放在 load_config 里每次计算，不缓存。
    """
    config = _defaults()
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as fh:
            file_data = yaml.safe_load(fh) or {}
        config = _deep_merge(config, file_data)
    return config


def load_config() -> dict[str, Any]:
    """加载并合并配置：默认值 <- config.yaml <- 环境变量（实时）。"""
    config = deepcopy(_base_config())
    # 环境变量覆盖（环境变量存在才覆盖，避免空字符串误关功能）
    for env_name, dotted in _ENV_MAP.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        _set_path(config, dotted, _resolve_env_refs(raw))

    # 解析所有 $ENV 引用
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return _resolve_env_refs(node)

    return walk(config)


def reload_config() -> dict[str, Any]:
    """测试用：强制重载。"""
    _base_config.cache_clear()
    return load_config()


def get(dotted: str, default: Any = None) -> Any:
    """点分路径取值，如 get("models.llm_model")。"""
    return _get_path(load_config(), dotted, default)


def get_int(
    dotted: str, default: int, lo: int | None = None, hi: int | None = None
) -> int:
    raw = get(dotted, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def get_float(dotted: str, default: float, lo: float | None = None) -> float:
    raw = get(dotted, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if lo is not None:
        value = max(lo, value)
    return value


def get_bool(dotted: str, default: bool = False) -> bool:
    """Read a boolean config value without treating arbitrary strings as true."""
    raw = get(dotted, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return bool(default)
