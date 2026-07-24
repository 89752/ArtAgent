"""
结构化日志配置 —— 为多步 Agent 提供可观测性。

一个每轮要发多次 LLM 调用的 Agent，没有日志就无法回答
"走了哪个分支 / 检索到几条 / 反思结论是什么 / 哪个节点慢"。
本模块提供：
  - setup_logging(): 全局配置一次（控制台 + 可选文件），级别由环境变量控制
  - get_logger():    取带命名空间的 logger
  - log_event():     结构化 key=value 事件日志（便于 grep / 后续接入日志系统）
  - traced():        节点计时包装器，记录每个节点的耗时与产出步骤

环境变量：
  ARTAGENT_LOG_LEVEL  日志级别，默认 INFO（可设 DEBUG/WARNING）
  ARTAGENT_LOG_FILE   若设置，同时写入该文件路径
"""

import logging
import os
import time
from functools import wraps
from typing import Callable

_CONFIGURED = False


def setup_logging() -> None:
    """全局配置日志（幂等，多次调用只生效一次）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.getenv("ARTAGENT_LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.getenv("ARTAGENT_LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
    # 压低第三方库噪声
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """取一个已配置的 logger（命名空间统一挂在 artagent 下）。"""
    setup_logging()
    return logging.getLogger(f"artagent.{name}")


def log_event(logger: logging.Logger, node: str, **fields) -> None:
    """
    输出结构化事件：`[node] k1=v1 k2=v2`。
    列表/字典会被压缩成简短字符串，避免刷屏。
    """
    parts = []
    for k, v in fields.items():
        if isinstance(v, (list, tuple)):
            v = f"[{len(v)}]{list(v)[:3]}" if v else "[]"
        elif isinstance(v, str) and len(v) > 60:
            v = v[:57] + "..."
        parts.append(f"{k}={v}")
    logger.info("[%s] %s", node, " ".join(parts))


def traced(node_name: str, fn: Callable) -> Callable:
    """
    节点计时包装器：记录节点耗时（ms）与返回的 current_step。
    在 graph 构建时对每个节点函数套一层，实现统一的延迟可观测性。
    """
    logger = get_logger("trace")

    @wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        result = fn(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        step = result.get("current_step", node_name) if isinstance(result, dict) else node_name
        logger.info("[%s] done in %.0fms → %s", node_name, elapsed_ms, step)
        return result

    return wrapper
