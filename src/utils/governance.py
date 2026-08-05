"""工具执行治理：统一超时、重试与失败包装。

graph 内所有工具调用经 governed_invoke 执行：超时（默认 60s，env TOOL_TIMEOUT_SEC）、
瞬时失败重试（默认 1 次，env TOOL_RETRIES）。最终失败返回结构化错误 JSON，
不抛异常中断整轮推理。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable


class ToolTimeout(RuntimeError):
    pass


def run_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    """在守护线程执行 fn；超时抛 ToolTimeout。"""
    box: dict[str, Any] = {"result": None, "error": None}

    def _run() -> None:
        try:
            box["result"] = fn()
        except Exception as e:  # noqa: BLE001 —— 原样装箱，统一在调用侧处理
            box["error"] = e

    worker = threading.Thread(target=_run, daemon=True, name="tool-call")
    worker.start()
    worker.join(timeout=timeout_sec)
    if worker.is_alive():
        raise ToolTimeout(f"工具执行超过 {timeout_sec}s")
    if box["error"] is not None:
        raise box["error"]
    return box["result"]


def _timeout_sec() -> float:
    try:
        return max(1.0, float(os.getenv("TOOL_TIMEOUT_SEC", "60")))
    except ValueError:
        return 60.0


def _retries() -> int:
    try:
        return max(0, int(os.getenv("TOOL_RETRIES", "1")))
    except ValueError:
        return 1


def _output_limit() -> int:
    """单条工具返回的 JSON 上限（对齐 skills runner 的 2000 字符）。"""
    try:
        return max(200, int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "2000")))
    except ValueError:
        return 2000


def _truncate_payload(payload: Any, limit: int) -> Any:
    """递归压缩超长工具返回，尽量保留结构化 JSON 形状。

    策略：字符串超限截断并加"[截断]"标记；列表按预算保留前 N 项并在尾部
    追加 {truncated: true, dropped: n}；字典逐键递归。
    """
    budget = {"left": limit}

    def walk(value: Any) -> Any:
        if budget["left"] <= 0:
            return "...[截断]"
        if isinstance(value, str):
            if len(value) > budget["left"]:
                keep = max(0, budget["left"] - 12)
                out = value[:keep] + "...[截断]"
                budget["left"] = 0
                return out
            budget["left"] -= len(value)
            return value
        if isinstance(value, dict):
            out: dict = {}
            for k, v in value.items():
                if budget["left"] <= 0:
                    break
                out[k] = walk(v)
            return out
        if isinstance(value, list):
            out: list = []
            total = len(value)
            for i, item in enumerate(value):
                if budget["left"] <= 0:
                    out.append({"truncated": True, "dropped": total - i})
                    break
                out.append(walk(item))
            return out
        s = json.dumps(value, ensure_ascii=False, default=str)
        budget["left"] -= len(s)
        return value

    return walk(payload)


def governed_invoke(tool: Any, args: dict) -> str:
    """带超时与重试地调用工具；最终失败返回 JSON 错误串，不抛异常。"""
    timeout = _timeout_sec()
    retries = _retries()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            out = run_with_timeout(
                lambda t=tool, a=args: t.invoke(a), timeout
            )
            if isinstance(out, str):
                return out[: _output_limit()]
            payload = json.dumps(out, ensure_ascii=False, default=str)
            if len(payload) > _output_limit():
                out = _truncate_payload(out, _output_limit())
            return json.dumps(out, ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return json.dumps(
        {
            "status": "TOOL_ERROR",
            "tool": getattr(tool, "name", "?"),
            "message": f"{type(last_error).__name__}: {last_error}"[:500],
        },
        ensure_ascii=False,
    )
