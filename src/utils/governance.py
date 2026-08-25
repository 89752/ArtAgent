"""工具执行治理：统一超时、重试与失败包装。

graph 内所有工具调用经 governed_invoke 执行：超时、瞬时失败重试、输出截断。
参数来源：config.yaml（governance.*），环境变量 TOOL_TIMEOUT_SEC 等仍可覆盖。
最终失败返回结构化错误 JSON，不抛异常中断整轮推理。
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.utils.config import get_float, get_int
from src.utils.logging_config import get_logger

logger = get_logger("utils.governance")


class ToolTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    """Execution policy attached to a registered tool.

    Defaults are deliberately conservative for unknown tools: they may run in
    the main agent, but are not retried after a timeout unless explicitly known
    to be read-only and idempotent.
    """

    risk: str = "medium"
    read_only: bool = False
    idempotent: bool = False
    timeout_sec: float | None = None
    retries: int | None = None
    requires_confirmation: bool = False
    max_output_chars: int | None = None
    allowed_for_subagent: bool = False


_READ_ONLY_TOOLS = {
    "semantic_search", "exact_lookup", "query_painter_knowledge", "image_lookup",
    "read_page_image", "web_search", "recall", "list_collections", "get_collection",
    "list_preferences", "color_analysis", "aggregate_stats", "compare_images",
    "museum_search", "wiki_lookup", "read_user_image", "analyze_user_artwork",
}
_SUBAGENT_TOOLS = {
    "semantic_search", "exact_lookup", "query_painter_knowledge", "image_lookup",
    "web_search", "wiki_lookup", "museum_search", "aggregate_stats",
    "color_analysis", "compare_images",
}
_CONFIRMATION_REQUIRED = {"forget", "delete_collection"}


def tool_spec(tool: Any) -> ToolSpec:
    """Return the central policy for a tool without duplicating registrations."""
    name = str(getattr(tool, "name", ""))
    if name in _READ_ONLY_TOOLS:
        return ToolSpec(
            risk="low", read_only=True, idempotent=True,
            allowed_for_subagent=name in _SUBAGENT_TOOLS,
        )
    # Persisting or mutating operations must never be replayed automatically
    # once a thread timeout leaves their actual remote state unknown.
    return ToolSpec(
        risk="high", read_only=False, idempotent=False,
        requires_confirmation=name in _CONFIRMATION_REQUIRED,
    )


def run_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    """在守护线程执行 fn；超时抛 ToolTimeout。

    Copy ContextVars into the worker.  Tool functions use this for the active
    user identity, and a plain ``threading.Thread`` would otherwise silently
    fall back to the process-wide default account.
    """
    box: dict[str, Any] = {"result": None, "error": None}
    caller_context = contextvars.copy_context()

    def _run() -> None:
        try:
            box["result"] = caller_context.run(fn)
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
    return get_float("governance.tool_timeout_sec", 60.0, lo=1.0)


def _retries() -> int:
    return get_int("governance.tool_retries", 1, lo=0)


def _output_limit() -> int:
    """单条工具返回的 JSON 上限（对齐 skills runner 的 2000 字符）。"""
    return get_int("governance.tool_output_max_chars", 2000, lo=200)


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


def _tool_schema(tool: Any) -> dict:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return {}
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
        return schema if isinstance(schema, dict) else {}
    if hasattr(schema, "schema"):
        schema = schema.schema()
        return schema if isinstance(schema, dict) else {}
    return schema if isinstance(schema, dict) else {}


class ToolExecutor:
    """Single tool-execution entry point for Agent, Skill and Subagent paths."""

    @staticmethod
    def execute(
        tool: Any, args: dict, *, context: str = "main", user_id: str = ""
    ) -> str:
        return _execute(tool, args, context=context, user_id=user_id)


def governed_invoke(
    tool: Any, args: dict, *, context: str = "main", user_id: str = ""
) -> str:
    """Validate, authorize, execute and normalize a tool result.

    Kept under its original name as the compatibility façade for all callers.
    It now is the single execution kernel used by main Agent, skills and
    subagents.
    """
    return ToolExecutor.execute(tool, args, context=context, user_id=user_id)


def _execute(tool: Any, args: dict, *, context: str = "main", user_id: str = "") -> str:
    from src.tools.guard import ToolDecision, validate_args

    spec = tool_spec(tool)
    name = str(getattr(tool, "name", "?"))
    if context == "subagent" and not spec.allowed_for_subagent:
        return json.dumps({"status": "TOOL_FORBIDDEN", "tool": name}, ensure_ascii=False)
    if spec.requires_confirmation and context != "confirmed":
        return json.dumps({"status": "CONFIRMATION_REQUIRED", "tool": name}, ensure_ascii=False)
    schema = _tool_schema(tool)
    # A few legacy/testing adapters have no machine-readable schema. They are
    # still executed through timeout/policy/output controls; registered tools
    # always expose a schema and therefore receive strict argument validation.
    decision = validate_args(schema, args) if schema else ToolDecision("SUCCESS", params=args)
    if decision.status != "SUCCESS":
        return json.dumps(
            {"status": decision.status, "tool": name, "missing": decision.missing, "errors": decision.errors},
            ensure_ascii=False,
        )
    args = decision.params
    timeout = spec.timeout_sec or _timeout_sec()
    retries = spec.retries if spec.retries is not None else _retries()
    # A timeout for a non-idempotent write means the work may still be running
    # in the daemon thread. Never replay it automatically.
    if not (spec.read_only and spec.idempotent):
        retries = 0
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            def invoke_with_identity(t=tool, a=args):
                if not user_id:
                    return t.invoke(a)
                from src.memory.memory_items import clear_active_user_id, set_active_user_id

                set_active_user_id(user_id)
                try:
                    return t.invoke(a)
                finally:
                    clear_active_user_id()

            out = run_with_timeout(invoke_with_identity, timeout)
            output_limit = spec.max_output_chars or _output_limit()
            if isinstance(out, str):
                return out[:output_limit]
            payload = json.dumps(out, ensure_ascii=False, default=str)
            if len(payload) > output_limit:
                out = _truncate_payload(out, output_limit)
            return json.dumps(out, ensure_ascii=False, default=str)
        except ToolTimeout as e:
            logger.warning(
                "[govern] 工具 %s 超时（后台线程可能仍在执行，副作用可能已生效）",
                name,
            )
            last_error = e
            if not (spec.read_only and spec.idempotent):
                return json.dumps(
                    {"status": "UNKNOWN_EXECUTION_STATE", "tool": name,
                     "message": "写操作超时，执行状态未知，禁止自动重试"},
                    ensure_ascii=False,
                )
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return json.dumps(
        {
            "status": "TOOL_ERROR",
            "tool": name,
            "message": f"{type(last_error).__name__}: {last_error}"[:500],
        },
        ensure_ascii=False,
    )
