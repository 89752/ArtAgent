"""子智能体执行器（对齐 DeerFlow 的 SubagentExecutor 思路）。

- 每个子任务在独立的 messages 上下文里跑一个迷你 ReAct 循环；
- 并发上限 / 单批总数 / 超时 / 最大轮数全部来自 config.yaml（subagents.*）；
- 禁止嵌套：子智能体只拿到只读研究白名单，没有 delegate_task；
- 结果契约：findings / sources / confidence 必填，缺字段有界补齐一次。
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.utils.config import get_float, get_int
from src.utils.llm import get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("subagents")

# 子智能体只读研究白名单：禁止 delegate_task（防嵌套）、记忆写工具（防并行写冲突）
RESEARCH_TOOL_WHITELIST = [
    "semantic_search",
    "exact_lookup",
    "query_painter_knowledge",
    "image_lookup",
    "color_analysis",
    "compare_images",
    "web_search",
    "wiki_lookup",
    "museum_search",
    "aggregate_stats",
]

REQUIRED_FIELDS = ("findings", "sources", "confidence")

_SUBAGENT_SYSTEM_PROMPT = (
    "你是 ArtAgent 的子智能体，负责独立完成派发的调研子任务。\n"
    "- 只使用提供的工具收集证据，不得编造作品细节。\n"
    '- 最终只输出 JSON：{"findings": "...", "sources": [...], '
    '"confidence": "high|medium|low"}\n'
    "- findings 用专家口吻写成可直接使用的素材，禁止出现“本地数据 / 数据集 / "
    "SemArt / 收录 N 幅 / 知识库 / 检索结果”等内部词。\n"
    "- 先判断本地证据是否足够：若样本明显偏少或缺少关键时期/代表作，"
    "优先调用 web_search / wiki_lookup / museum_search 补充，不要直接说“资料少”。\n"
    "- 联网也拿不到时，才在 findings 里用自然语言简要说明“资料有限”，不要硬凑。"
)


@dataclass
class SubagentResult:
    """单个子任务的执行结果。"""

    task_id: str
    status: str  # completed / failed / timed_out
    result: Optional[dict] = None
    error: Optional[str] = None
    tool_calls: int = 0


def _validate_result(text: str) -> Optional[dict]:
    """校验子智能体输出是否符合结果契约。"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    for key in REQUIRED_FIELDS:
        value = data.get(key)
        if key == "sources":
            if not isinstance(value, list):
                return None
        elif not str(value or "").strip():
            return None
    return data


def _run_subagent(
    task_id: str, prompt: str, max_turns: int, timeout_sec: float
) -> SubagentResult:
    """在独立上下文中运行一个子任务（迷你 ReAct）。"""
    from src.tools.registry import TOOL_BY_NAME

    tools = [TOOL_BY_NAME[n] for n in RESEARCH_TOOL_WHITELIST if n in TOOL_BY_NAME]
    if not tools:
        return SubagentResult(task_id=task_id, status="failed", error="无可用工具")

    llm = get_deterministic_llm().bind_tools(tools)
    messages: list = [
        SystemMessage(content=_SUBAGENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    deadline = time.monotonic() + timeout_sec
    corrected = False
    tool_calls = 0

    for _step in range(max_turns):
        if time.monotonic() > deadline:
            return SubagentResult(
                task_id=task_id,
                status="timed_out",
                error=f"超过 {timeout_sec:.1f}s",
                tool_calls=tool_calls,
            )
        try:
            resp = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            return SubagentResult(
                task_id=task_id, status="failed", error=f"LLM: {exc}", tool_calls=tool_calls
            )

        calls = getattr(resp, "tool_calls", None) or []
        if calls:
            messages.append(resp)
            for tc in calls:
                tool = TOOL_BY_NAME.get(tc.get("name"))
                try:
                    output = tool.invoke(tc.get("args") or {}) if tool else "未注册工具"
                except Exception as exc:  # noqa: BLE001 —— 工具失败回灌给模型
                    output = f"工具执行失败：{exc}"
                messages.append(
                    ToolMessage(
                        content=str(output)[:2000],
                        name=tc.get("name"),
                        tool_call_id=tc.get("id"),
                    )
                )
                tool_calls += 1
            continue

        text = getattr(resp, "content", "") or ""
        parsed = _validate_result(text)
        if parsed is not None:
            return SubagentResult(
                task_id=task_id, status="completed", result=parsed, tool_calls=tool_calls
            )
        if not corrected:
            messages.append(
                HumanMessage(
                    content=(
                        f"输出必须是 JSON，且包含字段 {list(REQUIRED_FIELDS)}。"
                        "请重新输出。"
                    )
                )
            )
            corrected = True
            continue
        return SubagentResult(
            task_id=task_id,
            status="completed",
            result={"findings": text[:1000], "sources": [], "confidence": "low"},
            tool_calls=tool_calls,
        )

    return SubagentResult(
        task_id=task_id, status="failed", error="步数上限", tool_calls=tool_calls
    )


def run_tasks(tasks: list[dict]) -> list[SubagentResult]:
    """并发执行多个子任务；并发/总数/超时/轮数来自 config.yaml。"""
    max_concurrent = get_int("subagents.max_concurrent", 3, lo=1)
    max_total = get_int("subagents.max_total_per_run", 6, lo=1)
    timeout_sec = get_float("subagents.timeout_sec", 300.0, lo=0.1)
    max_turns = get_int("subagents.max_turns", 15, lo=1)

    prepared = [
        {
            "task_id": t.get("task_id") or uuid.uuid4().hex[:12],
            "prompt": str(t.get("prompt") or "").strip(),
        }
        for t in (tasks or [])[:max_total]
        if str(t.get("prompt") or "").strip()
    ]
    if not prepared:
        return [SubagentResult(task_id="", status="failed", error="空任务")]

    log_event(
        logger,
        "delegate",
        tasks=len(prepared),
        max_concurrent=max_concurrent,
        timeout_sec=timeout_sec,
    )
    results: list[Optional[SubagentResult]] = [None] * len(prepared)
    deadline = time.monotonic() + timeout_sec
    pending = set()
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {
            pool.submit(_run_subagent, p["task_id"], p["prompt"], max_turns, timeout_sec): i
            for i, p in enumerate(prepared)
        }
        pending = set(futures)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for fut in pending:
                    fut.cancel()
                    i = futures[fut]
                    results[i] = SubagentResult(
                        task_id=prepared[i]["task_id"],
                        status="timed_out",
                        error=f"超过 {timeout_sec:.1f}s",
                    )
                break
            try:
                for fut in as_completed(pending, timeout=remaining):
                    pending.discard(fut)
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        results[i] = SubagentResult(
                            task_id=prepared[i]["task_id"],
                            status="failed",
                            error=str(exc),
                        )
            except FuturesTimeoutError:
                for fut in pending:
                    fut.cancel()
                    i = futures[fut]
                    results[i] = SubagentResult(
                        task_id=prepared[i]["task_id"],
                        status="timed_out",
                        error=f"超过 {timeout_sec:.1f}s",
                    )
                break
    return [r for r in results if r is not None]
