"""子智能体派发工具：delegate_task。"""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def delegate_task(tasks: list[dict[str, str]]) -> str:
    """把多个独立的调研子任务并行派发给子智能体执行。

    当用户的问题包含多个相互独立的研究对象时使用，例如：
    - 对比多位画家/多幅画（每位画家独立深挖后再对比）
    - 分别梳理多个流派/多个馆藏
    - 同一问题的不同角度并行调研

    子任务并行执行（受 config.yaml subagents.max_concurrent 限制），
    全部完成后一次性返回结构化结果（findings / sources / confidence）。

    Args:
        tasks: 子任务列表，每项 {"description": "3-5 字描述", "prompt": "详细任务指令"}
    """
    from src.subagents.executor import run_tasks

    results = run_tasks(tasks)
    payload = {
        "status": (
            "completed"
            if all(r.status == "completed" for r in results)
            else "partial"
        ),
        "results": [
            {
                "task_id": r.task_id,
                "status": r.status,
                "findings": (r.result or {}).get("findings", ""),
                "sources": (r.result or {}).get("sources", []),
                "confidence": (r.result or {}).get("confidence", "low"),
                "error": r.error,
            }
            for r in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False)
