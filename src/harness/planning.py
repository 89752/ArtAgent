"""Choose the smallest durable plan that matches the requested depth."""
from __future__ import annotations

import re


_DEEP_RESEARCH = re.compile(
    r"(深度研究|系统研究|深入研究|完整报告|研究报告|调研报告|专题报告|"
    r"全面分析|系统分析|长篇报告|论文|文献综述|多阶段研究|deep research)",
    re.IGNORECASE,
)


def default_plan(objective: str) -> list[str]:
    """Keep ordinary evidence questions fast; reserve orchestration for explicit research.

    A single ReAct step can retrieve several passages and compare them. Splitting every
    question into collection/comparison/report stages repeats the full graph and grows
    context without adding value for ordinary lookups.
    """
    if _DEEP_RESEARCH.search(objective or ""):
        return [
            "检索并整理与目标直接相关的可定位证据",
            "交叉比较证据，识别冲突、缺口与不确定性",
            "生成完整研究报告，使用 [E编号] 引用已有证据",
        ]
    return [
        "检索必要证据并直接生成可交付答案；明确区分事实、观察与推断，使用 [E编号] 引用来源"
    ]
