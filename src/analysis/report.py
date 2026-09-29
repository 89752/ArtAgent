"""S3 三层分层分析（视觉模型 #2，schema 校验 + 有界重试）。"""

from __future__ import annotations
from src.harness.context import invoke_model

import json

from langchain_core.messages import HumanMessage

from src.analysis.prompts import (
    ANALYSIS_PROMPT,
    FOCUS_HINTS,
    FRAMEWORK_LABELS,
    PRINCIPLES,
)
from src.analysis.validate import (
    boundary_hits,
    missing_fields,
    sanitize_report,
    vague_suggestions,
)
from src.utils.json_utils import parse_json
from src.utils.logging_config import get_logger, log_event

logger = get_logger("analysis.report")


def _quality_hint(flags: list[str]) -> str:
    if not flags:
        return "无"
    return "、".join(flags) + "。仅基于可见画面分析，必要时降低置信度。"


def _build_prompt(
    gate: dict,
    metrics: dict,
    focus: str,
    quality_flags: list[str],
    fix_hint: str = "",
) -> str:
    framework = gate["framework"]
    hint = FOCUS_HINTS.get(focus, FOCUS_HINTS["all"])
    prompt = ANALYSIS_PROMPT.format(
        framework_label=FRAMEWORK_LABELS.get(framework, framework),
        framework=framework,
        focus_hint=hint,
        metrics_json=json.dumps(metrics, ensure_ascii=False),
        quality_hint=_quality_hint(quality_flags),
        principles=PRINCIPLES,
    )
    if fix_hint:
        prompt += f"\n\n上次输出存在问题，请修正后重新输出完整 JSON：{fix_hint}"
    return prompt


def generate_layered_report(
    image_b64: str,
    image_ext: str,
    gate: dict,
    metrics: dict,
    focus: str = "all",
    quality_flags: list[str] | None = None,
) -> dict:
    """生成并校验三层报告；缺失/空泛/越界时有界重试一次。"""
    from src.utils.llm import get_vision_llm

    flags = quality_flags or []
    last_report: dict | None = None
    fix_hint = ""
    # 严格限制为首次生成 + 一次修复。旧实现会在修复失败后再次执行外层
    # 循环，最坏调用三次视觉模型，且第三次没有携带修复提示。
    for attempt in range(2):
        prompt = _build_prompt(gate, metrics, focus, flags, fix_hint)
        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_ext};base64,{image_b64}"
                    },
                },
                {"type": "text", "text": prompt},
            ]
        )
        try:
            resp = invoke_model(get_vision_llm(), [msg])
        except Exception as e:  # noqa: BLE001
            log_event(logger, "report_llm_failed", attempt=attempt, error=str(e))
            continue
        data = parse_json(str(resp.content))
        if not isinstance(data, dict):
            continue
        last_report = data
        missing = missing_fields(data)
        vague = vague_suggestions(data)
        boundary = boundary_hits(json.dumps(data, ensure_ascii=False))
        if not missing and not vague and not boundary:
            break
        hints: list[str] = []
        if missing:
            hints.extend(f"缺少字段 {m}" for m in missing)
        if vague:
            hints.extend(vague)
        if boundary:
            hints.append("禁止出现心理推断/诊断词：" + "、".join(boundary))
        fix_hint = "；".join(hints)
    if not last_report:
        last_report = {}
    log_event(
        logger, "report_done",
        missing=missing_fields(last_report),
        vague=vague_suggestions(last_report),
    )
    return sanitize_report(last_report)
