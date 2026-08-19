"""S4 校验与安全：完整性、框架一致性、建议具体性、越界词、免责声明。"""

from __future__ import annotations

import copy
import re

from src.analysis.schemas import (
    LAYER1_DIMS,
    LAYER1_REQUIRED,
    LAYER2_REQUIRED,
    LAYER3_REQUIRED,
    REQUIRED_TOP,
)

VAGUE_PHRASES = [
    "画得自然一点",
    "自然一点",
    "再多练练",
    "注意感觉",
    "凭感觉",
    "放松一点",
    "多观察",
    "再多画",
    "继续加油",
]

BOUNDARY_PATTERNS = [
    "焦虑症",
    "抑郁症",
    "抑郁",
    "心理问题",
    "心理诊断",
    "人格特质",
    "性格",
    "情绪困扰",
    "自闭",
    "多动",
    "创伤",
    "心理状态",
]

DISCLAIMER = "本分析由 AI 基于画面判断，存在误判可能，不构成专业美术教育替代。"

_NON_REALISTIC = {"abstract", "childlike", "decorative"}


def missing_fields(report: dict) -> list[str]:
    """返回缺失的必填字段路径。"""
    if not isinstance(report, dict):
        return ["<报告非 JSON 对象>"]
    missing: list[str] = []
    for k in REQUIRED_TOP:
        if not str(report.get(k) or "").strip():
            missing.append(k)
    l1 = report.get("layer1_technique")
    if isinstance(l1, dict):
        for dim in LAYER1_DIMS:
            item = l1.get(dim)
            if not isinstance(item, dict):
                missing.append(f"layer1_technique.{dim}")
                continue
            for f in LAYER1_REQUIRED:
                if f == "evidence":
                    if not isinstance(item.get(f), list):
                        missing.append(f"layer1_technique.{dim}.evidence")
                elif f == "applies":
                    continue  # applies 可为 bool（False 也是有效值）
                elif not str(item.get(f) or "").strip():
                    missing.append(f"layer1_technique.{dim}.{f}")
    l2 = report.get("layer2_style_mood")
    if isinstance(l2, dict):
        for f in LAYER2_REQUIRED:
            if not str(l2.get(f) or "").strip():
                missing.append(f"layer2_style_mood.{f}")
    l3 = report.get("layer3_suggestions")
    if isinstance(l3, dict):
        for f in LAYER3_REQUIRED:
            if not str(l3.get(f) or "").strip():
                missing.append(f"layer3_suggestions.{f}")
    return missing


def fix_framework_consistency(report: dict) -> dict:
    """强制修正：非写实框架下透视标记为不适用。"""
    out = copy.deepcopy(report)
    if str(out.get("framework") or "") in _NON_REALISTIC:
        l1 = out.get("layer1_technique")
        if isinstance(l1, dict) and isinstance(l1.get("perspective"), dict):
            l1["perspective"]["applies"] = False
            l1["perspective"]["kind"] = "not_applicable"
    return out


def vague_suggestions(report: dict) -> list[str]:
    """返回命中空泛表述的建议（要求模型重写）。"""
    hits: list[str] = []
    items = (
        (report.get("layer3_suggestions") or {}).get("priority_items")
        if isinstance(report.get("layer3_suggestions"), dict)
        else None
    )
    for i, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        text = f"{item.get('issue') or ''} {item.get('action') or ''}"
        for phrase in VAGUE_PHRASES:
            if phrase in text:
                hits.append(f"priority_items[{i}] 含空泛表述「{phrase}」")
                break
    return hits


def boundary_hits(text: str) -> list[str]:
    """返回心理推断/诊断相关越界词命中。"""
    hits: list[str] = []
    for pat in BOUNDARY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    return hits


def inject_disclaimer(report: dict) -> dict:
    out = copy.deepcopy(report)
    out["disclaimer"] = DISCLAIMER
    return out


def sanitize_report(report: dict) -> dict:
    """最终落库前强制规范化（不依赖模型自觉）。"""
    out = fix_framework_consistency(report)
    return inject_disclaimer(out)
