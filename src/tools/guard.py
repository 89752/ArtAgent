"""
工具调用守卫（借鉴 ragent `LLMMcpParameterExtractor` 的三态设计）。

对一次工具调用给出三种结局：
- SUCCESS：参数通过 schema 校验，可执行（必要时先填默认值）；
- NEED_CLARIFICATION：必填参数缺失（用户确实没给）→ 不调用工具，向用户追问；
- FAILED：值类型非法 / 枚举值非法 / 未知参数 / JSON 畸形 → 一律不调用工具，
  杜绝"garbage 进工具"和"过滤条件被静默丢弃"。

两条路径：
- validate_args(schema, args)：校验模型已提出的参数（图内实时闸门）；
- llm_extract_parameters(...)：从用户问题显式抽取参数（可独立用于非
  LangChain 工具直连），内部同样走 validate_args。

不引入 jsonschema 依赖，按 JSON Schema 常用子集手写校验，够用且可单测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.utils.json_utils import parse_json


@dataclass
class ToolDecision:
    """一次工具调用的守卫结论。"""

    status: str  # SUCCESS | NEED_CLARIFICATION | FAILED
    params: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# 校验（无第三方依赖）                                                  #
# ------------------------------------------------------------------ #

_PY_TYPE_TO_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _value_matches_type(value: Any, schema_type: Any) -> bool:
    """按 schema 的 type 声明检查单个值。支持联合类型与 number/int 兼容。"""
    if isinstance(schema_type, list):
        return any(_value_matches_type(value, t) for t in schema_type)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "null":
        return value is None
    return True  # 未知类型声明不误杀


def _check_property(name: str, value: Any, prop: dict, errors: list[str]) -> None:
    """按单个属性的 schema 做严格校验；违规直接记入 errors。"""
    # 联合/引用形态：取任一分支成立即可
    candidates: list[dict] = []
    if "anyOf" in prop:
        candidates = prop["anyOf"]
    elif "oneOf" in prop:
        candidates = prop["oneOf"]
    elif "$ref" in prop:
        return  # 无法解析引用，不误杀
    else:
        candidates = [prop]

    if "enum" in prop and value not in prop["enum"]:
        errors.append(f"参数 {name}: 值 {value!r} 不在允许枚举 {prop['enum']} 内")
        return

    type_ok = any(
        "type" in c and _value_matches_type(value, c["type"]) for c in candidates
    )
    # 未声明 type 的分支（如仅 default）视为通过类型检查
    declared = any("type" in c for c in candidates)
    if declared and not type_ok:
        expected = " / ".join(
            str(c["type"]) for c in candidates if "type" in c
        )
        errors.append(f"参数 {name}: 值 {value!r} 类型应为 {expected}")


def validate_args(schema: dict, args: dict) -> ToolDecision:
    """校验模型提出的工具参数。schema 为 JSON Schema（含 properties/required）。"""
    if not isinstance(schema, dict):
        schema = {}
    properties = schema.get("properties") or {}
    if not isinstance(args, dict):
        return ToolDecision(status="FAILED", params={}, errors=["参数必须是 JSON 对象"])

    params: dict[str, Any] = {}
    errors: list[str] = []
    missing: list[str] = []

    # 1) 未知参数（模型幻觉键名）→ FAILED，绝不静默丢弃
    for key in args:
        if key not in properties:
            errors.append(f"未知参数 {key}（合法参数：{sorted(properties)}）")

    # 2) 类型 / 枚举严格校验
    for name, value in args.items():
        prop = properties.get(name)
        # 显式 null 与"缺失"等价：不做类型校验，交给必填检查统一判定
        if prop is not None and value is not None:
            _check_property(name, value, prop, errors)
        params[name] = value

    # 3) 必填检查（显式 null 与缺失等价）
    for name in schema.get("required") or []:
        prop = properties.get(name) or {}
        if "default" in prop:
            continue
        if name not in params or params[name] is None:
            missing.append(name)

    if errors:
        return ToolDecision(status="FAILED", params=params, errors=errors)
    if missing:
        return ToolDecision(status="NEED_CLARIFICATION", params=params, missing=missing)
    return ToolDecision(status="SUCCESS", params=fill_defaults(params, properties))


def fill_defaults(params: dict[str, Any], properties: dict) -> dict[str, Any]:
    """SUCCESS 时补默认值（仅补缺失键）。"""
    out = dict(params)
    for name, prop in (properties or {}).items():
        if name not in out and isinstance(prop, dict) and "default" in prop:
            out[name] = prop["default"]
    return out


# ------------------------------------------------------------------ #
# LLM 显式参数抽取（独立路径，供工具直连使用）                             #
# ------------------------------------------------------------------ #

EXTRACT_SYSTEM_PROMPT = """你是工具参数抽取器。根据用户问题，为下面的工具抽取参数。

工具定义：
{tool_definition}

要求：
1. 只输出 JSON 对象，不要解释，不要 markdown 代码块；
2. 只从用户问题中提取信息；用户没提供的信息【不要编造】；
3. 必填参数缺失时，把该参数名原样放进 "missing" 数组并保持其缺失；
4. 输出格式：{{"params": {{...}}, "missing": [...]}}。

用户问题：
{user_question}"""


def build_extract_prompt(tool_name: str, schema: dict, question: str) -> str:
    """生成参数抽取 prompt。"""
    definition = {
        "name": tool_name,
        "description": (schema or {}).get("description", ""),
        "parameters": {k: v for k, v in (schema or {}).get("properties", {}).items()},
        "required": (schema or {}).get("required", []),
    }
    return EXTRACT_SYSTEM_PROMPT.format(
        tool_definition=json.dumps(definition, ensure_ascii=False, indent=2),
        user_question=question,
    )


def _parse_extraction(raw: str) -> tuple[Optional[dict], Optional[list[str]]]:
    """解析抽取响应：返回 (params, missing)；畸形返回 (None, None)。"""
    data = parse_json(raw)
    if not isinstance(data, dict):
        return None, None
    params = data.get("params")
    missing = data.get("missing") or []
    if not isinstance(params, dict):
        params = {}
    if not isinstance(missing, list):
        missing = []
    return params, missing


def llm_extract_parameters(
    tool_name: str,
    schema: dict,
    question: str,
    llm: Optional[Callable[[str], str]] = None,
) -> ToolDecision:
    """从用户问题显式抽取工具参数，再走同一套三态校验。

    llm 可注入（默认低温度实例），便于单测。任何失败都判 FAILED。
    """
    if llm is None:
        from src.utils.llm import get_llm

        def _default_llm(prompt: str) -> str:
            return get_llm(temperature=0.1).invoke(prompt).content

        llm = _default_llm

    prompt = build_extract_prompt(tool_name, schema, question)
    try:
        raw = llm(prompt)
    except Exception:
        return ToolDecision(status="FAILED", errors=["参数抽取 LLM 调用失败"])

    params, extracted_missing = _parse_extraction(raw)
    if params is None:
        return ToolDecision(status="FAILED", errors=["参数抽取响应无法解析"])

    decision = validate_args(schema, params)
    if decision.status == "FAILED":
        return decision
    # LLM 自报的 missing 与 schema 校验结果取并集，确保追问完整
    merged_missing = sorted(set(decision.missing) | set(extracted_missing or []))
    if merged_missing:
        return ToolDecision(
            status="NEED_CLARIFICATION",
            params=decision.params,
            missing=merged_missing,
        )
    return decision


# ------------------------------------------------------------------ #
# 消息构造（供 graph 节点把结论回灌给模型）                                #
# ------------------------------------------------------------------ #

def guard_tool_message(tool_call_id: str, tool_name: str, decision: ToolDecision):
    """把守卫结论构造成 ToolMessage，让模型看到并自我修正或向用户追问。"""
    from langchain_core.messages import ToolMessage

    if decision.status == "SUCCESS":
        raise ValueError("SUCCESS 不需要守卫消息，直接执行工具")

    if decision.status == "NEED_CLARIFICATION":
        content = json.dumps(
            {
                "status": "NEED_CLARIFICATION",
                "message": (
                    f"工具 {tool_name} 缺少必填参数 {decision.missing}。"
                    "如果可以从对话历史推断，请补全后重试；否则直接向用户提问获取，"
                    "不要编造参数值。"
                ),
                "missing": decision.missing,
            },
            ensure_ascii=False,
        )
    else:
        content = json.dumps(
            {
                "status": "FAILED",
                "message": (
                    f"工具 {tool_name} 的参数校验失败：{decision.errors}。"
                    "请修正参数后重新调用，不要重复同样的错误。"
                ),
                "errors": decision.errors,
            },
            ensure_ascii=False,
        )
    return ToolMessage(
        content=content,
        name=tool_name,
        tool_call_id=tool_call_id,
        id=f"guard:{tool_call_id}",
    )
