"""技能系统 v2：SKILL.md 发现、解析、结构化执行。

技能 = 程序化多步能力。v2 在 v1（迷你 ReAct）基础上加两层：
- A 结构化执行：front matter 声明 steps_json（步骤清单）与
  output_schema_json（必填输出字段），执行器对最终输出做确定性校验，
  缺字段自动要求补齐（有界重试），不再是"prompt 说什么算什么"；
- B 领域知识内置：SKILL.md 正文注入专家框架（检查单/维度表），
  随系统提示进入执行上下文。

目录约定：
    agent_skills/<skill_id>/SKILL.md
front matter（--- 包裹，key: value）：
    name / description / when_to_use / version / tools(JSON 数组) /
    max_steps / steps_json(JSON 数组) / output_schema_json(JSON 对象)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.utils.llm import get_deterministic_llm


@dataclass
class Skill:
    """一个已解析的技能定义（v2：含结构化步骤与输出 schema）。"""

    id: str
    name: str
    description: str
    when_to_use: str
    tools: list[str] = field(default_factory=list)
    max_steps: int = 6
    instructions: str = ""
    steps: list[str] = field(default_factory=list)
    output_schema: dict[str, str] = field(default_factory=dict)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """解析 `---` 包裹的 front matter 与正文（支持多行 JSON 值）。"""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, (text or "").strip()
    kv: dict[str, str] = {}
    lines = m.group(1).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line:
            i += 1
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        # 多行 JSON 值：从 [ 或 { 开始，消费到括号闭合
        if v.startswith(("[", "{")):
            depth = 0
            for c in v:
                if c in "[{":
                    depth += 1
                elif c in "]}":
                    depth -= 1
            buf = [v]
            while depth > 0 and i + 1 < len(lines):
                i += 1
                nxt = lines[i]
                buf.append(nxt)
                for c in nxt:
                    if c in "[{":
                        depth += 1
                    elif c in "]}":
                        depth -= 1
            v = "".join(buf)
        kv[k] = v
        i += 1
    return kv, m.group(2).strip()


def _parse_list(raw: str) -> list[str]:
    """解析 "[a, b]" 形式的列表。"""
    if not raw:
        return []
    cleaned = raw.strip().strip("[]")
    return [x.strip().strip("'\"") for x in cleaned.split(",") if x.strip()]


def _parse_json_field(raw: str, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def load_skills(skills_dir: Path = Path("agent_skills")) -> list[Skill]:
    """扫描 skills 目录，解析全部 SKILL.md。"""
    skills: list[Skill] = []
    if not skills_dir.exists():
        return skills
    for skill_dir in sorted(skills_dir.iterdir()):
        md = skill_dir / "SKILL.md"
        if not md.exists():
            continue
        kv, body = _parse_front_matter(md.read_text(encoding="utf-8"))
        if not kv.get("name"):
            continue
        skills.append(
            Skill(
                id=skill_dir.name,
                name=kv["name"],
                description=kv.get("description", ""),
                when_to_use=kv.get("when_to_use", ""),
                tools=_parse_list(kv.get("tools", "[]")),
                max_steps=int(kv.get("max_steps", "6") or 6),
                instructions=body,
                steps=_parse_json_field(kv.get("steps_json", "[]"), []),
                output_schema=_parse_json_field(kv.get("output_schema_json", "{}"), {}),
            )
        )
    return skills


# ── 工具注册表：技能允许调用的工具集（原子工具） ────────────────
def _build_tool_registry() -> dict[str, object]:
    """技能可用工具集：直接复用工具带统一注册表。"""
    from src.tools.registry import TOOL_BY_NAME

    return dict(TOOL_BY_NAME)


TOOL_REGISTRY: dict[str, object] = _build_tool_registry()


def _validate_output(text: str, schema: dict[str, str]) -> tuple[bool, list[str]]:
    """校验技能最终输出是否满足 output_schema：必填字段非空。"""
    if not schema:
        return True, []
    if not text:
        return False, list(schema.keys())
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        data = json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return False, list(schema.keys())
        try:
            data = json.loads(cleaned[start : end + 1])
        except Exception:
            return False, list(schema.keys())
    if not isinstance(data, dict):
        return False, list(schema.keys())
    missing = [
        k for k in schema
        if not str(data.get(k) or "").strip()
    ]
    return (not missing), missing


def _skill_runner(skill: Skill) -> Callable[[str], str]:
    """返回技能的执行函数（task → 结构化结果文本）。

    v2：步骤清单与领域框架注入系统提示；最终输出按 output_schema
    确定性校验，缺字段提示补齐（有界重试）。
    """

    def run(task: str) -> str:
        tools = [TOOL_REGISTRY[name] for name in skill.tools if name in TOOL_REGISTRY]
        if not tools:
            return f"技能 {skill.name} 未声明可用工具，无法执行。"
        llm = get_deterministic_llm().bind_tools(tools)
        step_hint = (
            "\n".join(f"{i + 1}. {s}" for i, s in enumerate(skill.steps))
            if skill.steps
            else skill.instructions
        )
        schema_hint = (
            json.dumps(skill.output_schema, ensure_ascii=False)
            if skill.output_schema
            else "(无)"
        )
        system = (
            f"你正在执行技能「{skill.name}」。\n\n执行步骤：\n{step_hint}\n\n"
            f"{skill.instructions}\n\n"
            f"最终必须输出 JSON（字段说明）：{schema_hint}\n"
            "先用声明允许的工具收集信息，最后只输出完整 JSON，不要输出其他内容。"
        )
        messages: list = [SystemMessage(content=system), HumanMessage(content=task)]
        for step in range(skill.max_steps):
            try:
                resp = llm.invoke(messages)
            except Exception as e:  # noqa: BLE001
                return f"技能执行失败（第{step + 1}步）：{e}"
            tool_calls = getattr(resp, "tool_calls", None) or []
            if tool_calls:
                messages.append(resp)
                for tc in tool_calls:
                    tool = TOOL_REGISTRY.get(tc.get("name"))
                    try:
                        if tool is None:
                            raise KeyError(f"未注册工具 {tc.get('name')}")
                        output = tool.invoke(tc.get("args") or {})
                    except Exception as e:  # noqa: BLE001 — 工具失败回灌给模型
                        output = f"工具执行失败：{e}"
                    messages.append(
                        ToolMessage(
                            content=str(output)[:2000],
                            name=tc.get("name"),
                            tool_call_id=tc.get("id"),
                        )
                    )
                continue
            text = str(resp.content)
            if skill.output_schema:
                ok, missing = _validate_output(text, skill.output_schema)
                if ok:
                    return text
                # 缺字段：有界补齐（每轮一次，超限由 max_steps 兜底）
                messages.append(AIMessage(content=text))
                messages.append(
                    HumanMessage(
                        content=f"输出缺少必填字段：{missing}。请补齐后只输出完整 JSON。"
                    )
                )
                continue
            return text
        last = str(messages[-1].content)[:2000]
        return last + "\n\n（已达技能步数上限）"

    return run


def register_skills(skills_dir: Path = Path("agent_skills")) -> list:
    """把全部技能注册为 skill_<id> 工具（StructuredTool）。"""
    from langchain_core.tools import StructuredTool

    tools: list = []
    for skill in load_skills(skills_dir):
        runner = _skill_runner(skill)
        description = skill.description or skill.when_to_use or f"执行技能 {skill.name}"
        tools.append(
            StructuredTool.from_function(
                func=runner,
                name=f"skill_{skill.id}",
                description=description,
            )
        )
    return tools
