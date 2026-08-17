"""技能显式激活：/skill-id 语法（对齐 DeerFlow 的斜杠激活）。

设计：
- 模型自觉调用 skill_<id> 工具是"软激活"（看心情）；
- 用户以 /<skill-id> 开头是"硬激活"（代码保证）：
  完整注入执行步骤、领域框架与输出 schema，并改写任务文本。
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage

from src.skills.loader import Skill

_SLASH_RE = re.compile(r"^\s*/([A-Za-z0-9_-]+)(?:\s+(.*?))?\s*$", re.S)
_USER_MESSAGE_NAME = "user-input"


def find_skill(skills: list[Skill], token: str) -> Optional[Skill]:
    """按 id 或 name 匹配技能（大小写不敏感）。"""
    token_l = token.lower()
    for skill in skills:
        if skill.id.lower() == token_l or skill.name.lower() == token_l:
            return skill
    return None


def parse_slash_skill(
    text: str, skills: list[Skill]
) -> Optional[tuple[Skill, str]]:
    """解析 "/skill-id 任务文本"；未匹配返回 None。"""
    match = _SLASH_RE.match(text or "")
    if not match:
        return None
    skill = find_skill(skills, match.group(1))
    if skill is None:
        return None
    return skill, (match.group(2) or "").strip()


def build_activation_block(skill: Skill) -> str:
    """把技能完整内容（步骤 + 领域框架 + 输出 schema）注入系统提示。"""
    steps = (
        "\n".join(f"{i + 1}. {s}" for i, s in enumerate(skill.steps))
        if skill.steps
        else skill.instructions
    )
    schema = (
        json.dumps(skill.output_schema, ensure_ascii=False)
        if skill.output_schema
        else "(无)"
    )
    return (
        f"<技能激活>{skill.name}</技能激活>\n"
        f"用户显式激活技能「{skill.name}」。必须严格执行以下流程并按格式输出：\n\n"
        f"执行步骤：\n{steps}\n\n"
        f"{skill.instructions}\n\n"
        f"输出 JSON schema：{schema}\n"
        "完成后把技能输出整理成对用户友好的回答。"
    )


def apply_slash_activation(
    messages: list, skills: list[Skill]
) -> tuple[list, Optional[str], Optional[str]]:
    """扫描最后一条真实用户消息；命中 /skill 则改写任务并返回激活块。

    Returns:
        (新消息列表, 激活的技能名, 激活块)；未命中时后两者为 None。
    """
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        msg = out[i]
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == _USER_MESSAGE_NAME:
            content = msg.content
            if isinstance(content, str):
                parsed = parse_slash_skill(content, skills)
                if parsed is not None:
                    skill, task = parsed
                    if not task:
                        task = "请执行该技能处理我的请求。"
                    out[i] = HumanMessage(
                        content=task, name=_USER_MESSAGE_NAME, id=msg.id
                    )
                    return out, skill.name, build_activation_block(skill)
            break
    return out, None, None
