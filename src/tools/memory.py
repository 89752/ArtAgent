"""Agent 主动记忆工具（Phase 4）：remember / recall / forget。

让 agent 显式决定"记什么、查什么、忘什么"——这是"有记忆主体感"的关键动作。
v1 用户维度固定为 default_user（多用户 Web 场景后续注入当前用户）。
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.memory.agent_memory import (
    DEFAULT_USER,
    forget as _forget,
    recall as _recall,
    remember as _remember,
)


@tool
def remember(key: str, fact: str) -> str:
    """把一条关于用户的事实记入长期记忆（如偏好、讨论过的画作、重要结论）。

    Args:
        key: 记忆键（简短名词，如 "preferred_style"、"discussed_works"）
        fact: 要记住的事实内容

    Returns:
        记忆已保存的确认信息
    """
    _remember(DEFAULT_USER, key, fact)
    return f"已记住 [{key}]：{fact[:80]}"


@tool
def recall(query: str) -> list[dict]:
    """按关键词检索长期记忆中的相关事实。

    适用场景：回答需要参考用户过往偏好/已讨论内容时。

    Args:
        query: 关键词（如 "风格"、"莫奈"）

    Returns:
        命中的记忆列表 [{key, content, updated_at}]
    """
    return _recall(DEFAULT_USER, query)


@tool
def forget(key: str) -> str:
    """删除一条长期记忆。

    Args:
        key: 要删除的记忆键

    Returns:
        删除结果确认
    """
    deleted = _forget(DEFAULT_USER, key)
    return f"已删除记忆 [{key}]" if deleted else f"未找到记忆 [{key}]"
