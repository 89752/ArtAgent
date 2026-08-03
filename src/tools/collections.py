"""收藏与偏好工具（Phase 5 工具多元化）：
save_collection / list_collections / list_preferences。
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.memory.agent_memory import DEFAULT_USER
from src.memory.collections import (
    list_collections as _list_collections,
    save_collection as _save_collection,
)
from src.memory.store import load_preferences


@tool
def save_collection(name: str, artwork_titles: list[str]) -> str:
    """把一批画作标题保存为用户的收藏清单（跨会话持久化）。

    适用场景：用户要求"收藏/记下这几幅画"、"整理一个清单"。

    Args:
        name: 清单名称（如 "印象派最爱"）
        artwork_titles: 画作标题列表

    Returns:
        保存确认
    """
    _save_collection(DEFAULT_USER, name, artwork_titles)
    return f"已保存收藏清单 [{name}]（{len(artwork_titles)} 幅）"


@tool
def list_collections() -> list[dict]:
    """列出用户保存过的全部收藏清单。"""
    return _list_collections(DEFAULT_USER)


@tool
def list_preferences() -> dict:
    """读取用户长期偏好（喜欢的画家/风格，带权重），用于个性化推荐。"""
    return load_preferences(DEFAULT_USER)
