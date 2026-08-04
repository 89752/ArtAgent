"""收藏与偏好工具（Phase 5 工具多元化 + P1-6 补全）：
save_collection / list_collections / get_collection / delete_collection /
rename_collection / list_preferences。
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.memory.memory_items import get_memory_user_id
from src.memory.collections import (
    delete_collection as _delete_collection,
    get_collection as _get_collection,
    list_collections as _list_collections,
    rename_collection as _rename_collection,
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
    _save_collection(get_memory_user_id(), name, artwork_titles)
    return f"已保存收藏清单 [{name}]（{len(artwork_titles)} 幅）"


@tool
def list_collections() -> list[dict]:
    """列出用户保存过的全部收藏清单。"""
    return _list_collections(get_memory_user_id())


@tool
def get_collection(name: str) -> dict | None:
    """查看单个收藏清单的内容（按清单名精确匹配）。

    适用场景：用户问"我的'印象派'清单里有什么"。
    """
    return _get_collection(get_memory_user_id(), name)


@tool
def delete_collection(name: str) -> str:
    """删除一个收藏清单（跨会话持久化）。

    适用场景：用户要求"删掉这个清单""不要这份收藏了"。
    删除后不可恢复。
    """
    ok = _delete_collection(get_memory_user_id(), name)
    return f"已删除收藏清单 [{name}]" if ok else f"未找到收藏清单 [{name}]"


@tool
def rename_collection(old_name: str, new_name: str) -> str:
    """重命名一个收藏清单。

    适用场景：用户要求"把'印象派'改名为'最爱'"。
    新名称已存在或旧清单不存在时返回说明。
    """
    ok = _rename_collection(get_memory_user_id(), old_name, new_name)
    if ok:
        return f"已把收藏清单 [{old_name}] 改名为 [{new_name}]"
    return f"重命名失败：请检查 [{old_name}] 是否存在、[{new_name}] 是否已被占用"


@tool
def list_preferences() -> dict:
    """读取用户长期偏好（喜欢的画家/风格，带权重），用于个性化推荐。"""
    return load_preferences(get_memory_user_id())
