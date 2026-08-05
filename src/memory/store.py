"""用户偏好读写兼容层：统一落在 memory_items（kind='preference'）。

历史沿革：
- 旧实现用 preferences.db（user_id/kind/value/weight）独立落盘；
- 记忆系统上线后，新写入统一走 memory_items 主存储；
- 本模块保留旧函数签名（upsert/load/list/delete/clear），内部全部
  转为 memory_items 读写，外部调用方（web/service、图节点、工具层）无需改动。

注意：入参 kind 兼容旧值 "artist"/"style"（仅作校验），落库统一为
memory_items 的 kind='preference'，因此 list_preferences 返回的
kind 恒为 'preference'。
"""

from __future__ import annotations

from src.memory.memory_items import (
    add_memory,
    delete_memory,
    list_memories,
)

VALID_KINDS = {"artist", "style"}


def _pref_items(user_id: str) -> list[dict]:
    """读取某用户 user scope 下的全部偏好条目（kind='preference'）。"""
    if not user_id:
        return []
    try:
        return [
            i for i in list_memories(user_id, scope="user")
            if i.get("kind") == "preference"
        ]
    except Exception:  # noqa: BLE001 —— 表不可用时按空处理，不阻塞主流程
        return []


def upsert_preference(
    user_id: str, kind: str, value: str, weight: float = 1.0
) -> None:
    """兼容接口：写一条用户偏好到 memory_items。

    kind 接受旧值 artist/style（仅作入参校验），落库统一为
    kind='preference'；importance 由 weight 折算（封顶 1.0）。
    """
    kind = (kind or "").strip().lower()
    value = (value or "").strip()
    if kind not in VALID_KINDS or not value or not user_id:
        return
    try:
        importance = min(1.0, max(0.1, float(weight)))
    except (TypeError, ValueError):
        importance = 0.5
    add_memory(
        user_id=user_id,
        content=value,
        entity=value,
        kind="preference",
        scope="user",
        source="user_explicit",
        importance=importance,
    )


def load_preferences(user_id: str, top_k: int = 5) -> dict[str, list[str]]:
    """读取用户偏好（与旧 API 形状一致）。

    Returns:
        {"artists": [...], "styles": [...]}
        无偏好时返回 {"artists": [], "styles": []}
    """
    result: dict[str, list[str]] = {"artists": [], "styles": []}
    items = _pref_items(user_id)
    if items:
        result["artists"] = [str(i["content"]) for i in items[:top_k]]
    return result


def clear_preferences(user_id: str) -> int:
    """清空某用户的全部偏好条目；返回删除条数。"""
    n = 0
    for item in _pref_items(user_id):
        if delete_memory(user_id, item["id"]):
            n += 1
    return n


def list_preferences(user_id: str) -> list[dict]:
    """返回该用户的全部偏好分项（记忆面板：kind/value/weight/updated_at）。"""
    return [
        {
            "kind": "preference",
            "value": str(i["content"]),
            "weight": float(i.get("importance") or 0.5),
            "updated_at": i.get("updated_at") or "",
        }
        for i in _pref_items(user_id)
    ]


def delete_preference(user_id: str, kind: str, value: str) -> bool:
    """按内容删除一条偏好；kind 兼容 artist/style，实际匹配 content。"""
    kind = (kind or "").strip().lower()
    value = (value or "").strip()
    if kind not in VALID_KINDS or not value or not user_id:
        return False
    for item in _pref_items(user_id):
        if item.get("content") == value:
            return delete_memory(user_id, item["id"])
    return False
