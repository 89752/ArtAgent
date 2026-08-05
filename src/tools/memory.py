"""记忆工具层（remember / recall / forget）：薄封装 memory_items 主存储。

记忆条目走 memory_items 主表（kind/scope/entity/importance/审计/软删除），
用户身份由 MEMORY_USER_ID 控制（默认 default_user，评估用 eval-test）；
冲突合并由 memory.conflict 按 MEMORY_SMART_MERGE 开关启用。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from src.memory.memory_items import (
    add_memory,
    delete_by_entity,
    delete_memory,
    get_memory_user_id,
    search_memories,
)


@tool
def remember(
    content: str,
    kind: str = "preference",
    entity: str = "",
    scope: str = "user",
    importance: float = 0.5,
) -> str:
    """把一条关于用户的事实/偏好记入长期记忆（跨会话持久化）。

    适用场景：用户明确说"记住/记下/以后记得"某件事（偏好、讨论过的画作、
    重要结论、身份信息）。**用户要求记忆时必须调用本工具，不能只口头确认。**

    Args:
        content: 单条陈述式记忆内容，如 "用户偏好莫奈的睡莲系列"
        kind:    记忆类型："preference" 偏好（默认）/ "fact" 事实 /
                 "profile" 用户画像 / "event" 事件
        entity:  关键实体（如 "莫奈" / "睡莲"），用于检索与冲突覆盖
        scope:   作用域："user" 跨会话（默认）/ "thread" 仅当前会话
        importance: 重要性 0-1（默认 0.5）

    Returns:
        保存确认（含记忆 id）
    """
    from src.memory.conflict import smart_merge_enabled

    item = add_memory(
        user_id=get_memory_user_id(),
        content=content,
        kind=kind,
        entity=entity,
        scope=scope,
        source="user_explicit",
        importance=importance,
        smart_conflict=smart_merge_enabled(),
    )
    action = item.get("action", "create")
    if action == "supersede":
        return f"好的，已更新为：{content[:80]}"
    if action == "skip":
        return f"这条和之前的记忆一致，无需重复保存：{content[:80]}"
    return f"好的，已记住：{content[:80]}"


@tool
def recall(query: str, top_k: int = 5) -> list[dict]:
    """按语义检索长期记忆中的相关条目（偏好/事实/画像）。

    适用场景：回答需要参考用户过往偏好/已讨论内容时，先查记忆再作答。

    Args:
        query: 自然语言查询（如 "用户喜欢什么风格"、"莫奈"）
        top_k: 返回条数（默认5）

    Returns:
        命中的记忆列表 [{id, kind, content, entity, source, importance,
        created_at, updated_at, score}]
    """
    return search_memories(
        user_id=get_memory_user_id(),
        query=query,
        top_k=top_k,
    )


@tool
def forget(entity: str = "", item_id: str = "") -> str:
    """删除一条长期记忆（软删除，删除后不再注入）。

    Args:
        entity: 按实体删除（如 "卡拉瓦乔"；可删多条同实体记忆）
        item_id: 按记忆 id 精确删除（recall 返回里的 id）

    Returns:
        删除结果确认
    """
    user_id = get_memory_user_id()
    if item_id:
        ok = delete_memory(user_id, item_id)
        return f"已删除记忆 [{item_id[:18]}]" if ok else f"未找到记忆 [{item_id[:18]}]"
    if entity:
        n = delete_by_entity(user_id, entity)
        return f"已删除 {n} 条与 [{entity}] 相关的记忆" if n else f"未找到与 [{entity}] 相关的记忆"
    return "请提供 entity 或 item_id 参数"
