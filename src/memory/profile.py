"""跨线程用户画像聚合（ChatGPT 式"记住你"）。

把 user scope 的有效记忆按 重要性×新鲜度 取 top，LLM 压缩成 3-6 条画像
陈述，存为 kind='profile'、entity='user_profile'（add_memory 自动 supersede
旧画像，保留版本历史）。

纪律：关闭 / LLM 失败 / 畸形输出 → 确定性兜底（top 内容拼接），不阻塞主流程；
画像条目在容量淘汰中受保护。
"""

from __future__ import annotations

from typing import Callable, Optional

from src.memory.memory_items import (
    _days_since,
    add_memory,
    list_memories,
)
from src.utils.env import env_flag, env_int


PROFILE_ENTITY = "user_profile"
PROFILE_MAX_CHARS = 1200

PROFILE_PROMPT = """你是用户画像聚合模块。根据用户的长期记忆条目，生成一份
简洁、稳定的用户画像（3-6 条陈述）。

记忆条目：
{items}

要求：
- 每条用第三人称陈述句（"用户喜欢印象派…"），中文，单条一个事实；
- 只保留稳定偏好/事实（审美、居住、职业、长期目标），去掉一次性/临时信息；
- 合并同义条目，不编造；
- 直接输出画像内容（可用 "；" 分隔多条），不要解释、不要 markdown。"""


def profile_enabled() -> bool:
    return env_flag("MEMORY_PROFILE_REFRESH")


def profile_max_age_days() -> int:
    return max(1, env_int("MEMORY_PROFILE_MAX_AGE_DAYS", 7))


def _score(item: dict) -> float:
    imp = float(item.get("importance") or 0.5)
    return imp * (0.9 ** _days_since(str(item.get("updated_at") or "")))


def load_profile_item(user_id: str) -> Optional[dict]:
    """当前生效的用户画像条目（无则 None）。"""
    for item in list_memories(user_id, scope="user"):
        if item.get("kind") == "profile" and item.get("entity") == PROFILE_ENTITY:
            return item
    return None


def _default_llm() -> Callable[[str], str]:
    from src.utils.llm import get_deterministic_llm

    def _invoke(p: str) -> str:
        return get_deterministic_llm().invoke(p).content

    return _invoke


def build_profile_text(items: list[dict], llm: Optional[Callable[[str], str]] = None) -> str:
    """压缩 top 记忆为画像文本；LLM 失败回落确定性拼接。"""
    if not items:
        return ""
    lines = "\n".join(
        f"- [{i.get('kind')}] {str(i.get('content') or '').strip()[:200]}"
        for i in items[:20]
    )
    try:
        if llm is None:
            llm = _default_llm()
        text = str(llm(PROFILE_PROMPT.format(items=lines))).strip().strip('"')
        return text[:PROFILE_MAX_CHARS] if text else ""
    except Exception:  # noqa: BLE001 —— 失败回落确定性拼接
        pass
    top = sorted(items, key=_score, reverse=True)[:5]
    parts = [str(i.get("content") or "").strip() for i in top if str(i.get("content") or "").strip()]
    return ("用户画像：" + "；".join(parts))[:PROFILE_MAX_CHARS] if parts else ""


def maybe_refresh_profile(
    user_id: str,
    llm: Optional[Callable[[str], str]] = None,
) -> dict:
    """按需刷新用户画像：开关关闭/画像新鲜/无记忆 → 跳过；否则聚合落库。"""
    if not profile_enabled():
        return {}
    items = [
        i for i in list_memories(user_id, scope="user")
        if i.get("kind") != "profile"
    ]
    if not items:
        return {"skipped": "no_memories"}
    existing = load_profile_item(user_id)
    if existing and _days_since(str(existing.get("updated_at") or "")) < profile_max_age_days():
        return {"skipped": "fresh"}
    text = build_profile_text(items, llm)
    if not text:
        return {"skipped": "empty_profile"}
    item = add_memory(
        user_id=user_id,
        content=text,
        kind="profile",
        entity=PROFILE_ENTITY,
        scope="user",
        source="extracted",
        importance=0.95,
    )
    return {
        "action": item.get("action", "create"),
        "item_id": item.get("id"),
        "content_len": len(text),
    }
