"""记忆冲突解析 + 守卫内容规范化（MEMORY_SMART_MERGE 默认关闭）。

对标 Mem0 / ChatGPT 记忆的做法：同实体旧记忆与新信息冲突时，不再机械地
"新盖旧"，而是让 LLM 判断 REPLACE（漂移覆盖）/ MERGE（并存）/ SKIP（不写）。

纪律：
- 关闭 / LLM 失败 / 畸形输出 → 回落确定性行为（REPLACE 覆盖），不阻塞主流程；
- 规范化只在"用户明确要求记忆"的守卫路径生效，成本可控。
"""

from __future__ import annotations

from typing import Callable, Optional

from src.utils.env import env_flag
from src.utils.json_utils import parse_json


CONFLICT_PROMPT = """你是记忆冲突判定模块。用户长期记忆里已有一条旧记录，现在来了新信息，
请判断如何处理。

旧记忆（kind={kind}，entity={entity}）：
{old}

新信息：
{new}

只输出 JSON，不要解释：
{{"action": "REPLACE", "content": "合并后的记忆内容"}}

规则：
- REPLACE：新旧矛盾/新信息更新更准 → 用新内容覆盖旧记忆（content 填新内容）；
- MERGE：两者都是有效的独立事实，不矛盾、方向不同 → 并存（content 填新内容）；
- SKIP：新信息与旧记忆重复、或置信度明显更低、或只是口头客套 → 不写入
  （content 填空串）。

content 一律用第三人称陈述句（"用户偏好…"），中文，单条一个事实。"""


NORMALIZE_PROMPT = """把用户表达的记忆意图规范化为一条第三人称陈述句（中文）。

用户原话：
{raw}

只输出规范化后的句子本身，不要解释、不要引号。要求：
- 主语用"用户"（如 "用户特别喜欢莫奈的睡莲系列"）；
- 去掉"记住/记下/以后记得"等指令词与客套；
- 保持原意，不编造。"""


def smart_merge_enabled() -> bool:
    """MEMORY_SMART_MERGE 开关（默认开，对齐 DeerFlow 的合并式记忆维护）。"""
    return env_flag("MEMORY_SMART_MERGE", default="1")


def _default_llm() -> Callable[[str], str]:
    from src.utils.llm import get_deterministic_llm

    def _invoke(p: str) -> str:
        return get_deterministic_llm().invoke(p).content

    return _invoke


def resolve_conflict(
    old_content: str,
    new_content: str,
    kind: str = "preference",
    entity: str = "",
    llm: Optional[Callable[[str], str]] = None,
) -> dict:
    """判定新旧记忆关系；关闭/失败回落 REPLACE（确定性覆盖）。"""
    if not smart_merge_enabled():
        return {"action": "REPLACE", "content": (new_content or "").strip()}
    prompt = CONFLICT_PROMPT.format(
        kind=kind, entity=entity or "(无)",
        old=(old_content or "").strip(), new=(new_content or "").strip(),
    )
    try:
        if llm is None:
            llm = _default_llm()
        data = parse_json(llm(prompt))
    except Exception:  # noqa: BLE001 —— 失败回落确定性行为
        return {"action": "REPLACE", "content": (new_content or "").strip()}
    if not isinstance(data, dict):
        return {"action": "REPLACE", "content": (new_content or "").strip()}
    action = str(data.get("action") or "REPLACE").strip().upper()
    if action not in {"REPLACE", "MERGE", "SKIP"}:
        action = "REPLACE"
    content = str(data.get("content") or "").strip() or (new_content or "").strip()
    return {"action": action, "content": content}


def normalize_memory_text(
    raw: str,
    llm: Optional[Callable[[str], str]] = None,
) -> str:
    """把守卫代调的用户原话规范化为单条陈述；关闭/失败回落原文。"""
    raw = (raw or "").strip()
    if not raw or not smart_merge_enabled():
        return raw
    prompt = NORMALIZE_PROMPT.format(raw=raw)
    try:
        if llm is None:
            llm = _default_llm()
        text = str(llm(prompt)).strip().strip('"').strip()
        return text[:500] or raw
    except Exception:  # noqa: BLE001
        return raw
