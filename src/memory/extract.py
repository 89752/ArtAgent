"""记忆系统 Phase 1.5：自动抽取层（默认关闭）。

对标 ChatGPT Dreaming / Claude memory tool / Mem0 extract 的模式：
用户没有说"记住"时，系统在对话自然推进后主动抽取稳定事实与偏好，
落库 source='extracted'，冲突沿用 add_memory 的同义合并/漂移覆盖。

开关（默认关，验证后再开）：
- MEMORY_AUTO_EXTRACT=1         开启
- MEMORY_EXTRACT_INTERVAL=N     每 N 轮对话抽取一次（默认 2）

纪律：
- 只抽稳定事实/偏好，不抽一次性任务与寒暄；
- 敏感信息（证件号/密码/银行卡等）不落库；
- LLM 失败不阻塞主流程，且推进节流计数，避免每轮重试。
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from src.memory.memory_items import (
    VALID_KINDS,
    add_memory,
    delete_by_entity,
    get_memory_user_id,
    list_memories,
)


EXTRACT_PROMPT = """你是记忆管理模块。从【对话】中抽取值得长期记住的用户信息。

输出 JSON（不要解释、不要 markdown 代码块）：
{{"items": [
  {"action": "ADD", "kind": "preference", "entity": "莫奈",
   "content": "用户特别喜欢莫奈的睡莲系列", "importance": 0.8},
  {"action": "NOOP", "kind": "fact", "entity": "", "content": "", "importance": 0.0}
]}}

规则：
1. 只抽稳定事实/偏好：常住城市、审美/绘画偏好、职业、家人、长期目标等；
   不抽一次性任务、临时请求、寒暄、过程性回答。
2. 用户没有明确说"记住/喜欢"也可以抽（对标成熟 Agent 的自动记忆），
   但置信度低时宁可 NOOP。
3. 敏感信息一律不抽：身份证号、银行卡/信用卡、密码、验证码、健康诊断、
   宗教/政治立场等。
4. action：新信息 → ADD；与已有记忆相反/变更 → UPDATE；不明确 → NOOP。
5. kind：偏好 → preference，客观事实 → fact，用户画像 → profile。
6. content 用第三人称陈述句（"用户住在上海"），中文，单条一个事实；
   entity 填关键实体（画家/风格/城市/主题等），没有就填空串。
7. importance 0-1：越稳定、越影响后续对话越高（默认 0.5）。
8. 如果对话中用户已明确说"记住/记下"且该内容已由系统保存（或本回合
   即将保存），不要重复抽取 → NOOP；自动抽取只补"用户没说但值得记"的内容。

【对话】
{conversation}"""


_SENSITIVE_RE = re.compile(
    r"\d{15,19}|密码|身份证|银行卡|信用卡|验证码|账号|口令|社保|护照号",
    re.IGNORECASE,
)
_TRUTHY = {"1", "true", "yes", "on", "y"}


def _dedup_key(text: str) -> str:
    """抽取/显式记忆共用的去重键：去标点空白 + 去指令词/主语前缀。"""
    t = re.sub(r"[，。！？、,.!?；;：:\s\"'“”]", "", str(text or ""))
    for prefix in ("请记住", "帮我记住", "记一下", "记住", "记下", "请", "用户", "我"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    for prefix in ("用户", "我"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return t


def extract_enabled() -> bool:
    """MEMORY_AUTO_EXTRACT 开关（默认关）。"""
    return os.getenv("MEMORY_AUTO_EXTRACT", "0").strip().lower() in _TRUTHY


def extract_interval() -> int:
    """每 N 轮抽取一次（默认 2，最小 1）。"""
    try:
        return max(1, int(os.getenv("MEMORY_EXTRACT_INTERVAL", "2")))
    except (TypeError, ValueError):
        return 2


def recent_conversation_text(messages, max_entries: int = 6) -> str:
    """取最近 max_entries 条用户/助手消息（各截 200 字）作为抽取输入。"""
    lines: list[str] = []
    for msg in messages[-max_entries * 2:]:
        mtype = getattr(msg, "type", "")
        content = str(getattr(msg, "content", "") or "")[:200].strip()
        if not content:
            continue
        if mtype == "human":
            lines.append(f"用户：{content}")
        elif mtype == "ai":
            lines.append(f"助手：{content}")
    return "\n".join(lines[-max_entries:]) or "(无)"


def _parse_items(raw: str) -> list[dict]:
    """鲁棒解析抽取 JSON（容错 markdown fence / 截断）。"""
    if not raw:
        return []
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    try:
        data = json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(cleaned[start : end + 1])
        except Exception:
            return []
    items = data.get("items") if isinstance(data, dict) else data
    return items if isinstance(items, list) else []


def extract_memories(
    conversation: str,
    llm: Optional[Callable[[str], str]] = None,
) -> list[dict]:
    """调用 LLM 抽取记忆条目；任何失败/畸形输出返回空列表（安全 NOOP）。"""
    # 用 replace 而非 .format：提示词内含 JSON 花括号示例，format 会误当占位符
    prompt = EXTRACT_PROMPT.replace("{conversation}", conversation)
    if llm is None:
        from src.utils.llm import get_deterministic_llm

        def _default_llm(p: str) -> str:
            return get_deterministic_llm().invoke(p).content

        llm = _default_llm
    try:
        raw = llm(prompt)
    except Exception:  # noqa: BLE001 —— 抽取失败不阻塞主流程
        return []

    out: list[dict] = []
    for item in _parse_items(raw):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "NOOP").strip().upper()
        if action not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            action = "NOOP"
        content = str(item.get("content") or "").strip()
        entity = str(item.get("entity") or "").strip()
        kind = str(item.get("kind") or "preference").strip().lower()
        if kind not in VALID_KINDS:
            kind = "preference"
        try:
            importance = float(item.get("importance") or 0.5)
        except (TypeError, ValueError):
            importance = 0.5
        importance = max(0.0, min(1.0, importance))
        if action in {"ADD", "UPDATE"} and not content:
            continue
        if action == "DELETE" and not entity:
            continue
        if _SENSITIVE_RE.search(f"{content} {entity}"):
            continue
        out.append({
            "action": action,
            "kind": kind,
            "entity": entity,
            "content": content,
            "importance": importance,
        })
    return out


def apply_extracted(
    user_id: str,
    thread_id: Optional[str],
    items: list[dict],
) -> dict:
    """把抽取结果落库（source='extracted'）；冲突沿用同义合并/漂移覆盖。"""
    stats = {"added": 0, "updated": 0, "superseded": 0, "deleted": 0,
             "noop": 0, "skipped": 0, "dup": 0}
    existing = list_memories(user_id, scope="user")
    existing_keys = {_dedup_key(i.get("content")) for i in existing}
    for item in items:
        action = item.get("action", "NOOP")
        if action == "NOOP":
            stats["noop"] += 1
            continue
        if action == "DELETE":
            n = delete_by_entity(user_id, item.get("entity", ""))
            stats["deleted"] += n
            continue
        # 与已有记忆（含用户明确保存）内容一致 → 跳过，避免自动抽取重复
        if _dedup_key(item.get("content")) in existing_keys:
            stats["dup"] += 1
            continue
        try:
            row = add_memory(
                user_id=user_id,
                content=item.get("content", ""),
                kind=item.get("kind", "preference"),
                scope="user",
                entity=item.get("entity") or None,
                thread_id=thread_id,
                source="extracted",
                importance=item.get("importance", 0.5),
            )
        except Exception:  # noqa: BLE001 —— 单条失败不影响其余
            stats["skipped"] += 1
            continue
        act = row.get("action", "")
        if act == "supersede":
            stats["superseded"] += 1
        elif act == "update":
            stats["updated"] += 1
        else:
            stats["added"] += 1
    return stats


def maybe_extract(
    messages,
    user_id: Optional[str] = None,
    extracted_turns: int = 0,
) -> tuple[int, dict]:
    """save_memory 节流入口：到期则抽取并落库，返回 (已抽取轮数, 结果)。

    - 开关关闭 / 未到间隔：直接放行；
    - LLM 失败或畸形：推进计数（避免每轮重试），结果记 error；
    - 落库身份：优先传入的 user_id（save_memory 用 state.user_id），
      否则 get_memory_user_id()（与 remember/recall 工具一致，评估隔离自动生效）。
    """
    human_turns = sum(1 for m in messages if getattr(m, "type", "") == "human")
    if not extract_enabled():
        return extracted_turns, {}
    interval = extract_interval()
    if human_turns - extracted_turns < interval:
        return extracted_turns, {}

    conversation = recent_conversation_text(messages)
    items = extract_memories(conversation)
    user = user_id or get_memory_user_id()
    if not items:
        return human_turns, {"items": [], "error": "parse_or_llm_failed" if not conversation else ""}
    # 抽取结果统一 user scope（跨线程生效，thread_id 不落库）
    stats = apply_extracted(user, None, items)
    return human_turns, {"items": items, "stats": stats}
