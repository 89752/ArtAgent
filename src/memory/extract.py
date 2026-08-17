"""自动抽取层：对话自然推进后抽取稳定事实/偏好（MEMORY_AUTO_EXTRACT 默认开启）。

对标 ChatGPT Dreaming / Claude memory tool / Mem0 extract 的模式：
用户没有说"记住"时，系统在对话自然推进后主动抽取稳定事实与偏好，
落库 source='extracted'，冲突沿用 add_memory 的同义合并/漂移覆盖。

纪律：
- 只抽稳定事实/偏好，不抽一次性任务与寒暄；
- 敏感信息（证件号/密码/银行卡等）不落库；
- LLM 失败不阻塞主流程，且推进节流计数，避免每轮重试。
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Optional

from src.memory.memory_items import (
    VALID_KINDS,
    add_memory,
    delete_by_entity,
    get_memory_user_id,
    list_memories,
)
from src.utils.env import env_flag, env_int
from src.utils.json_utils import parse_json
from src.utils.logging_config import get_logger

logger = get_logger("memory.extract")


EXTRACT_PROMPT = """你是记忆管理模块。从【对话】中抽取值得长期记住的用户信息。

输出 JSON（不要解释、不要 markdown 代码块）：
{{"items": [
  {"action": "ADD", "kind": "preference", "entity": "莫奈",
   "content": "用户特别喜欢莫奈的睡莲系列", "importance": 0.8,
   "scope": "user", "durability": "durable", "authority": "descriptive"},
  {"action": "NOOP", "kind": "fact", "entity": "", "content": "", "importance": 0.0,
   "scope": "thread", "durability": "temporary", "authority": "descriptive"}
]}}

规则：
1. 只抽稳定事实/偏好：常住城市、审美/绘画偏好、职业、家人、长期目标等；
   不抽一次性任务、临时请求、寒暄、过程性回答。
2. 本系统是艺术史助手，以下内容优先抽取（对后续推荐/对比最有价值）：
   - 用户喜欢的画家/流派/时期（如"我喜欢莫奈""偏爱印象派"）；
   - 偏好的色彩/笔触/构图/主题/氛围（如"喜欢宁静的风景""偏好厚涂笔触"）；
   - 明确不喜欢的风格或画家；
   - 收藏/观展/学习方向（如"想了解立体主义"）。
   语言与沟通偏好也要抽（对齐 DeerFlow 的 personalContext）：
   - 语言倾向：主要使用中文/英文/中英混合，或明确要求用某语言回复；
   - 回复风格：希望简洁/详细、专业/口语、是否喜欢结构化输出（列表/表格/图文）；
   - 称呼与语气（如"叫我小鹿就行"）。
   即使只聊了一轮，用户明确表达偏好也要抽（如"梵高的星空很震撼"→ ADD）。
3. 用户没有明确说"记住/喜欢"也可以抽（对标成熟 Agent 的自动记忆），
   但置信度低时宁可 NOOP。
4. 敏感信息一律不抽：身份证号、银行卡/信用卡、密码、验证码、健康诊断、
   宗教/政治立场等。
5. action：新信息 → ADD；与已有记忆相反/变更 → UPDATE；不明确 → NOOP。
6. kind：偏好 → preference，客观事实 → fact，用户画像 → profile。
7. content 用第三人称陈述句（"用户住在上海"），中文，单条一个事实；
   entity 填关键实体（画家/风格/城市/主题等），没有就填空串。
8. importance 0-1：越稳定、越影响后续对话越高（默认 0.5）。
9. 如果对话中用户已明确说"记住/记下"且该内容已由系统保存（或本回合
   即将保存），不要重复抽取 → NOOP；自动抽取只补"用户没说但值得记"的内容。
10. 每条记忆必须给出 scope / durability / authority：
    - scope：user=可跨会话长期使用；thread=仅本次对话相关；project=仅某个项目/任务相关。
      拿不准时用 thread，绝不猜 user；
    - durability：durable=长期成立；temporary=短期/一次性。拿不准用 temporary；
    - authority：descriptive=描述用户；transactional=授权/指令
      （如"帮我删除/发送/提交"）。transactional 一律不落库。
    只有 scope=user、durability=durable、authority=descriptive 的条目才会被系统保存。
11. 可选 expected_valid_days（整数天数）：越稳定越大（如偏好 180-365，
    身份背景 365+）；临时/不确定信息不要给或给短（如 30）。不给表示长期有效。
12. 用户纠正过 Agent 的内容（如"不是，我说的是…""你搞错了""应该是…"），
    如果属于"可复用的用户级纠正"（以后不该再犯），抽为 kind=correction、
    entity="纠正"、importance>=0.9；只针对当前任务的纠正不要记（scope=thread）。

【对话】
{conversation}"""


_SENSITIVE_RE = re.compile(
    r"\d{15,19}|密码|身份证|银行卡|信用卡|验证码|账号|口令|社保|护照号",
    re.IGNORECASE,
)

ALLOWED_SCOPES = {"user", "thread", "project"}
ALLOWED_DURABILITY = {"durable", "temporary"}
ALLOWED_AUTHORITY = {"descriptive", "transactional"}
# 保守默认：字段缺失时按"仅本次对话 / 临时 / 描述性"处理，由写入门控拒掉
_DEFAULT_SCOPE = "thread"
_DEFAULT_DURABILITY = "temporary"
_DEFAULT_AUTHORITY = "descriptive"


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
    """MEMORY_AUTO_EXTRACT 开关（默认开，对齐 DeerFlow 被动记忆）。"""
    return env_flag("MEMORY_AUTO_EXTRACT", default="1")


def extract_interval() -> int:
    """每 N 轮抽取一次（默认 1=每轮，最小 1）。"""
    return max(1, env_int("MEMORY_EXTRACT_INTERVAL", 1))


def extract_debounce_seconds() -> float:
    """异步抽取防抖窗口（默认 2 秒，连续多轮合并为一次 LLM 调用）。"""
    try:
        return max(0.5, float(os.getenv("MEMORY_EXTRACT_DEBOUNCE_SEC", "2")))
    except (TypeError, ValueError):
        return 2.0


def extract_confidence_threshold() -> float:
    """importance 低于该阈值不落库（MEMORY_EXTRACT_CONFIDENCE，默认 0.5）。"""
    try:
        return max(0.0, min(1.0, float(os.getenv("MEMORY_EXTRACT_CONFIDENCE", "0.5"))))
    except (TypeError, ValueError):
        return 0.5


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
    data = parse_json(raw)
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
        scope = str(item.get("scope") or _DEFAULT_SCOPE).strip().lower()
        if scope not in ALLOWED_SCOPES:
            scope = _DEFAULT_SCOPE
        durability = str(item.get("durability") or _DEFAULT_DURABILITY).strip().lower()
        if durability not in ALLOWED_DURABILITY:
            durability = _DEFAULT_DURABILITY
        authority = str(item.get("authority") or _DEFAULT_AUTHORITY).strip().lower()
        if authority not in ALLOWED_AUTHORITY:
            authority = _DEFAULT_AUTHORITY
        try:
            expected_valid_days = int(item.get("expected_valid_days"))
        except (TypeError, ValueError):
            expected_valid_days = None
        if expected_valid_days is not None:
            expected_valid_days = max(1, min(3650, expected_valid_days))
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
            "scope": scope,
            "durability": durability,
            "authority": authority,
            "expected_valid_days": expected_valid_days,
        })
    return out


def gate_items(items: list[dict]) -> tuple[list[dict], list[str]]:
    """确定性写入门控（对齐 DeerMem 的 scope/durability/authority 门）。

    只放行 scope=user + durability=durable + authority=descriptive 且
    importance >= 阈值的 ADD/UPDATE；DELETE/NOOP 不参与内容门控。
    返回 (放行列表, 拒绝原因列表)，拒绝原因形如 "scope:thread"。
    """
    passed: list[dict] = []
    rejected: list[str] = []
    threshold = extract_confidence_threshold()
    for item in items:
        action = item.get("action", "NOOP")
        if action in ("NOOP", "DELETE"):
            passed.append(item)
            continue
        if item.get("scope") != "user":
            rejected.append(f"scope:{item.get('scope')}")
            continue
        if item.get("durability") != "durable":
            rejected.append(f"durability:{item.get('durability')}")
            continue
        if item.get("authority") != "descriptive":
            rejected.append(f"authority:{item.get('authority')}")
            continue
        try:
            importance = float(item.get("importance") or 0)
        except (TypeError, ValueError):
            importance = 0
        if importance < threshold:
            rejected.append(f"confidence:{importance}")
            continue
        passed.append(item)
    return passed, rejected

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
                expected_valid_days=item.get("expected_valid_days"),
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
    raw_items = extract_memories(conversation)
    items, rejected = gate_items(raw_items)
    user = user_id or get_memory_user_id()
    from src.memory.metrics import record_extraction_metrics

    err = "parse_or_llm_failed" if not conversation else ""
    metrics = record_extraction_metrics(
        user, len(raw_items), len(items), rejected, error=err,
    )
    if not items:
        return human_turns, {
            "items": [],
            "rejected": rejected,
            "error": err,
            "metrics": metrics,
        }
    # 抽取结果统一 user scope（跨线程生效，thread_id 不落库）
    stats = apply_extracted(user, None, items)
    return human_turns, {"items": items, "stats": stats, "metrics": metrics}


# ── 异步防抖抽取（对齐 DeerFlow DeerMem 的被动更新队列） ─────────
_extract_lock = threading.Lock()
_pending: dict[str, dict] = {}
_worker_started = False


def _extract_worker() -> None:
    """后台线程：取一个待抽取任务 → 防抖等待 → 落库；循环直到队列空。"""
    global _worker_started
    while True:
        with _extract_lock:
            if not _pending:
                _worker_started = False
                return
            user_id, task = _pending.popitem()
        # 防抖窗口内如果来了新任务，旧任务让位给新任务（合并连续轮次）
        time.sleep(extract_debounce_seconds())
        with _extract_lock:
            if _pending.get(user_id) is not None:
                continue
        try:
            _turns, result = maybe_extract(
                task["messages"], task["user_id"], extracted_turns=0
            )
            lang_result = record_language_preference(
                task["messages"], task["user_id"]
            )
            stats = (result or {}).get("stats") or {}
            changed = sum(
                stats.get(k, 0)
                for k in ("added", "updated", "superseded", "deleted")
            )
            if lang_result.get("added"):
                changed += 1
            if changed:
                from src.memory.user_doc import update_user_doc

                doc_result = update_user_doc(task["user_id"], task["messages"])
                if doc_result.get("updated"):
                    from src.memory.profile import sync_profile_item_from_doc

                    sync_profile_item_from_doc(task["user_id"])
                from src.memory.lifecycle import maybe_maintenance

                maybe_maintenance(task["user_id"])
        except Exception:  # noqa: BLE001 —— 后台抽取失败不影响主流程
            logger.exception("memory extract worker failed: user=%s", user_id)


def _detect_language(texts: list[str]) -> str | None:
    """按字符分布确定性判断用户主要语言（不依赖 LLM）。"""
    joined = "".join(texts or [])
    cjk = sum(1 for ch in joined if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in joined if ch.isascii() and ch.isalpha())
    if cjk == 0 and latin == 0:
        return None
    if cjk and latin:
        return "中英混合"
    return "中文" if cjk else "英文"


def record_language_preference(messages, user_id: str) -> dict:
    """把用户消息的语言倾向确定性落库（对齐 DeerFlow personalContext）。

    与 LLM 抽取解耦：只要本轮出现中文/英文，就保证有一条语言记忆，
    已存在则跳过。返回 {"added": "中文"} 或 {"skipped": "exists"}。
    """
    texts = [
        str(getattr(m, "content", "") or "")
        for m in messages
        if getattr(m, "type", "") == "human"
    ]
    lang = _detect_language(texts)
    if not lang:
        return {}
    content = {
        "中文": "用户主要使用中文交流",
        "英文": "用户主要使用英文交流",
        "中英混合": "用户中英混合使用交流",
    }[lang]
    existing = list_memories(user_id, scope="user")
    if any(_dedup_key(i.get("content")) == _dedup_key(content) for i in existing):
        return {"skipped": "exists"}
    add_memory(
        user_id=user_id,
        content=content,
        kind="preference",
        entity="语言",
        scope="user",
        source="extracted",
        importance=0.7,
    )
    return {"added": lang}


def schedule_extract(messages, user_id: Optional[str] = None) -> dict:
    """把本轮对话交给后台防抖抽取（不阻塞响应）；返回调度结果。

    与 DeerFlow 一致：抽取是异步的、每轮被动触发；开关关闭或没有用户
    消息时直接放行。落库由 worker 线程完成，重复内容由去重兜底。
    """
    global _worker_started
    user = user_id or get_memory_user_id()
    if not extract_enabled():
        return {}
    human_turns = sum(1 for m in messages if getattr(m, "type", "") == "human")
    if human_turns < 1:
        return {}
    with _extract_lock:
        _pending[user] = {"messages": list(messages), "user_id": user}
        if not _worker_started:
            _worker_started = True
            threading.Thread(
                target=_extract_worker,
                daemon=True,
                name="memory-extract",
            ).start()
    return {"scheduled": True, "pending": len(_pending)}


def shutdown_flush(timeout: float = 30.0) -> bool:
    """优雅退出：等待后台抽取队列清空、worker 退出（最多 timeout 秒）。

    对齐 DeerMem 的 shutdown_flush：避免进程退出时把防抖缓冲里的
    记忆更新丢掉。返回 True 表示队列已清空；超时返回 False。
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        with _extract_lock:
            idle = not _pending and not _worker_started
        if idle:
            return True
        time.sleep(0.1)
    return False
