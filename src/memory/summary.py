"""会话滚动摘要（Phase 4）：conversation_summary 表 + 增量摘要器。

设计（借鉴 ragent JdbcConversationMemorySummaryService）：
- conversations.db 增加 conversation_summary 表；
- 增量式：达到触发轮数后才摘要"最近窗口 + 旧摘要"的合并结果，
  并记录已摘要轮数，避免每轮全量重算；
- 摘要由 save_memory 节点调用，注入 context.summary 块。
"""

from __future__ import annotations

import os
import sqlite3
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "conversations.db"
_LEGACY_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "data" / "memory"
    / "conversations.db"
) if not os.getenv("ARTAGENT_MEMORY_DIR") else None

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

SUMMARY_TRIGGER_TURNS = 8   # 达到该轮数才触发摘要
SUMMARY_WINDOW_TURNS = 6    # 每次并入最近 6 轮
SUMMARY_MAX_CHARS = 300
VOLUME_TRIGGER_CHARS = 15000  # 上下文体积超限也触发（v2 阈值触发）


def _get_conn() -> sqlite3.Connection:
    global _conn, _DB_PATH
    if _conn is None:
        _DB_PATH = _migrate_legacy_db()
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summary (
                conversation_id TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                content         TEXT NOT NULL,
                summarized_turns INTEGER NOT NULL DEFAULT 0,
                updated_at      TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def _migrate_legacy_db() -> Path:
    """旧路径 data/data/memory/conversations.db → data/memory/（一次性迁移）。"""
    if _LEGACY_DB_PATH is None or _DB_PATH.exists() or not _LEGACY_DB_PATH.exists():
        return _DB_PATH
    try:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_LEGACY_DB_PATH), str(_DB_PATH))
        return _DB_PATH
    except OSError:
        return _LEGACY_DB_PATH


def load_summary(conversation_id: str) -> str:
    if not conversation_id:
        return ""
    try:
        row = _get_conn().execute(
            "SELECT content FROM conversation_summary WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return ""
    return (row[0] if row else "") or ""


def _save_summary(conversation_id: str, user_id: str, content: str,
                  summarized_turns: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _get_conn().execute(
            """
            INSERT INTO conversation_summary
                (conversation_id, user_id, content, summarized_turns, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                content = excluded.content,
                summarized_turns = excluded.summarized_turns,
                updated_at = excluded.updated_at
            """,
            (conversation_id, user_id, content[:SUMMARY_MAX_CHARS], summarized_turns, now),
        )
        _get_conn().commit()


def _human_turn_count(messages) -> int:
    return sum(1 for m in messages or [] if getattr(m, "type", "") == "human")


def _recent_turns_text(messages, window: int = SUMMARY_WINDOW_TURNS) -> str:
    """取最近 window 轮的 用户/助手 消息文本。"""
    pairs: list[str] = []
    for m in messages or []:
        mtype = getattr(m, "type", "")
        if mtype not in ("human", "ai"):
            continue
        content = str(getattr(m, "content", "") or "")[:200]
        if mtype == "human":
            pairs.append(f"用户：{content}")
        else:
            pairs.append(f"助手：{content}")
    return "\n".join(pairs[-window * 2 :])


def _summarize(existing: str, recent_text: str,
               llm: Optional[Callable[[str], str]] = None) -> str:
    """把旧摘要 + 最近对话压缩成新摘要（纯函数，llm 可注入）。"""
    if llm is None:
        from src.utils.llm import get_deterministic_llm

        def _default(prompt: str) -> str:
            return get_deterministic_llm().invoke(prompt).content

        llm = _default
    prompt = (
        "你是会话记忆压缩器。合并【已有摘要】与【新增对话】，输出更新后的摘要。\n"
        f"要求：{SUMMARY_MAX_CHARS} 字以内；保留用户喜欢的画家/风格、讨论过的具体画作、"
        "用户明确表达的偏好；丢弃寒暄与过程细节；只输出摘要本身。\n\n"
        f"【已有摘要】\n{existing or '(无)'}\n\n"
        f"【新增对话】\n{recent_text}\n\n更新后的摘要："
    )
    return str(llm(prompt)).strip()[:SUMMARY_MAX_CHARS]


def maybe_summarize(
    messages,
    conversation_id: str,
    user_id: str,
    llm: Optional[Callable[[str], str]] = None,
    volume_chars: int = 0,
) -> str:
    """达到触发轮数且新增对话足够时，增量摘要并落库；返回新摘要（无则空串）。"""
    if not conversation_id:
        return ""
    turns = _human_turn_count(messages)
    # v2：轮数 OR 上下文体积超限都触发（成熟平台"接近上限自动压缩"）
    triggered_by_turns = turns >= SUMMARY_TRIGGER_TURNS
    triggered_by_volume = volume_chars > VOLUME_TRIGGER_CHARS
    if not triggered_by_turns and not triggered_by_volume:
        return ""
    existing = load_summary(conversation_id)
    summarized_turns = _summarized_turns_of(conversation_id)
    delta = turns - summarized_turns
    if delta < 1:
        return existing  # 增量不足，复用旧摘要
    if triggered_by_turns and delta < SUMMARY_WINDOW_TURNS:
        return existing  # 轮数触发要求满窗口；体积触发不要求（首次即可压缩）
    new_summary = _summarize(existing, _recent_turns_text(messages), llm)
    if new_summary:
        _save_summary(conversation_id, user_id, new_summary, turns)
    return new_summary


def _summarized_turns_of(conversation_id: str) -> int:
    try:
        row = _get_conn().execute(
            "SELECT summarized_turns FROM conversation_summary WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0
