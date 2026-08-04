"""记忆系统 Phase 2（L3）：情景记忆 memory_episodes。

对标 ChatGPT/Claude 的"上次我们聊过…"：按 (user_id, conversation_id) 保存
会话滚动摘要，同一会话回来时注入上下文，跨轮/跨会话可回顾。

- 与 memory_items 共用 data/memory/agent_memory.db（表独立）；
- upsert：同一会话只保留最新摘要（turn_count 递增）；
- 用户隔离：所有读写强制带 user_id（评估用 eval-test 自动隔离）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.memory.memory_items import _get_conn


EPISODE_MAX_CHARS = 1200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_table() -> None:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_episodes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            summary         TEXT NOT NULL,
            turn_count      INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            UNIQUE (user_id, conversation_id)
        )
        """
    )
    conn.commit()


_EPISODE_COLS = (
    "id", "user_id", "conversation_id", "summary",
    "turn_count", "created_at", "updated_at",
)


def _row_to_dict(row) -> dict:
    if hasattr(row, "keys"):
        return dict(row)
    return dict(zip(_EPISODE_COLS, row))


def upsert_episode(
    user_id: str,
    conversation_id: str,
    summary: str,
    turn_count: int = 0,
) -> dict:
    """保存/更新某个会话的情景摘要；返回落库行。"""
    if not user_id or not conversation_id:
        raise ValueError("user_id 与 conversation_id 不能为空")
    summary = (summary or "").strip()[:EPISODE_MAX_CHARS]
    if not summary:
        raise ValueError("摘要不能为空")
    _ensure_table()
    now = _now()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO memory_episodes
          (user_id, conversation_id, summary, turn_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, conversation_id) DO UPDATE SET
          summary = excluded.summary,
          turn_count = excluded.turn_count,
          updated_at = excluded.updated_at
        """,
        (user_id, conversation_id, summary, max(0, int(turn_count)), now, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM memory_episodes WHERE user_id = ? AND conversation_id = ?",
        (user_id, conversation_id),
    ).fetchone()
    return _row_to_dict(row) if row else {}


def load_episode(user_id: str, conversation_id: str) -> Optional[dict]:
    """读取某会话的最新情景摘要（无则 None）。"""
    if not user_id or not conversation_id:
        return None
    _ensure_table()
    row = _get_conn().execute(
        "SELECT * FROM memory_episodes WHERE user_id = ? AND conversation_id = ?",
        (user_id, conversation_id),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_episodes(user_id: str, limit: int = 10) -> list[dict]:
    """按更新时间倒序列出某用户的情景摘要（供 UI/诊断）。"""
    if not user_id:
        return []
    _ensure_table()
    rows = _get_conn().execute(
        "SELECT * FROM memory_episodes WHERE user_id = ? "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, max(1, int(limit))),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def clear_user_episodes(user_id: str) -> int:
    """清空某用户全部情景摘要（评估用例前清场用）。"""
    _ensure_table()
    cur = _get_conn().execute(
        "DELETE FROM memory_episodes WHERE user_id = ?", (user_id,)
    )
    _get_conn().commit()
    return cur.rowcount
