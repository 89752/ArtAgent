"""Agent 主动长期记忆（Phase 4）：remember / recall / forget 的存储层。

SQLite 关键词匹配（按规划：先不建向量记忆库，验证价值后再升级）。
表 agent_memory(user_id, key, content, updated_at)，同偏好库目录。
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"
_DB_PATH = _DB_DIR / "agent_memory.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

DEFAULT_USER = "default_user"


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_memory (
                user_id    TEXT NOT NULL,
                key        TEXT NOT NULL,
                content    TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        _conn.commit()
    return _conn


def remember(user_id: str, key: str, fact: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _get_conn().execute(
            """
            INSERT INTO agent_memory (user_id, key, content, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                content = excluded.content, updated_at = excluded.updated_at
            """,
            (user_id, key.strip(), fact.strip(), now),
        )
        _get_conn().commit()


def recall(user_id: str, query: str, limit: int = 5) -> list[dict]:
    pattern = f"%{query.strip()}%"
    try:
        rows = _get_conn().execute(
            """
            SELECT key, content, updated_at FROM agent_memory
            WHERE user_id = ? AND (key LIKE ? OR content LIKE ?)
            ORDER BY updated_at DESC LIMIT ?
            """,
            (user_id, pattern, pattern, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"key": r[0], "content": r[1], "updated_at": r[2]}
        for r in rows
    ]


def forget(user_id: str, key: str) -> bool:
    with _lock:
        cur = _get_conn().execute(
            "DELETE FROM agent_memory WHERE user_id = ? AND key = ?",
            (user_id, key.strip()),
        )
        _get_conn().commit()
        return cur.rowcount > 0
