"""用户反馈闭环：feedback 表 + 读写 + 导出。

落库 data/memory/feedback.db（ARTAGENT_MEMORY_DIR 可覆盖，测试隔离用）。
rating: 1 = 👍 / -1 = 👎；reason 来自前端原因标签，comment 为补充文字。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "feedback.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                rating     INTEGER NOT NULL CHECK(rating IN (1, -1)),
                reason     TEXT NOT NULL DEFAULT '',
                comment    TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_feedback(
    session_id: str,
    rating: int,
    reason: str = "",
    comment: str = "",
) -> int:
    """写入一条反馈；返回自增 id。rating 仅接受 1 / -1。"""
    rating = int(rating)
    if rating not in (1, -1):
        raise ValueError("rating 必须是 1（赞）或 -1（踩）")
    reason = (reason or "").strip()[:40]
    comment = (comment or "").strip()[:500]
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """
            INSERT INTO feedback (session_id, rating, reason, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(session_id)[:128], rating, reason, comment, _now()),
        )
        conn.commit()
    return int(cur.lastrowid)


def list_feedback(
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """按时间倒序返回 (反馈列表, 总数)，供导出/人工审核。"""
    limit = min(max(1, int(limit)), 1000)
    offset = max(0, int(offset))
    with _lock:
        conn = _get_conn()
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, session_id, rating, reason, comment, created_at
            FROM feedback ORDER BY id DESC LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return (
        [
            {
                "id": r[0], "session_id": r[1], "rating": r[2],
                "reason": r[3], "comment": r[4], "created_at": r[5],
            }
            for r in rows
        ],
        total,
    )


def export_feedback(path: Path, limit: int = 10000) -> int:
    """导出为 JSONL（eval 候选池），返回导出条数。"""
    items, _ = list_feedback(limit=limit)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def count_feedback(rating: Optional[int] = None) -> int:
    with _lock:
        conn = _get_conn()
        if rating is None:
            return int(conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = ?", (int(rating),)
            ).fetchone()[0]
        )


def _reset_for_tests(path: Path | None = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _conn, _DB_PATH
    _conn = None
    _DB_PATH = path or Path("./data/memory/_test_feedback.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
