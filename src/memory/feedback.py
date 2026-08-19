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

from src.data import db

_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "feedback.db"

_lock = threading.Lock()
_db_ready = False


def _get_conn() -> sqlite3.Connection:
    global _db_ready
    conn = db.get_conn(_DB_PATH)
    if not _db_ready:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL DEFAULT 'web_user',
                session_id TEXT NOT NULL,
                rating     INTEGER NOT NULL CHECK(rating IN (1, -1)),
                reason     TEXT NOT NULL DEFAULT '',
                comment    TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()}
        if "user_id" not in cols:
            conn.execute(
                "ALTER TABLE feedback ADD COLUMN user_id TEXT NOT NULL DEFAULT 'web_user'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id, id)"
        )
        conn.commit()
        _db_ready = True
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_feedback(
    session_id: str,
    rating: int,
    reason: str = "",
    comment: str = "",
    user_id: str = "web_user",
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
            INSERT INTO feedback (user_id, session_id, rating, reason, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, str(session_id)[:128], rating, reason, comment, _now()),
        )
        conn.commit()
    return int(cur.lastrowid)


def list_feedback(
    limit: int = 100,
    offset: int = 0,
    user_id: str = "web_user",
) -> tuple[list[dict], int]:
    """按时间倒序返回 (反馈列表, 总数)，供导出/人工审核。"""
    limit = min(max(1, int(limit)), 1000)
    offset = max(0, int(offset))
    with _lock:
        conn = _get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, session_id, rating, reason, comment, created_at
            FROM feedback WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
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


def delete_user_feedback(user_id: str) -> int:
    """删除某用户全部反馈；返回删除条数（级联删除用）。"""
    if not user_id:
        return 0
    with _lock:
        cur = _get_conn().execute(
            "DELETE FROM feedback WHERE user_id = ?", (user_id,)
        )
        _get_conn().commit()
        return cur.rowcount


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
    global _db_ready, _DB_PATH
    db.close_all()
    _db_ready = False
    _DB_PATH = path or Path("./data/memory/_test_feedback.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
