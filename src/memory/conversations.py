"""多轮会话历史持久化（conversations 表）。

存储 Web 界面的完整历史对话（含渲染好的 HTML 气泡），支持左侧
「历史对话」列表的真实切换与续聊。与摘要/记忆职责不同：
- conversations：完整消息流，供回看/续聊；
- summary（conversation_summary）：压缩后的滚动摘要，供上下文注入。

设计要点：
- 一张表 conversations(session_id, title, messages_json, updated_at)；
- messages_json 存前端气泡列表（含内联图片/折叠思考的 HTML）；
- list 按 updated_at 降序，供侧栏展示。
"""

import json
import os
import shutil
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone

# 与偏好库同目录：data/memory/conversations.db（P1 目录收敛）
# 测试可通过 ARTAGENT_MEMORY_DIR 覆盖，避免污染真实数据
_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "conversations.db"
# 旧路径（历史笔误多了一层 data/）：存在则迁移一次
_LEGACY_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "data" / "memory"
    / "conversations.db"
) if not os.getenv("ARTAGENT_MEMORY_DIR") else None

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

DEFAULT_USER_ID = "web_user"


def _get_conn() -> sqlite3.Connection:
    """返回全局单例连接，首次调用时建表。"""
    global _conn, _DB_PATH
    if _conn is None:
        _DB_PATH = _migrate_legacy_db()
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL DEFAULT 'web_user',
                title         TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
            """
        )
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "user_id" not in cols:
            _conn.execute(
                "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'web_user'"
            )
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user "
            "ON conversations(user_id, updated_at)"
        )
        _conn.commit()
    return _conn


def _migrate_legacy_db() -> Path:
    """旧路径 data/data/memory/conversations.db → data/memory/（一次性迁移）。

    返回最终应使用的 DB 路径（迁移失败时回退旧路径）。
    """
    if _LEGACY_DB_PATH is None or _DB_PATH.exists() or not _LEGACY_DB_PATH.exists():
        return _DB_PATH
    try:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_LEGACY_DB_PATH), str(_DB_PATH))
        return _DB_PATH
    except OSError:
        return _LEGACY_DB_PATH


def save_conversation(
    session_id: str,
    title: str,
    messages: list[dict],
    user_id: str = DEFAULT_USER_ID,
) -> None:
    """写入/更新一条会话。title 取首条用户消息，messages 为完整气泡列表。"""
    if not session_id or not messages:
        return
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(messages, ensure_ascii=False)
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO conversations (session_id, user_id, title, messages_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET title = excluded.title,
                          messages_json = excluded.messages_json,
                          updated_at = excluded.updated_at
            """,
            (session_id, user_id, title[:60], payload, now),
        )
        conn.commit()


def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user_id: str = DEFAULT_USER_ID,
) -> tuple[list[dict], int]:
    """按最近更新降序返回 (会话列表, 总数)，供侧栏分页渲染。"""
    with _lock:
        conn = _get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT session_id, title, updated_at FROM conversations
            WHERE user_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        ).fetchall()
    return (
        [{"session_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows],
        total,
    )


def rename_conversation(
    session_id: str,
    title: str,
    user_id: str = DEFAULT_USER_ID,
) -> bool:
    """重命名会话标题；返回是否找到并更新。"""
    if not session_id or not title:
        return False
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE conversations SET title = ? WHERE session_id = ? AND user_id = ?",
            (title[:60], session_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def load_conversation(
    session_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> list[dict]:
    """读取某会话的完整消息列表；不存在时返回 []。"""
    if not session_id:
        return []
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT messages_json FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    if not row:
        return []
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return []


def delete_conversation(
    session_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> None:
    """删除单条会话。"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        conn.commit()


def delete_user_conversations(user_id: str) -> int:
    """删除某用户全部会话；返回删除条数（级联删除用）。"""
    if not user_id:
        return 0
    with _lock:
        cur = _get_conn().execute(
            "DELETE FROM conversations WHERE user_id = ?", (user_id,)
        )
        _get_conn().commit()
        return cur.rowcount


def remove_attachment_from_all(
    doc_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> int:
    """从所有会话历史中移除引用某文档的附件记录；返回受影响会话数。

    文档删除后调用，避免会话里残留指向已删除文档的附件卡片。
    """
    if not doc_id:
        return 0
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT session_id, messages_json FROM conversations WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        now = datetime.now(timezone.utc).isoformat()
        changed = 0
        for session_id, payload in rows:
            try:
                messages = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            new_msgs = [
                m for m in messages
                if not (m.get("role") == "attachment" and m.get("doc_id") == doc_id)
            ]
            if len(new_msgs) == len(messages):
                continue
            if new_msgs:
                title = next(
                    (m["content"] for m in new_msgs if m["role"] == "user"), "新对话"
                )
                conn.execute(
                    """
                    UPDATE conversations
                    SET title = ?, messages_json = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (title[:60], json.dumps(new_msgs, ensure_ascii=False), now, session_id),
                )
            else:
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )
            changed += 1
        conn.commit()
    return changed


def relative_time(iso_ts: str) -> str:
    """把 ISO 时间戳转成「刚刚 / N分钟前 / N小时前 / 昨天 / N天前 / 日期」。"""
    try:
        then = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - then
    secs = delta.total_seconds()
    if secs < 60:
        return "刚刚"
    if secs < 3600:
        return f"{int(secs // 60)} 分钟前"
    if secs < 86400:
        return f"{int(secs // 3600)} 小时前"
    days = int(secs // 86400)
    if days == 1:
        return "昨天"
    if days < 7:
        return f"{days} 天前"
    return then.astimezone().strftime("%m-%d")
