"""用户、API Key 与每用户设置（P0-1）。

首版最小用户模型：users + api_keys + user_settings。
- users：平台身份，delete_user 级联清空会话/偏好/摘要/文档/向量/文件。
- api_keys：静态 token（secrets.token_hex(24)），供 /v1 鉴权。
- user_settings：每用户生效数据源，避免全局 active_dataset 串户。

数据落在 data/platform/platform.db（ARTAGENT_PLATFORM_DIR 可覆盖，测试隔离用）。
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger("platform.users")

DEFAULT_USER_ID = "web_user"  # 旧单用户模式的兼容身份

_DB_DIR = Path(os.getenv(
    "ARTAGENT_PLATFORM_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "platform"),
))
_DB_PATH = _DB_DIR / "platform.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                key        TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                label      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id    TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL DEFAULT 'core'
            )
            """
        )
        _conn.commit()
    return _conn


def init_db() -> None:
    """建表并确保默认 web_user 存在（旧单用户数据归属）。"""
    _get_conn()
    ensure_default_user()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_default_user() -> dict:
    """返回默认 web_user；不存在则创建（兼容旧数据与本地 UI）。"""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (DEFAULT_USER_ID,)
        ).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
            (DEFAULT_USER_ID, "本地默认用户", _now()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, dataset_id) VALUES (?, 'core')",
            (DEFAULT_USER_ID,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (DEFAULT_USER_ID,)
        ).fetchone()
        return dict(row)


def create_user(name: str, api_key: str | None = None, label: str = "default") -> dict:
    """创建用户并生成一个 API Key；返回 {user, api_key}。"""
    name = (name or "").strip()[:60] or f"user-{secrets.token_hex(3)}"
    user_id = f"u_{secrets.token_hex(8)}"
    key = api_key or f"sk-{secrets.token_hex(24)}"
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
            (user_id, name, _now()),
        )
        conn.execute(
            "INSERT INTO api_keys (key, user_id, label, created_at) VALUES (?, ?, ?, ?)",
            (key, user_id, label[:40], _now()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, dataset_id) VALUES (?, 'core')",
            (user_id,),
        )
        conn.commit()
    logger.info("[users] 已创建用户 %s (%s)", user_id, name)
    return {"user": get_user(user_id), "api_key": key}


def get_user(user_id: str) -> Optional[dict]:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with _lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def create_api_key(user_id: str, label: str = "default") -> str:
    """为用户补发一个 Key（不吊销旧 Key）。"""
    key = f"sk-{secrets.token_hex(24)}"
    with _lock:
        conn = _get_conn()
        if conn.execute(
            "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
        ).fetchone() is None:
            raise KeyError(f"用户不存在：{user_id}")
        conn.execute(
            "INSERT INTO api_keys (key, user_id, label, created_at) VALUES (?, ?, ?, ?)",
            (key, user_id, label[:40], _now()),
        )
        conn.commit()
    return key


def list_api_keys(user_id: str) -> list[dict]:
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT key, label, created_at FROM api_keys WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_api_key(key: str) -> bool:
    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM api_keys WHERE key = ?", (key,))
        conn.commit()
    return cur.rowcount > 0


def get_user_by_api_key(api_key: str) -> Optional[dict]:
    """Key → user；未知 Key 返回 None。"""
    if not api_key:
        return None
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            """
            SELECT u.* FROM api_keys k JOIN users u ON u.user_id = k.user_id
            WHERE k.key = ?
            """,
            (api_key,),
        ).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------ #
# 每用户设置：生效数据源（避免全局 active_dataset 串户）              #
# ------------------------------------------------------------------ #


def get_user_dataset(user_id: str) -> str:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT dataset_id FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["dataset_id"] if row else "core"


def set_user_dataset(user_id: str, dataset_id: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO user_settings (user_id, dataset_id) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET dataset_id = excluded.dataset_id
            """,
            (user_id, dataset_id),
        )
        conn.commit()


# ------------------------------------------------------------------ #
# 级联删除                                                            #
# ------------------------------------------------------------------ #


def delete_user(user_id: str, cascade: bool = True) -> dict:
    """删除用户；cascade=True 时清空其会话/偏好/摘要/文档（向量+文件）。"""
    if get_user(user_id) is None:
        raise KeyError(f"用户不存在：{user_id}")

    result: dict = {"user_id": user_id, "sessions": 0, "preferences": 0,
                    "summaries": 0, "documents": 0, "api_keys": 0}
    if cascade:
        # 会话/偏好/摘要（级联接口依赖平台集成进度；缺失时跳过并告警）
        try:
            from src.memory.conversations import delete_user_conversations

            result["sessions"] = delete_user_conversations(user_id)
        except ImportError:
            logger.warning(
                "[users] conversations 用户隔离接口尚未集成，跳过会话级联（P0-1 待续）"
            )
        try:
            from src.memory.store import clear_preferences
            from src.memory.summary import delete_user_summaries

            clear_preferences(user_id)
            result["summaries"] = delete_user_summaries(user_id)
        except ImportError:
            logger.warning("[users] 偏好/摘要级联接口缺失，跳过")

        # 文档：逐个走 service.delete_document（连带 Chroma 向量与上传文件）
        try:
            from src.data import documents_store
            from web.service import delete_document as service_delete_document

            docs = documents_store.list_documents(user_id=user_id)
            for doc in docs:
                try:
                    service_delete_document(doc["doc_id"], user_id=user_id)
                    result["documents"] += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[users] 删除用户文档失败 %s: %s", doc.get("doc_id"), e
                    )
        except ImportError:
            logger.warning("[users] service 文档删除接口尚未就绪，跳过文档级联")

    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
        result["api_keys"] = cur.rowcount
        conn.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    logger.info("[users] 已删除用户 %s：%s", user_id, result)
    return result


def _reset_for_tests(path: Path | None = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _conn, _DB_PATH
    _conn = None
    _DB_PATH = path or Path("./data/platform/_test_platform.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
