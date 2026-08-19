"""用户、API Key 与每用户设置（规划中）。

首版最小用户模型：users + api_keys + user_settings。
- users：平台身份，delete_user 级联清空会话/偏好/摘要/文档/向量/文件。
- api_keys：静态 token（secrets.token_hex(24)），供 /v1 鉴权。
- user_settings：每用户生效数据源，避免全局 active_dataset 串户。

数据落在 data/platform/platform.db（ARTAGENT_PLATFORM_DIR 可覆盖，测试隔离用）。
"""

from __future__ import annotations

import os
import hashlib
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data import db
from src.utils.logging_config import get_logger

logger = get_logger("platform.users")

DEFAULT_USER_ID = "web_user"  # 旧单用户模式的兼容身份

_DB_DIR = Path(os.getenv(
    "ARTAGENT_PLATFORM_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "platform"),
))
_DB_PATH = _DB_DIR / "platform.db"

_lock = threading.Lock()
_db_ready = False


def _get_conn() -> sqlite3.Connection:
    global _db_ready
    conn = db.get_conn(_DB_PATH, row_factory=sqlite3.Row)
    if not _db_ready:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                username   TEXT UNIQUE,
                password_hash TEXT,
                is_admin   INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "username" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN username TEXT UNIQUE")
        if "password_hash" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        if "is_admin" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                key        TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                label      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id    TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL DEFAULT 'core'
            )
            """
        )
        conn.commit()
        _db_ready = True
    return conn


def init_db() -> None:
    """建表并确保默认 web_user 存在（旧单用户数据归属）。

    默认管理员账号（user/11111111）不再自动创建：共享部署不预设任何
    已知凭据，如需保留旧单机体验，设置 ARTAGENT_SEED_DEFAULT_ACCOUNT=1。
    """
    _get_conn()
    ensure_default_user()
    if os.getenv("ARTAGENT_SEED_DEFAULT_ACCOUNT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        ensure_default_account()


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}:{digest}"


def _verify_password(password: str, stored: str) -> bool:
    if not stored or ":" not in stored:
        return False
    salt, digest = stored.split(":", 1)
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() == digest


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")
_RESERVED_USERNAMES = {"user", "web_user", "admin", "root", "system"}
PASSWORD_MIN_LEN = 8


def validate_username(username: str) -> str:
    """自助注册用用户名校验：3-40 位字母/数字/._-，禁保留名。"""
    u = (username or "").strip()
    if not _USERNAME_RE.match(u):
        raise ValueError("用户名需为 3-40 位，仅限字母、数字、下划线、点或短横线")
    if u.lower() in _RESERVED_USERNAMES:
        raise ValueError("该用户名不可用，请换一个")
    return u


def validate_password(password: str) -> str:
    """自助注册/改密用密码强度校验。"""
    if not password or len(password) < PASSWORD_MIN_LEN:
        raise ValueError(f"密码至少 {PASSWORD_MIN_LEN} 位")
    if len(password) > 128:
        raise ValueError("密码过长（最多 128 位）")
    return password


def register_user(username: str, password: str, name: str = "") -> dict:
    """自助注册：校验用户名/密码 → 建号 → 返回 {user, api_key}。"""
    username = validate_username(username)
    password = validate_password(password)
    name = (name or "").strip()[:60] or username
    uid = f"u_{secrets.token_hex(8)}"
    key = f"sk-{secrets.token_hex(24)}"
    with _lock:
        conn = _get_conn()
        if conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone():
            raise KeyError(f"用户名已存在：{username}")
        conn.execute(
            """
            INSERT INTO users (user_id, name, username, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (uid, name, username, _hash_password(password), _now()),
        )
        conn.execute(
            "INSERT INTO api_keys (key, user_id, label, created_at) VALUES (?, ?, 'default', ?)",
            (key, uid, _now()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, dataset_id) VALUES (?, 'core')",
            (uid,),
        )
        conn.commit()
    logger.info("[users] 自助注册新用户 %s (%s)", uid, username)
    return {"user": get_user(uid), "api_key": key}


def change_password(
    user_id: str,
    old_password: str,
    new_password: str,
    keep_token: str | None = None,
) -> bool:
    """本人修改密码：校验旧密码 → 更新哈希 → 吊销其他会话 token（保留当前）。"""
    user = get_user(user_id)
    if user is None:
        raise KeyError("用户不存在")
    if not _verify_password(old_password or "", user.get("password_hash")):
        raise ValueError("当前密码不正确")
    new_password = validate_password(new_password)
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (_hash_password(new_password), user_id),
        )
        if keep_token:
            conn.execute(
                "DELETE FROM api_keys WHERE user_id = ? AND label = 'session' AND key != ?",
                (user_id, keep_token),
            )
        else:
            conn.execute(
                "DELETE FROM api_keys WHERE user_id = ? AND label = 'session'",
                (user_id,),
            )
        conn.commit()
    logger.info("[users] 用户 %s 已修改密码", user_id)
    return True


def ensure_default_account() -> dict:
    """可选默认登录账号 user / 11111111（旧单机模式的默认管理员）。

    仅当 ARTAGENT_SEED_DEFAULT_ACCOUNT=1 时由 init_db 调用；共享部署
    不应开启（已知凭据即安全风险），改用 manage_users.py 创建管理员。
    """
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", ("user",)
        ).fetchone()
        if row:
            return dict(row)
        user_id = "user"
        conn.execute(
            """
            INSERT INTO users (user_id, name, username, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (user_id, "默认用户", "user", _hash_password("11111111"), _now()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, dataset_id) VALUES (?, 'core')",
            (user_id,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row)


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
            """
            INSERT INTO users (user_id, name, username, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (user_id, name, None, None, _now()),
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


def create_user_with_password(
    name: str,
    username: str,
    password: str,
    user_id: str | None = None,
    is_admin: bool = False,
) -> dict:
    """创建带账号密码的用户；返回 {user, api_key}。"""
    name = (name or "").strip()[:60] or username or f"user-{secrets.token_hex(3)}"
    username = (username or "").strip()[:40] or f"u_{secrets.token_hex(4)}"
    uid = user_id or f"u_{secrets.token_hex(8)}"
    key = f"sk-{secrets.token_hex(24)}"
    with _lock:
        conn = _get_conn()
        if conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone():
            raise KeyError(f"用户名已存在：{username}")
        conn.execute(
            """
            INSERT INTO users (user_id, name, username, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (uid, name, username, _hash_password(password), 1 if is_admin else 0, _now()),
        )
        conn.execute(
            "INSERT INTO api_keys (key, user_id, label, created_at) VALUES (?, ?, ?, ?)",
            (key, uid, "default", _now()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, dataset_id) VALUES (?, 'core')",
            (uid,),
        )
        conn.commit()
    return {"user": get_user(uid), "api_key": key}


def get_user_by_username(username: str) -> Optional[dict]:
    if not username:
        return None
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def verify_login(username: str, password: str) -> Optional[dict]:
    """校验账号密码；成功返回用户，失败返回 None。"""
    user = get_user_by_username((username or "").strip())
    if user is None or not _verify_password(password or "", user.get("password_hash")):
        return None
    return user


def issue_session_token(user_id: str) -> str:
    """为已登录用户签发会话 token（存 api_keys，label='session'）。"""
    token = secrets.token_hex(24)
    with _lock:
        _get_conn().execute(
            "INSERT INTO api_keys (key, user_id, label, created_at) VALUES (?, ?, 'session', ?)",
            (token, user_id, _now()),
        )
        _get_conn().commit()
    return token


def reset_password(user_id: str, password: str) -> bool:
    """重置用户密码；返回是否更新成功。"""
    if not user_id or not password:
        return False
    with _lock:
        cur = _get_conn().execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (_hash_password(password), user_id),
        )
        _get_conn().commit()
        return cur.rowcount > 0


def public_user(user: Optional[dict]) -> Optional[dict]:
    """对外暴露的用户信息（不含密码哈希）。"""
    if not user:
        return None
    return {
        "user_id": user.get("user_id"),
        "name": user.get("name"),
        "username": user.get("username"),
        "is_admin": bool(user.get("is_admin")),
    }


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
                    "summaries": 0, "documents": 0, "feedback": 0,
                    "api_keys": 0}
    if cascade:
        # 会话/偏好/摘要（级联接口依赖平台集成进度；缺失或签名不符时跳过并告警）
        try:
            from src.memory.conversations import delete_user_conversations

            result["sessions"] = delete_user_conversations(user_id)
        except (ImportError, TypeError) as e:
            logger.warning("[users] 会话级联接口缺失或签名不符，跳过：%s", e)
        try:
            from src.memory.memory_items import clear_user_memories
            from src.memory.summary import delete_user_summaries
            from src.memory.user_doc import delete_user_doc
            from src.memory.lifecycle import delete_user_meta

            result["preferences"] = clear_user_memories(user_id)
            result["summaries"] = delete_user_summaries(user_id)
            result["user_doc"] = delete_user_doc(user_id)
            result["lifecycle_meta"] = delete_user_meta(user_id)
        except (ImportError, TypeError) as e:
            logger.warning("[users] 偏好/摘要级联接口缺失，跳过：%s", e)

        try:
            from src.data.documents_store import delete_documents_by_user
            from src.memory.feedback import delete_user_feedback

            result["documents"] = delete_documents_by_user(user_id)
            result["feedback"] = delete_user_feedback(user_id)
        except (ImportError, TypeError) as e:
            logger.warning("[users] 文档/反馈级联接口缺失，跳过：%s", e)

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
    global _db_ready, _DB_PATH
    db.close_all()
    _db_ready = False
    _DB_PATH = path or Path("./data/platform/_test_platform.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
