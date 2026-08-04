"""
跨会话长期记忆存储（场景5）。

用 Python 标准库 sqlite3 实现一个轻量的用户偏好存储，
记录用户喜欢的画家 / 风格，跨会话持久化。

设计要点：
  - 一张表 preferences(user_id, kind, value, weight, updated_at)
  - kind ∈ {"artist", "style"}
  - 同一 (user_id, kind, value) 重复出现时累加 weight（表示偏好强度）
  - load_preferences 返回按 weight 降序的偏好，供 synthesizer 个性化

为什么不用 LangGraph SqliteSaver：
  该库未安装，且 SqliteSaver 存的是"对话检查点"（短期记忆），
  而 S5 需要的是"结构化用户画像"（长期记忆），两者职责不同。
"""

import os
import shutil
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone

# 数据库落盘位置：data/memory/preferences.db（与 conversations/summary 统一目录）
# 测试可通过 ARTAGENT_MEMORY_DIR 覆盖，避免污染真实数据
_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "preferences.db"

# sqlite3 连接非线程安全，用锁保护（Gradio 多线程环境下需要）
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

VALID_KINDS = {"artist", "style"}


def _legacy_preferences_path() -> Path | None:
    """旧路径：历史上 INDEX_DIR 被设置为 ./data/index 时，偏好库落在
    data/data/memory/preferences.db；显式指定 ARTAGENT_MEMORY_DIR 时不迁移。"""
    if os.getenv("ARTAGENT_MEMORY_DIR"):
        return None
    idx = os.getenv("INDEX_DIR", "./data")
    legacy = Path(idx).parent / "data" / "memory" / "preferences.db"
    return legacy if legacy != _DB_PATH else None


def _pref_rows(path: Path) -> int:
    try:
        conn = sqlite3.connect(str(path))
        try:
            return int(conn.execute(
                "SELECT COUNT(*) FROM preferences").fetchone()[0])
        except sqlite3.Error:
            return 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _migrate_legacy_preferences() -> Path:
    """旧路径 data/data/memory/preferences.db → data/memory/（一次性迁移）。

    规则：新路径缺失或为空、且旧路径有数据时，用旧库覆盖新路径；
    否则保留新路径。返回最终应使用的 DB 路径。
    """
    legacy = _legacy_preferences_path()
    if legacy is None or not legacy.exists():
        return _DB_PATH
    if _DB_PATH.exists():
        try:
            if _pref_rows(_DB_PATH) or not _pref_rows(legacy):
                return _DB_PATH
            _DB_PATH.unlink()   # 新库是空壳，先移除再移动旧库
        except OSError:
            return legacy
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(legacy), str(_DB_PATH))
        return _DB_PATH
    except OSError:
        return legacy


def _get_conn() -> sqlite3.Connection:
    """返回全局单例连接，首次调用时建表。"""
    global _conn, _DB_PATH
    if _conn is None:
        _DB_PATH = _migrate_legacy_preferences()
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                user_id    TEXT NOT NULL,
                kind       TEXT NOT NULL,
                value      TEXT NOT NULL,
                weight     REAL NOT NULL DEFAULT 1.0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, kind, value)
            )
            """
        )
        _conn.commit()
    return _conn


def upsert_preference(user_id: str, kind: str, value: str, weight: float = 1.0) -> None:
    """
    写入/累加一条用户偏好。

    Args:
        user_id: 用户标识
        kind:    "artist" 或 "style"
        value:   偏好值（如 "Vincent van Gogh" 或 "浓烈奔放"）
        weight:  本次增加的权重（默认 1.0）
    """
    kind = kind.strip().lower()
    value = value.strip()
    if kind not in VALID_KINDS or not value or not user_id:
        return

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO preferences (user_id, kind, value, weight, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, kind, value)
            DO UPDATE SET weight = weight + excluded.weight,
                          updated_at = excluded.updated_at
            """,
            (user_id, kind, value, weight, now),
        )
        conn.commit()


def load_preferences(user_id: str, top_k: int = 5) -> dict[str, list[str]]:
    """
    读取用户偏好，按权重降序。

    Returns:
        {"artists": ["Van Gogh", ...], "styles": ["浓烈奔放", ...]}
        无偏好时返回 {"artists": [], "styles": []}
    """
    result: dict[str, list[str]] = {"artists": [], "styles": []}
    if not user_id:
        return result

    # 记忆系统 Phase 1：优先读 memory_items（主存储）；无数据回退旧 preferences 表
    try:
        from src.memory.memory_items import list_memories

        items = list_memories(user_id, scope="user")
        prefs = [i for i in items if i.get("kind") == "preference"]
        if prefs:
            result["artists"] = [str(i["content"]) for i in prefs[:top_k]]
            return result
    except Exception:  # noqa: BLE001 —— 新表不可用时回退旧表
        pass

    with _lock:
        conn = _get_conn()
        for kind, out_key in (("artist", "artists"), ("style", "styles")):
            rows = conn.execute(
                """
                SELECT value FROM preferences
                WHERE user_id = ? AND kind = ?
                ORDER BY weight DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, kind, top_k),
            ).fetchall()
            result[out_key] = [r[0] for r in rows]
    return result


def clear_preferences(user_id: str) -> None:
    """清空某用户的全部偏好（测试/重置用）。"""
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
        conn.commit()


def list_preferences(user_id: str) -> list[dict]:
    """返回该用户的全部偏好分项（G2 记忆面板：kind/value/weight/updated_at）。"""
    if not user_id:
        return []
    try:
        from src.memory.memory_items import list_memories

        items = list_memories(user_id, scope="user")
        prefs = [i for i in items if i.get("kind") == "preference"]
        if prefs:
            return [
                {
                    "kind": "preference",
                    "value": str(i["content"]),
                    "weight": float(i.get("importance") or 0.5),
                    "updated_at": i.get("updated_at") or "",
                }
                for i in prefs
            ]
    except Exception:  # noqa: BLE001
        pass
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT kind, value, weight, updated_at FROM preferences
            WHERE user_id = ? ORDER BY kind, weight DESC, updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {"kind": r[0], "value": r[1], "weight": r[2], "updated_at": r[3]}
        for r in rows
    ]


def delete_preference(user_id: str, kind: str, value: str) -> bool:
    """单项删除一条偏好；返回是否删到。"""
    kind = (kind or "").strip().lower()
    value = (value or "").strip()
    if kind not in VALID_KINDS or not value or not user_id:
        return False
    deleted = False
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM preferences WHERE user_id = ? AND kind = ? AND value = ?",
            (user_id, kind, value),
        )
        conn.commit()
        deleted = cur.rowcount > 0
    # 同步删除新表中等价内容条目（记忆系统 Phase 1 双库兼容）
    try:
        from src.memory.memory_items import list_memories

        items = list_memories(user_id, scope="user")
        for item in items:
            if item.get("kind") == "preference" and item.get("content") == value:
                from src.memory.memory_items import delete_memory as _del

                _del(user_id, item["id"])
                deleted = True
    except Exception:  # noqa: BLE001
        pass
    return deleted
