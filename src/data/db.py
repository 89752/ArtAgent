"""SQLite 连接共享层。

把散落在各存储模块里的连接样板（mkdir + connect + 锁）收敛到一处：
按 (解析后的路径, row_factory) 缓存连接，避免同一个 .db 文件被多个模块
各开一个独立连接。测试隔离/重载用 close_all() 关闭全部缓存连接。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

_conns: dict[tuple[str, Optional[Callable]], sqlite3.Connection] = {}
_guard = threading.Lock()


def _key(db_path, row_factory):
    return (str(Path(db_path).resolve()), row_factory)


def get_conn(
    db_path, *, row_factory: Optional[Callable] = None
) -> sqlite3.Connection:
    """取（或创建）指定数据库文件的缓存连接。

    row_factory 作为缓存键的一部分：同一文件可同时存在 tuple 与
    sqlite3.Row 两种读法，互不干扰。
    """
    key = _key(db_path, row_factory)
    with _guard:
        conn = _conns.get(key)
        if conn is None:
            Path(key[0]).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key[0], check_same_thread=False)
            if row_factory is not None:
                conn.row_factory = row_factory
            _conns[key] = conn
        return conn


def close_all() -> None:
    """关闭并清空全部缓存连接（测试重置 / 配置变更后重载用）。"""
    with _guard:
        for conn in _conns.values():
            try:
                conn.close()
            except Exception:  # noqa: BLE001 —— 关闭失败不阻塞重置
                pass
        _conns.clear()
