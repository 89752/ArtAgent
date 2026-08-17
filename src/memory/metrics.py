"""记忆抽取质量指标（对齐 DeerMem 的 extraction metrics）。

每次自动抽取落一条记录：提取数 / 放行数 / 各门控拒绝数 / 拒绝率。
拒绝率 > 60% 时告警，帮助发现提示词退化或阈值失调。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger("memory.metrics")

_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "agent_memory.db"

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_metrics (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                 TEXT NOT NULL,
                extracted               INTEGER NOT NULL DEFAULT 0,
                passed                  INTEGER NOT NULL DEFAULT 0,
                rejected_scope          INTEGER NOT NULL DEFAULT 0,
                rejected_durability     INTEGER NOT NULL DEFAULT 0,
                rejected_authority      INTEGER NOT NULL DEFAULT 0,
                rejected_confidence     INTEGER NOT NULL DEFAULT 0,
                rejected_total          INTEGER NOT NULL DEFAULT 0,
                rejection_rate          REAL NOT NULL DEFAULT 0,
                error                   TEXT NOT NULL DEFAULT '',
                created_at              TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_extraction_metrics(
    user_id: str,
    extracted: int,
    passed: int,
    rejected: Optional[list[str]] = None,
    error: str = "",
) -> dict:
    """记录一次抽取质量；返回摘要。拒绝率 > 60% 时告警。"""
    rejected = rejected or []
    breakdown = {
        "scope": sum(1 for r in rejected if r.startswith("scope:")),
        "durability": sum(1 for r in rejected if r.startswith("durability:")),
        "authority": sum(1 for r in rejected if r.startswith("authority:")),
        "confidence": sum(1 for r in rejected if r.startswith("confidence:")),
    }
    total_rejected = len(rejected)
    rate = (total_rejected / extracted) if extracted else 0.0
    with _lock:
        _get_conn().execute(
            """
            INSERT INTO extraction_metrics (
                user_id, extracted, passed,
                rejected_scope, rejected_durability, rejected_authority,
                rejected_confidence, rejected_total, rejection_rate, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                int(extracted or 0),
                int(passed or 0),
                breakdown["scope"],
                breakdown["durability"],
                breakdown["authority"],
                breakdown["confidence"],
                total_rejected,
                round(rate, 4),
                str(error or "")[:200],
                _now(),
            ),
        )
        _get_conn().commit()
    if rate > 0.6:
        logger.warning(
            "[memory.metrics] 抽取拒绝率 %.0f%% 超过 60%%（user=%s extracted=%d passed=%d）",
            rate * 100,
            user_id,
            extracted,
            passed,
        )
    return {
        "extracted": int(extracted or 0),
        "passed": int(passed or 0),
        "rejected": breakdown,
        "rejection_rate": round(rate, 4),
    }


def recent_extraction_metrics(limit: int = 50) -> list[dict]:
    """最近 N 次抽取质量记录（倒序）。"""
    limit = min(max(1, int(limit)), 500)
    with _lock:
        rows = _get_conn().execute(
            """
            SELECT user_id, extracted, passed,
                   rejected_scope, rejected_durability, rejected_authority,
                   rejected_confidence, rejected_total, rejection_rate, error, created_at
            FROM extraction_metrics ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def _reset_for_tests(path: Optional[Path] = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _conn, _DB_PATH
    _conn = None
    _DB_PATH = path or (Path(__file__).resolve().parent.parent.parent
                        / "data" / "index" / "_test_metrics.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
