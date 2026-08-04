"""每轮 Agent 轨迹记录与指标汇总（G8 / 2.4）。

落库 data/index/observability.db（INDEX_DIR 可覆盖）。成本估算采用
env 单价（每 1K token），默认 0 即关闭成本估算；token 数用 4 字符 ≈ 1 token 近似。
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(os.getenv("INDEX_DIR", "./data/index")) / "observability.db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id         TEXT NOT NULL DEFAULT '',
                session_id         TEXT NOT NULL DEFAULT '',
                intent             TEXT NOT NULL DEFAULT '',
                steps_json         TEXT NOT NULL DEFAULT '[]',
                tools_json         TEXT NOT NULL DEFAULT '[]',
                context_chars      INTEGER NOT NULL DEFAULT 0,
                tool_rounds        INTEGER NOT NULL DEFAULT 0,
                latency_ms         REAL NOT NULL DEFAULT 0,
                est_cost           REAL NOT NULL DEFAULT 0,
                final_answer_len   INTEGER NOT NULL DEFAULT 0,
                reflection_triggered INTEGER NOT NULL DEFAULT 0,
                web_fallback       INTEGER NOT NULL DEFAULT 0,
                cancelled          INTEGER NOT NULL DEFAULT 0,
                error              TEXT NOT NULL DEFAULT '',
                created_at         TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_rate() -> float:
    return float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0") or 0)


def _output_rate() -> float:
    return float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0") or 0)


def estimate_cost(context_chars: int, answer_len: int) -> float:
    """按 4 字符 ≈ 1 token 估算成本（单价为 0 时返回 0）。"""
    return (
        (context_chars / 4 / 1000) * _input_rate()
        + (answer_len / 4 / 1000) * _output_rate()
    )


def record_run(
    *,
    request_id: str = "",
    session_id: str = "",
    intent: str = "",
    steps: Optional[list] = None,
    tools: Optional[list] = None,
    context_chars: int = 0,
    tool_rounds: int = 0,
    latency_ms: float = 0.0,
    final_answer_len: int = 0,
    reflection_triggered: bool = False,
    web_fallback: bool = False,
    cancelled: bool = False,
    error: str = "",
) -> int:
    """写入一条轨迹；返回自增 id。"""
    est = estimate_cost(int(context_chars or 0), int(final_answer_len or 0))
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """
            INSERT INTO agent_runs (
                request_id, session_id, intent, steps_json, tools_json,
                context_chars, tool_rounds, latency_ms, est_cost,
                final_answer_len, reflection_triggered, web_fallback,
                cancelled, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(request_id or "")[:64],
                str(session_id or "")[:128],
                str(intent or "")[:32],
                json.dumps(steps or [], ensure_ascii=False),
                json.dumps(tools or [], ensure_ascii=False),
                int(context_chars or 0),
                int(tool_rounds or 0),
                round(float(latency_ms or 0), 1),
                round(est, 6),
                int(final_answer_len or 0),
                1 if reflection_triggered else 0,
                1 if web_fallback else 0,
                1 if cancelled else 0,
                str(error or "")[:300],
                _now(),
            ),
        )
        conn.commit()
    return int(cur.lastrowid)


def list_runs(limit: int = 50) -> list[dict]:
    limit = min(max(1, int(limit)), 500)
    with _lock:
        rows = _get_conn().execute(
            "SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["steps"] = json.loads(d.pop("steps_json") or "[]")
        except json.JSONDecodeError:
            d["steps"] = []
        try:
            d["tools"] = json.loads(d.pop("tools_json") or "[]")
        except json.JSONDecodeError:
            d["tools"] = []
        out.append(d)
    return out


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def metrics(limit: int = 500) -> dict:
    """最近 N 条轨迹汇总：请求量/延迟/成本/工具分布/反思与兜底率/错误率。"""
    runs = list_runs(limit=limit)
    if not runs:
        return {"count": 0}
    latencies = [r["latency_ms"] for r in runs]
    tool_counter: dict[str, int] = {}
    for r in runs:
        for t in r.get("tools") or []:
            tool_counter[str(t)] = tool_counter.get(str(t), 0) + 1
    return {
        "count": len(runs),
        "window": f"last {len(runs)} runs",
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 1),
            "p50": round(_pct(latencies, 0.5), 1),
            "p95": round(_pct(latencies, 0.95), 1),
        },
        "est_cost_total": round(sum(r["est_cost"] for r in runs), 6),
        "tool_calls": tool_counter,
        "reflection_rate": round(
            sum(r["reflection_triggered"] for r in runs) / len(runs), 4
        ),
        "web_fallback_rate": round(
            sum(r["web_fallback"] for r in runs) / len(runs), 4
        ),
        "error_rate": round(sum(1 for r in runs if r["error"]) / len(runs), 4),
        "cancelled": sum(r["cancelled"] for r in runs),
    }


def _reset_for_tests(path: Path | None = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _conn, _DB_PATH
    _conn = None
    _DB_PATH = path or Path("./data/index/_test_observability.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
