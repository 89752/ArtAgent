"""记忆条目主存储层：memory_items / memory_events（长期记忆唯一事实源）。

- memory_items：单条陈述式记忆（用户偏好/事实/画像），带 entity/embedding/
  importance/source/version/superseded_by/软删除；
- memory_events：审计（create/update/supersede/recall/delete）。

隔离：namespace = (user_id, thread_id, scope)，所有读写强制带 user_id；
工具层默认用户可用 MEMORY_USER_ID 覆盖（评估用 eval-test，生产用 web_user）；
store.py（偏好兼容层）与 tools/memory.py（remember/recall/forget）都读写本表。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"
_DB_PATH = _DB_DIR / "agent_memory.db"

# RLock：add_memory 持有锁时还会调用 _audit（内部再次加锁）
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

DEFAULT_MEMORY_USER = "default_user"

VALID_KINDS = {"preference", "fact", "profile", "event"}
VALID_SCOPES = {"user", "thread", "agent"}


def get_memory_user_id() -> str:
    """工具层当前记忆身份（MEMORY_USER_ID 可覆盖，评估隔离用）。"""
    return os.getenv("MEMORY_USER_ID", DEFAULT_MEMORY_USER).strip() or DEFAULT_MEMORY_USER


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                thread_id     TEXT,
                scope         TEXT NOT NULL DEFAULT 'user',
                kind          TEXT NOT NULL,
                content       TEXT NOT NULL,
                entity        TEXT,
                embedding     TEXT,
                importance    REAL NOT NULL DEFAULT 0.5,
                source        TEXT NOT NULL DEFAULT 'user_explicit',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                version       INTEGER NOT NULL DEFAULT 1,
                superseded_by TEXT,
                deleted_at    TEXT
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                item_id    TEXT,
                action     TEXT NOT NULL,
                detail     TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_items(user_id, scope)"
        )
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_entity ON memory_items(user_id, entity)"
        )
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embed(text: str) -> Optional[str]:
    """bge-m3 编码（复用检索层单例）；失败返回 None（回退关键词检索）。"""
    try:
        from src.retrieval.hybrid import get_bge_m3_embed_fn

        vec = get_bge_m3_embed_fn()(text)
        return json.dumps([float(v) for v in vec])
    except Exception:  # noqa: BLE001 —— 模型不可用时记忆功能不中断
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _vector_backend() -> str:
    """记忆向量后端：sqlite（默认，Python 全量打分）| chroma（可选加速）。"""
    return os.getenv("MEMORY_VECTOR_BACKEND", "sqlite").strip().lower()


def _chroma_upsert(item: dict) -> None:
    """可选：把条目向量同步进 Chroma collection memory_items。"""
    if _vector_backend() != "chroma":
        return
    try:
        emb = item.get("embedding")
        if not emb:
            return
        from src.retrieval.hybrid import get_or_create_chroma_collection

        collection = get_or_create_chroma_collection("memory_items")
        collection.upsert(
            ids=[item["id"]],
            embeddings=[json.loads(emb)],
            metadatas=[{
                "user_id": item.get("user_id") or "",
                "scope": item.get("scope") or "user",
                "entity": item.get("entity") or "",
                "kind": item.get("kind") or "",
                "content": (item.get("content") or "")[:2000],
            }],
        )
    except Exception:  # noqa: BLE001 —— 向量索引不可用不影响主流程
        pass


def _chroma_delete(item_id: str) -> None:
    """可选：从 Chroma 移除条目（软删除/淘汰时同步）。"""
    if _vector_backend() != "chroma":
        return
    try:
        from src.retrieval.hybrid import get_or_create_chroma_collection

        get_or_create_chroma_collection("memory_items").delete(ids=[item_id])
    except Exception:  # noqa: BLE001
        pass


def _days_since(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return 0.0


def _capacity_limits() -> tuple[int, int]:
    """单用户容量上限（条目数 / 总字符），env 可覆盖。"""
    try:
        max_items = max(1, int(os.getenv("MEMORY_MAX_ITEMS_PER_USER", "200")))
    except (TypeError, ValueError):
        max_items = 200
    try:
        max_chars = max(0, int(os.getenv("MEMORY_MAX_CHARS_PER_USER", "40000")))
    except (TypeError, ValueError):
        max_chars = 40000
    return max_items, max_chars


def _evict_over_capacity(user_id: str) -> int:
    """按 重要性×新鲜度衰减 淘汰超限条目（软删除 + 审计）。

    score = importance · 0.9^days；低分先淘汰；kind='profile'（用户画像
    聚合）受保护，不参与淘汰；max_chars=0 表示只按条目数限制。
    """
    max_items, max_chars = _capacity_limits()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM memory_items WHERE user_id = ? "
        "AND deleted_at IS NULL AND superseded_by IS NULL",
        (user_id,),
    ).fetchall()
    active = [_row_to_dict(r) for r in rows]
    total_chars = sum(len(str(i.get("content") or "")) for i in active)
    over_items = len(active) > max_items
    over_chars = max_chars > 0 and total_chars > max_chars
    if not over_items and not over_chars:
        return 0

    evictable = [i for i in active if i.get("kind") != "profile"]

    def _score(i: dict) -> float:
        imp = float(i.get("importance") or 0.5)
        return imp * (0.9 ** _days_since(str(i.get("updated_at") or "")))

    evictable.sort(key=lambda i: (_score(i), str(i.get("updated_at") or "")))
    removed = 0
    while evictable and (len(active) - removed > max_items
                         or (max_chars > 0 and total_chars > max_chars)):
        victim = evictable.pop(0)
        conn.execute(
            "UPDATE memory_items SET deleted_at = ? WHERE id = ?",
            (_now(), victim["id"]),
        )
        total_chars -= len(str(victim.get("content") or ""))
        removed += 1
        _audit(
            user_id, victim["id"], "evict",
            f"容量淘汰 score={_score(victim):.4f}",
        )
        _chroma_delete(victim["id"])
    conn.commit()
    return removed


def _row_to_dict(row: sqlite3.Row | tuple) -> dict:
    if hasattr(row, "keys"):
        return dict(row)
    cols = (
        "id", "user_id", "thread_id", "scope", "kind", "content", "entity",
        "embedding", "importance", "source", "created_at", "updated_at",
        "version", "superseded_by", "deleted_at",
    )
    return dict(zip(cols, row))


def _audit(user_id: str, item_id: Optional[str], action: str, detail: str = "") -> None:
    with _lock:
        _get_conn().execute(
            "INSERT INTO memory_events (user_id, item_id, action, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, item_id, action, detail[:500], _now()),
        )
        _get_conn().commit()


def _active_duplicate(
    user_id: str,
    kind: str,
    entity: Optional[str],
    scope: str,
    thread_id: Optional[str],
) -> Optional[dict]:
    """查同 namespace 下未删除、未覆盖的同 entity 旧条目。"""
    conn = _get_conn()
    sql = (
        "SELECT * FROM memory_items WHERE user_id = ? AND kind = ? AND scope = ? "
        "AND deleted_at IS NULL AND superseded_by IS NULL"
    )
    params: list = [user_id, kind, scope]
    if entity:
        sql += " AND entity = ? COLLATE NOCASE"
        params.append(entity)
    if thread_id:
        sql += " AND thread_id = ?"
        params.append(thread_id)
    else:
        sql += " AND (thread_id IS NULL OR thread_id = '')"
    row = conn.execute(sql + " ORDER BY updated_at DESC LIMIT 1", params).fetchone()
    return _row_to_dict(row) if row else None


def add_memory(
    user_id: str,
    content: str,
    kind: str = "preference",
    scope: str = "user",
    entity: Optional[str] = None,
    thread_id: Optional[str] = None,
    source: str = "user_explicit",
    importance: float = 0.5,
    smart_conflict: bool = False,
    llm=None,
) -> dict:
    """写入一条记忆；处理同义合并与漂移覆盖（supersede），并写审计。

    smart_conflict=True 时（MEMORY_SMART_MERGE）：新旧内容不同且同
    entity+kind 时，先用 LLM 判定 REPLACE（覆盖）/ MERGE（并存）/ SKIP（不写），
    失败回落确定性 REPLACE。默认 False 保持纯确定性行为。

    返回落库条目（含 id / action）。
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("记忆内容不能为空")
    if kind not in VALID_KINDS:
        kind = "preference"
    if scope not in VALID_SCOPES:
        scope = "user"
    entity = (entity or "").strip() or None
    try:
        importance = float(importance)
    except (TypeError, ValueError):
        importance = 0.5
    importance = max(0.0, min(1.0, importance))

    # 语义冲突决策放锁外（LLM 网络调用不持库锁）
    decision = None
    if smart_conflict:
        try:
            old0 = _active_duplicate(user_id, kind, entity, scope, thread_id)
            if old0 is not None and (old0["content"] or "").strip() != content:
                from src.memory.conflict import resolve_conflict

                decision = resolve_conflict(
                    old0["content"], content, kind, entity or "", llm=llm,
                )
                if decision["action"] == "SKIP":
                    return {**old0, "action": "skip"}
        except Exception:  # noqa: BLE001 —— 决策失败回落确定性覆盖
            decision = None

    now = _now()
    with _lock:
        conn = _get_conn()
        old = _active_duplicate(user_id, kind, entity, scope, thread_id)
        if old is not None and (old["content"] or "").strip() == content:
            # 同义合并：只更新时间戳/重要性，不新增
            conn.execute(
                "UPDATE memory_items SET updated_at = ?, importance = ? WHERE id = ?",
                (now, importance, old["id"]),
            )
            conn.commit()
            _audit(user_id, old["id"], "update", f"同义合并：{content[:120]}")
            return {**old, "updated_at": now, "importance": importance, "action": "update"}

        item_id = f"mem_{uuid.uuid4().hex[:12]}"
        embedding = _embed(content)
        conn.execute(
            """
            INSERT INTO memory_items
              (id, user_id, thread_id, scope, kind, content, entity, embedding,
               importance, source, created_at, updated_at, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (item_id, user_id, thread_id, scope, kind, content, entity, embedding,
             importance, source, now, now),
        )
        if old is not None and not (decision and decision["action"] == "MERGE"):
            # 漂移覆盖：旧条目保留可追溯，但不再参与检索
            conn.execute(
                "UPDATE memory_items SET superseded_by = ? WHERE id = ?",
                (item_id, old["id"]),
            )
            conn.commit()
            _audit(user_id, old["id"], "supersede", f"被 {item_id} 覆盖")
        conn.commit()
        _audit(user_id, item_id, "create", content[:200])
        row = conn.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        item = _row_to_dict(row)
        if old is not None and decision and decision["action"] == "MERGE":
            item["action"] = "merge"
        else:
            item["action"] = "supersede" if old is not None else "create"
    _chroma_upsert(item)
    _evict_over_capacity(user_id)
    return item


def search_memories(
    user_id: str,
    query: str,
    scope: str = "user",
    thread_id: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.25,
) -> list[dict]:
    """语义检索有效记忆：score = 0.7·相似度 + 0.2·新鲜度 + 0.1·重要性。

    embedding 缺失时回退关键词包含匹配（按新鲜度+重要性排序）。
    """
    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k = 5
    conn = _get_conn()
    sql = (
        "SELECT * FROM memory_items WHERE user_id = ? AND scope = ? "
        "AND deleted_at IS NULL AND superseded_by IS NULL"
    )
    params: list = [user_id, scope]
    if thread_id:
        sql += " AND thread_id = ?"
        params.append(thread_id)
    else:
        sql += " AND (thread_id IS NULL OR thread_id = '')"
    rows = conn.execute(sql + " ORDER BY updated_at DESC", params).fetchall()
    items = [_row_to_dict(r) for r in rows]
    if not items:
        return []

    q_vec = _embed(query) if query else None
    q_vec_list = json.loads(q_vec) if q_vec else None

    # 可选：Chroma 候选加速（失败/不可用自动回落全量扫描）
    if q_vec_list and _vector_backend() == "chroma":
        try:
            from src.retrieval.hybrid import get_or_create_chroma_collection

            col = get_or_create_chroma_collection("memory_items")
            res = col.query(
                query_embeddings=[q_vec_list],
                n_results=min(max(10, top_k * 4), 100),
                where={"$and": [{"user_id": user_id}, {"scope": scope}]},
            )
            candidate_ids = set((res.get("ids") or [[]])[0])
            if candidate_ids:
                items = [i for i in items if i["id"] in candidate_ids]
        except Exception:  # noqa: BLE001 —— 向量后端异常回落全量
            pass

    now = datetime.now(timezone.utc)

    scored: list[tuple[float, dict]] = []
    for item in items:
        sim = 0.0
        if q_vec_list:
            try:
                item_vec = json.loads(item.get("embedding") or "null")
                sim = _cosine(q_vec_list, item_vec or [])
            except (TypeError, ValueError, json.JSONDecodeError):
                sim = 0.0
        if not q_vec_list and query:
            # 无向量兜底：关键词包含
            sim = 1.0 if query.lower() in (item.get("content") or "").lower() else 0.0
        try:
            updated = datetime.fromisoformat(item.get("updated_at") or "")
            days = max(0.0, (now - updated).total_seconds() / 86400.0)
        except (TypeError, ValueError):
            days = 0.0
        freshness = 2.71828 ** (-days / 30.0)
        importance = float(item.get("importance") or 0.5)
        if not query:
            # 空查询 = 按新鲜度+重要性取最近/重要条目（跨轮引用兜底用）
            score = 0.2 * freshness + 0.1 * importance
        else:
            score = 0.7 * sim + 0.2 * freshness + 0.1 * importance
            if q_vec_list is None and sim == 0.0:
                continue  # 无向量且无关键词命中：不注入
        scored.append((score, item))

    scored.sort(key=lambda t: -t[0])
    out = []
    for score, item in scored[:top_k]:
        if score < min_score:
            continue
        out.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "content": item["content"],
                "entity": item.get("entity") or "",
                "source": item.get("source") or "",
                "importance": float(item.get("importance") or 0.5),
                "created_at": item.get("created_at") or "",
                "updated_at": item.get("updated_at") or "",
                "score": round(score, 4),
            }
        )
        _audit(user_id, item["id"], "recall", query[:200])
    return out


def list_memories(
    user_id: str,
    scope: Optional[str] = None,
    include_superseded: bool = False,
) -> list[dict]:
    """列出有效记忆（用户可见/管理用）。"""
    sql = "SELECT * FROM memory_items WHERE user_id = ? AND deleted_at IS NULL"
    params: list = [user_id]
    if scope:
        sql += " AND scope = ?"
        params.append(scope)
    if not include_superseded:
        sql += " AND superseded_by IS NULL"
    rows = _get_conn().execute(sql + " ORDER BY updated_at DESC", params).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_memory(user_id: str, item_id: str) -> bool:
    """软删除一条记忆 + 审计。"""
    with _lock:
        cur = _get_conn().execute(
            "UPDATE memory_items SET deleted_at = ? "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            (_now(), item_id, user_id),
        )
        _get_conn().commit()
    _chroma_delete(item_id)
    if cur.rowcount:
        _audit(user_id, item_id, "delete", "")
        return True
    return False


def delete_by_entity(user_id: str, entity: str, kind: Optional[str] = None) -> int:
    """按实体软删除（forget(entity) 用）；返回删除条数。"""
    sql = (
        "UPDATE memory_items SET deleted_at = ? "
        "WHERE user_id = ? AND entity = ? COLLATE NOCASE AND deleted_at IS NULL"
    )
    params: list = [_now(), user_id, entity]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    with _lock:
        cur = _get_conn().execute(sql, params)
        _get_conn().commit()
    if cur.rowcount:
        for row in _get_conn().execute(
            "SELECT id FROM memory_items WHERE user_id = ? AND entity = ? "
            "COLLATE NOCASE AND deleted_at IS NOT NULL",
            (user_id, entity),
        ).fetchall():
            _chroma_delete(row[0])
        _audit(user_id, None, "delete", f"entity={entity}")
    return cur.rowcount


def clear_user_memories(user_id: str) -> int:
    """清空某用户全部记忆（评估用例前清场用，硬删）。"""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM memory_items WHERE user_id = ?", (user_id,)
        )
        conn.execute(
            "DELETE FROM memory_events WHERE user_id = ?", (user_id,)
        )
        conn.commit()
    return cur.rowcount


def _reset_for_tests(path: Path | None = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _conn, _DB_PATH
    _conn = None
    _DB_PATH = path or Path("./data/memory/_test_memory_items.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
