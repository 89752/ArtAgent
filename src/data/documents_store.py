"""
文档/数据源生命周期持久化。

把早期的 JSON 状态文件（data/index/doc_status.json）整体替换为 SQLite
`documents` 表，支撑文件库列表、删除按钮与级联清理。

设计原则：
  · 通用字段放列，kind-specific 字段放 metadata JSON。
  · 对外保持与旧 list_doc_status() 近似的 dict 形状，减少上层改动。
  · 首次启动时自动迁移旧 JSON（迁移后重命名原文件）。
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger("data.documents_store")

DB_PATH = Path(os.getenv("INDEX_DIR", "./data/index")) / "documents.db"
_LEGACY_STATUS_FILE = Path(os.getenv("INDEX_DIR", "./data/index")) / "doc_status.json"


# ------------------------------------------------------------------ #
# Schema                                                              #
# ------------------------------------------------------------------ #
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL DEFAULT 'web_user',
    kb_id        TEXT NOT NULL DEFAULT 'default',
    kind         TEXT NOT NULL CHECK(kind IN ('pdf', 'table')),
    doc_name     TEXT,
    status       TEXT NOT NULL CHECK(status IN ('processing','pending_confirm','active','done','failed')),
    started_at   TEXT,
    finished_at  TEXT,
    file_path    TEXT,
    file_size    INTEGER,
    pages        INTEGER,
    text_chunks  INTEGER,
    image_pages  INTEGER,
    elapsed_sec  REAL,
    error        TEXT,
    metadata     TEXT
);
"""


# ------------------------------------------------------------------ #
# Connection helper                                                   #
# ------------------------------------------------------------------ #
def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------ #
# Init + migration                                                    #
# ------------------------------------------------------------------ #
def init_db() -> None:
    """建表；迁移旧 JSON；重置服务重启导致中断的解析任务（防僵尸轮询）。"""
    with _connect() as conn:
        conn.executescript(_CREATE_TABLE_SQL)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "user_id" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN user_id TEXT NOT NULL DEFAULT 'web_user'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, started_at)"
        )
        cur = conn.execute(
            """
            UPDATE documents SET status = 'failed', error = ?
            WHERE status IN ('processing', 'pending')
            """,
            ("服务重启，解析任务中断，请重新上传",),
        )
        if cur.rowcount:
            logger.info(
                "[documents_store] 已将 %d 条中断中的解析任务标记为 failed",
                cur.rowcount,
            )
        conn.commit()
    _migrate_legacy_json()


def _migrate_legacy_json() -> None:
    if not _LEGACY_STATUS_FILE.exists():
        return

    with _connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM documents")
        if cur.fetchone()[0] > 0:
            logger.info("[documents_store] documents 表已有数据，跳过 JSON 迁移")
            return

    try:
        data = json.loads(_LEGACY_STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[documents_store] 旧状态文件读取失败，跳过迁移：%s", e)
        return

    if not isinstance(data, dict):
        return

    count = 0
    for doc_id, info in data.items():
        try:
            _insert_from_legacy(doc_id, info)
            count += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("[documents_store] 迁移 doc_id=%s 失败：%s", doc_id, e)

    logger.info("[documents_store] 已从旧 JSON 迁移 %d 条文档状态", count)

    # 迁移成功后重命名原文件，避免重复迁移
    migrated_name = _LEGACY_STATUS_FILE.with_suffix(".json.migrated")
    try:
        if migrated_name.exists():
            migrated_name.unlink()
        _LEGACY_STATUS_FILE.rename(migrated_name)
    except OSError as e:
        logger.warning("[documents_store] 无法重命名旧状态文件：%s", e)


def _insert_from_legacy(doc_id: str, info: dict) -> None:
    kind = info.get("kind", "pdf")
    if kind not in ("pdf", "table"):
        kind = "pdf"

    metadata: dict = {}
    if kind == "pdf":
        metadata["route_distribution"] = info.get("route_distribution", {})
    else:
        # 表格相关字段全部进 metadata
        for key in (
            "dataset_id", "table_path", "rows", "cols", "sheet_name", "columns",
            "proposed_schema", "confirmed_schema", "display_name",
            "supports_timeline", "supports_recommendation",
        ):
            if key in info:
                metadata[key] = info[key]

    row = {
        "doc_id": doc_id,
        "kb_id": info.get("kb_id", "default"),
        "kind": kind,
        "doc_name": info.get("doc_name", ""),
        "status": info.get("status", "done"),
        "started_at": info.get("started_at", ""),
        "finished_at": info.get("finished_at", ""),
        "file_path": info.get("file_path", ""),
        "file_size": info.get("file_size"),
        "pages": info.get("pages"),
        "text_chunks": info.get("text_chunks"),
        "image_pages": info.get("image_pages"),
        "elapsed_sec": info.get("elapsed_sec"),
        "error": info.get("error", ""),
        "metadata": metadata,  # add_document 会负责 JSON 序列化
    }
    add_document(**row)


# ------------------------------------------------------------------ #
# CRUD                                                                #
# ------------------------------------------------------------------ #
def add_document(
    doc_id: str,
    kind: str,
    user_id: str = "web_user",
    doc_name: str = "",
    kb_id: str = "default",
    status: str = "processing",
    started_at: str = "",
    finished_at: str = "",
    file_path: str = "",
    file_size: Optional[int] = None,
    pages: Optional[int] = None,
    text_chunks: Optional[int] = None,
    image_pages: Optional[int] = None,
    elapsed_sec: Optional[float] = None,
    error: str = "",
    metadata: Optional[dict] = None,
) -> None:
    """插入一条文档记录；已存在则忽略（幂等，用于迁移）。"""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO documents
            (doc_id, user_id, kb_id, kind, doc_name, status, started_at, finished_at,
             file_path, file_size, pages, text_chunks, image_pages,
             elapsed_sec, error, metadata)
            VALUES
            (:doc_id, :user_id, :kb_id, :kind, :doc_name, :status, :started_at, :finished_at,
             :file_path, :file_size, :pages, :text_chunks, :image_pages,
             :elapsed_sec, :error, :metadata)
            """,
            {
                "doc_id": doc_id,
                "user_id": user_id,
                "kb_id": kb_id,
                "kind": kind,
                "doc_name": doc_name,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "file_path": file_path,
                "file_size": file_size,
                "pages": pages,
                "text_chunks": text_chunks,
                "image_pages": image_pages,
                "elapsed_sec": elapsed_sec,
                "error": error,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            },
        )
        conn.commit()


def update_document(doc_id: str, **fields) -> None:
    """更新文档记录。fields 中可含 metadata dict，会与现有 metadata 合并。"""
    if not fields:
        return

    allowed = {
        "kb_id", "doc_name", "status", "started_at", "finished_at",
        "file_path", "file_size", "pages", "text_chunks", "image_pages",
        "elapsed_sec", "error", "metadata",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return

    with _connect() as conn:
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            row = conn.execute(
                "SELECT metadata FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            existing = json.loads(row["metadata"] or "{}") if row else {}
            existing.update(updates["metadata"])
            updates["metadata"] = json.dumps(existing, ensure_ascii=False)

        sets = ", ".join(f"{k} = :{k}" for k in updates)
        updates["doc_id"] = doc_id
        conn.execute(f"UPDATE documents SET {sets} WHERE doc_id = :doc_id", updates)
        conn.commit()


def upsert_document(
    doc_id: str,
    kind: str = "pdf",
    status: str = "processing",
    user_id: str = "web_user",
    **fields,
) -> None:
    """更新文档记录；不存在时自动创建（原 pipeline.update_doc_status 语义）。"""
    # 文档在上传时已经按真实用户落库。这里必须用同一 user_id 查询，
    # 否则非默认用户的记录会被误判为不存在；随后 INSERT OR IGNORE 又会
    # 因 doc_id 主键冲突被忽略，最终状态便会永远停在 processing。
    if get_document(doc_id, user_id) is None:
        add_document(doc_id=doc_id, kind=kind, user_id=user_id, status=status, **fields)
    else:
        fields["status"] = status  # status 是命名参数，不会进 fields，必须显式补回
        update_document(doc_id, **fields)


def get_document(doc_id: str, user_id: str = "web_user") -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ? AND user_id = ?",
            (doc_id, user_id),
        ).fetchone()
    return _to_status_dict(row) if row else None


def list_documents(user_id: str = "web_user") -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY started_at DESC",
            (user_id,),
        ).fetchall()
    return [_to_status_dict(r) for r in rows]


def delete_document(doc_id: str, user_id: str = "web_user") -> bool:
    """删除 SQLite 中的记录；返回是否真删了一条。"""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE doc_id = ? AND user_id = ?", (doc_id, user_id)
        )
        conn.commit()
    return cur.rowcount > 0


def delete_documents_by_user(user_id: str) -> int:
    """删除某用户全部文档记录；返回删除条数（级联删除用，文件另行清理）。"""
    if not user_id:
        return 0
    with _connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
        conn.commit()
    return cur.rowcount


# ------------------------------------------------------------------ #
# Serialization helper                                                #
# ------------------------------------------------------------------ #
def _to_status_dict(row: sqlite3.Row) -> dict:
    """把 SQLite row 还原成与旧 list_doc_status 兼容的扁平 dict。"""
    metadata = json.loads(row["metadata"] or "{}")
    out = {
        "doc_id": row["doc_id"],
        "kb_id": row["kb_id"],
        "kind": row["kind"],
        "doc_name": row["doc_name"] or "",
        "status": row["status"],
        "started_at": row["started_at"] or "",
        "file_path": row["file_path"] or "",
        "elapsed_sec": row["elapsed_sec"],
        "error": row["error"] or "",
    }

    # PDF 通用字段
    if row["pages"] is not None:
        out["pages"] = row["pages"]
    if row["text_chunks"] is not None:
        out["text_chunks"] = row["text_chunks"]
    if row["image_pages"] is not None:
        out["image_pages"] = row["image_pages"]
    if row["file_size"] is not None:
        out["file_size"] = row["file_size"]
    if row["finished_at"]:
        out["finished_at"] = row["finished_at"]

    # metadata 展开，保持前端/旧代码看到的字段形状
    out.update(metadata)
    return out


def _reset_for_tests(path: Optional[Path] = None) -> None:
    """测试专用：重置到指定文件数据库（:memory: 不支持跨连接共享，故用临时文件）。"""
    global DB_PATH
    DB_PATH = path or Path("./data/index/_test_documents.db")
    try:
        DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
