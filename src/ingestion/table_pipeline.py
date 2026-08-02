"""
表格入库编排（Stage 5）：加载 → schema 推断 → 待确认 → 确认后注册生效。

与 PDF 通道（pipeline.py）的关系：共用 doc_status.json 状态存储（kind 字段
区分 "pdf"/"table"，Stage 6 换 SQLite 时一起迁移）；但表格**不入向量库**——
注册为 StructuredTableRetriever 后走 Stage 2 预留的 fuzzy_match 兜底路径
（无索引表），不需要 BGE/Chroma。

状态机：processing → pending_confirm（推断完成，等用户确认/纠正）
       → active（已注册，可被检索与管线使用）/ failed。

注册是内存行为（注册表与 HybridRetriever 都是单例），服务重启后由
restore_active_tables() 从状态存储重建。
"""

from __future__ import annotations

import time
from pathlib import Path

from src.ingestion.pipeline import (
    UPLOADS_DIR,
    get_doc_status,
    list_doc_status,
    update_doc_status,
)
from src.ingestion.schema_inference import InferredSchema, infer_table_schema
from src.ingestion.table_loader import load_table
from src.retrieval.structured_retriever import (
    TableSchema,
    get_structured_retriever,
    register_structured_dataset,
    _REGISTRY,
)
from src.utils.logging_config import get_logger, log_event

logger = get_logger("ingestion.table_pipeline")

CONFIRMABLE_STATUS = ("pending_confirm", "active")  # active 允许改 schema 重确认


def table_dataset_id(doc_id: str) -> str:
    return f"table_{doc_id}"


def save_table_file(data: bytes, suffix: str, doc_id: str, kb_id: str) -> str:
    """落盘为 uploads/{kb_id}/{doc_id}/table{suffix}，返回路径。"""
    work_dir = UPLOADS_DIR / kb_id / doc_id
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"table{suffix}"
    path.write_bytes(data)
    return str(path)


# ------------------------------------------------------------------ #
# 入库（推断 → 待确认）                                                 #
# ------------------------------------------------------------------ #


def ingest_table(
    table_path: str,
    doc_id: str,
    doc_name: str = "",
    kb_id: str = "default",
    llm=None,
) -> dict:
    """表格入库：加载 + schema 推断，落 pending_confirm 状态（此时不注册）。

    llm 可注入（测试用 fake）；None 走确定性 LLM。
    """
    t0 = time.time()
    doc_name = doc_name or Path(table_path).name
    path_obj = Path(table_path)
    file_size = path_obj.stat().st_size if path_obj.exists() else None
    update_doc_status(
        doc_id,
        doc_name=doc_name, kb_id=kb_id, kind="table",
        status="processing", started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        file_path=str(table_path), file_size=file_size,
    )
    try:
        loaded = load_table(table_path)
        inferred = infer_table_schema(loaded.df, doc_name, llm=llm)
        summary = {
            "kind": "table",
            "status": "pending_confirm",
            "elapsed_sec": round(time.time() - t0, 1),
            "table_path": str(table_path),
            "dataset_id": table_dataset_id(doc_id),
            "rows": len(loaded.df),
            "cols": len(loaded.df.columns),
            "sheet_name": loaded.sheet_name,
            "columns": [str(c) for c in loaded.df.columns],
            "proposed_schema": inferred.to_dict(),
        }
        # Stage 6：kind-specific 字段存 metadata；返回仍保持旧扁平形状兼容
        update_doc_status(doc_id, metadata={k: summary[k] for k in (
            "table_path", "dataset_id", "rows", "cols", "sheet_name",
            "columns", "proposed_schema",
        )}, **{k: v for k, v in summary.items() if k not in (
            "table_path", "dataset_id", "rows", "cols", "sheet_name",
            "columns", "proposed_schema",
        )})
        log_event(logger, "table_ingest", doc_id=doc_id, rows=len(loaded.df),
                  sheet=loaded.sheet_name)
        return {"doc_id": doc_id, **summary}
    except Exception as e:  # noqa: BLE001 — 失败落状态供前端展示
        logger.exception("[table] 入库失败 doc_id=%s", doc_id)
        update_doc_status(doc_id, status="failed", error=str(e))
        raise


# ------------------------------------------------------------------ #
# 确认（注册生效）                                                      #
# ------------------------------------------------------------------ #


def _validate_role(col: str | None, columns: list[str], role: str) -> str | None:
    """确认时校验角色列真实存在（用户从下拉框选，异常输入直接报错）。"""
    if col is None or str(col).strip() == "":
        return None
    col = str(col).strip()
    if col not in columns:
        raise ValueError(f"{role} 指定的列 {col!r} 不存在于表头 {columns}")
    return col


def confirm_table_schema(doc_id: str, roles: dict) -> dict:
    """用户确认/纠正 schema：构建 TableSchema → 注册结构化检索器 + Hybrid → active。

    roles: {entity_col, group_axis_col, description_col, image_col, display_name}，
    空串/None 表示该角色无列。entity_col 为必填——它是模糊匹配与排除逻辑的
    锚点，连实体列都没有的表无法接入任何管线（报错让用户重选）。
    """
    st = get_doc_status(doc_id)
    if not st or st.get("kind") != "table":
        raise KeyError(f"非表格文档：{doc_id}")
    if st.get("status") not in CONFIRMABLE_STATUS:
        raise ValueError(f"当前状态 {st.get('status')} 不可确认 schema")

    columns = [str(c) for c in st.get("columns") or []]
    entity_col = _validate_role(roles.get("entity_col"), columns, "entity_col")
    if not entity_col:
        raise ValueError("entity_col（实体名列）必填——没有实体列的表无法接入检索与管线")
    schema = TableSchema(
        entity_col=entity_col,
        group_axis_col=_validate_role(roles.get("group_axis_col"), columns, "group_axis_col"),
        description_col=_validate_role(roles.get("description_col"), columns, "description_col") or "",
        image_col=_validate_role(roles.get("image_col"), columns, "image_col"),
    )
    display_name = str(roles.get("display_name") or "").strip()[:20]

    dataset_id = st["dataset_id"]
    table_path = st["table_path"]
    retriever = register_structured_dataset(
        dataset_id,
        schema,
        source="user_table",
        df_loader=lambda: load_table(table_path).df,  # 懒加载，首次访问才读盘
    )

    from src.retrieval.hybrid import get_hybrid_retriever

    get_hybrid_retriever().register(dataset_id, retriever)

    confirmed = {
        "status": "active",
        "metadata": {
            "confirmed_schema": {
                "entity_col": schema.entity_col,
                "group_axis_col": schema.group_axis_col,
                "description_col": schema.description_col,
                "image_col": schema.image_col,
                "display_name": display_name,
            },
            "display_name": display_name or st.get("doc_name") or dataset_id,
            "supports_timeline": schema.supports_timeline,
            "supports_recommendation": schema.supports_recommendation,
        },
    }
    update_doc_status(doc_id, **confirmed)
    log_event(logger, "table_confirm", doc_id=doc_id, dataset_id=dataset_id,
              timeline=schema.supports_timeline,
              recommendation=schema.supports_recommendation)
    return {"doc_id": doc_id, **get_doc_status(doc_id)}


def unregister_table(dataset_id: str) -> None:
    """注销表格数据源（Stage 6 删除级联用）：从注册表与 Hybrid 移除。"""
    _REGISTRY.pop(dataset_id, None)
    from src.retrieval.hybrid import get_hybrid_retriever

    get_hybrid_retriever()._retrievers.pop(dataset_id, None)
    logger.info("[table] 已注销数据源 %s", dataset_id)


def restore_active_tables() -> int:
    """服务重启后从状态存储重建注册（幂等）；返回恢复的数据源个数。"""
    restored = 0
    for st in list_doc_status():
        if st.get("kind") != "table" or st.get("status") != "active":
            continue
        cs = st.get("confirmed_schema") or {}
        table_path = st.get("table_path")
        dataset_id = st.get("dataset_id")
        if not (table_path and dataset_id and cs.get("entity_col")):
            continue
        try:
            schema = TableSchema(
                entity_col=cs["entity_col"],
                group_axis_col=cs.get("group_axis_col") or None,
                description_col=cs.get("description_col") or "",
                image_col=cs.get("image_col") or None,
            )
            retriever = register_structured_dataset(
                dataset_id, schema, source="user_table",
                df_loader=lambda p=table_path: load_table(p).df,
            )
            from src.retrieval.hybrid import get_hybrid_retriever

            get_hybrid_retriever().register(dataset_id, retriever)
            restored += 1
        except Exception as e:  # noqa: BLE001 — 单表恢复失败不影响其他
            logger.warning("[table] 恢复 %s 失败：%s", dataset_id, e)
    if restored:
        logger.info("[table] 重启恢复 %d 个表格数据源", restored)
    return restored
