"""
表格 schema 推断（Stage 5）：LLM 看表头+前几行猜列角色，人工确认后生效。

为什么必须有人工确认（方案 §6.2）：猜错 entity_col 会让 recommendation
的排除逻辑静默出错（排除了错误的行而不报错），比明显报错更危险——所以
本模块只产出"建议值"，确认/纠正权在用户（confirm 流程见 table_pipeline）。

工程纪律：
- 推断结果逐列与 df.columns 校验：LLM 猜出不存在的列名时，该角色置空并
  记 warning——宁可少判，不可错判（与提示词"没有合适列就 null"呼应）。
- 任何调用失败返回"全空 + 失败原因"，由调用方决定降级（仍允许用户手填）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.agent.prompts import SCHEMA_INFER_PROMPT
from src.ingestion.table_loader import sample_for_prompt
from src.retrieval.structured_retriever import TableSchema
from src.utils.logging_config import get_logger, log_event

logger = get_logger("ingestion.schema_inference")

_ROLE_KEYS = ("entity_col", "group_axis_col", "description_col", "image_col")


@dataclass
class InferredSchema:
    """schema 推断结果（确认前的建议值）；空串/None 表示该角色无合适列。"""

    entity_col: str = ""
    group_axis_col: str | None = None
    description_col: str = ""
    image_col: str | None = None
    display_name: str = ""
    reasoning: str = ""

    def to_table_schema(self) -> TableSchema:
        return TableSchema(
            entity_col=self.entity_col,
            group_axis_col=self.group_axis_col or None,
            description_col=self.description_col,
            image_col=self.image_col or None,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _sanitize_role(value, columns: list[str]) -> str | None:
    """校验单个角色列：必须真实存在于表头，否则置空并告警。"""
    if value is None:
        return None
    col = str(value).strip()
    if not col:
        return None
    if col not in columns:
        logger.warning("[schema_infer] LLM 猜了不存在的列 %r，该角色置空", col)
        return None
    return col


def infer_table_schema(df: pd.DataFrame, table_name: str = "", llm=None) -> InferredSchema:
    """LLM 推断列角色。任何失败返回全空 InferredSchema（reasoning 记原因）。"""
    columns = [str(c) for c in df.columns]
    cols_with_dtype = "\n".join(f"  {c}: {df[c].dtype}" for c in df.columns)
    prompt = SCHEMA_INFER_PROMPT.format(
        table_name=table_name or "(未命名)",
        columns=cols_with_dtype,
        sample_rows=sample_for_prompt(df),
    )
    try:
        from src.agent.nodes.common import parse_json  # 延迟导入，避免模块级重依赖

        model = llm
        if model is None:
            from src.utils.llm import get_deterministic_llm

            model = get_deterministic_llm()
        raw = model.invoke(prompt).content
        parsed = parse_json(raw)
    except Exception as e:  # noqa: BLE001 — 推断失败不拖垮上传，用户仍可手填
        logger.warning("[schema_infer] 推断调用失败：%s", e)
        return InferredSchema(reasoning=f"(推断失败：{e}，请手动指定列角色)")

    if not isinstance(parsed, dict):
        logger.warning("[schema_infer] 输出非对象，返回空建议")
        return InferredSchema(reasoning="(推断输出异常，请手动指定列角色)")

    result = InferredSchema(
        entity_col=_sanitize_role(parsed.get("entity_col"), columns) or "",
        group_axis_col=_sanitize_role(parsed.get("group_axis_col"), columns),
        description_col=_sanitize_role(parsed.get("description_col"), columns) or "",
        image_col=_sanitize_role(parsed.get("image_col"), columns),
        display_name=str(parsed.get("display_name") or "").strip()[:20],
        reasoning=str(parsed.get("reasoning") or "").strip()[:200],
    )
    schema = result.to_table_schema()
    log_event(
        logger, "schema_infer",
        table=table_name, entity=result.entity_col, axis=result.group_axis_col,
        desc=result.description_col, timeline=schema.supports_timeline,
        recommendation=schema.supports_recommendation,
    )
    return result
