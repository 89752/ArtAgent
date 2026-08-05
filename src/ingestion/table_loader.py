"""
结构化表格加载器：文件类型路由 + 多 sheet 选择，全部零模型调用。

路由：.pdf → PDF 通道；.csv/.xlsx/.xls → 本模块表格通道；
其余扩展名拒绝。xlsx/xls 多 sheet 时用确定性打分选"最像数据表"的子表
（有效列数 × 数据行数），学习计划那种首表是说明页的工作簿也能选对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("ingestion.table_loader")

PDF_EXTENSIONS = {".pdf"}
TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}

_UNNAMED_PREFIX = "Unnamed"  # pandas 对无表头列的自动命名


def classify_upload(filename: str) -> str | None:
    """按扩展名分通道：'pdf' / 'table' / None（不支持的类型）。"""
    suffix = Path(filename or "").suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in TABLE_EXTENSIONS:
        return "table"
    return None


@dataclass
class LoadedTable:
    """加载结果：df + 来源 sheet 名（CSV 为 ""）+ 有效列清单。"""

    df: pd.DataFrame
    sheet_name: str = ""
    columns: list[str] = field(default_factory=list)


def _effective_columns(df: pd.DataFrame) -> list[str]:
    """有效列：列名不是 pandas 自动补的 Unnamed:N，且不是全空列。"""
    return [
        str(c)
        for c in df.columns
        if not str(c).startswith(_UNNAMED_PREFIX) and not df[c].isna().all()
    ]


def _sheet_score(df: pd.DataFrame) -> int:
    """多 sheet 选择打分：有效列数 × 数据行数（越大越像真正的数据表）。"""
    return len(_effective_columns(df)) * len(df)


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
    """CSV 编码兜底：utf-8 → gb18030（gbk 超集）→ latin1（永不失败）。

    中文 Windows/Excel 导出的 CSV 常是 GBK，直接 utf-8 会 UnicodeDecodeError。
    """
    for enc in ("utf-8", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # latin1 理论上不会走到这里；保底
    return pd.read_csv(path, encoding="latin1", encoding_errors="replace")


def load_table(path: str) -> LoadedTable:
    """读取表格文件（CSV/Excel），多 sheet 时确定性选最大数据表。

    抛 ValueError：文件读不出任何有效数据表（<2 有效列或无数据行）。
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        df = _read_csv_any_encoding(path)
        loaded = LoadedTable(df=df, sheet_name="", columns=_effective_columns(df))
    else:
        book = pd.ExcelFile(path)  # xlsx 走 openpyxl，xls 走 xlrd
        best_name, best_df, best_score = "", None, -1
        for name in book.sheet_names:
            df = book.parse(name)
            score = _sheet_score(df)
            logger.info("[table] sheet=%s rows=%d 有效列=%d score=%d",
                        name, len(df), len(_effective_columns(df)), score)
            if score > best_score:
                best_name, best_df, best_score = name, df, score
        loaded = LoadedTable(
            df=best_df, sheet_name=best_name, columns=_effective_columns(best_df)
        )

    if len(loaded.columns) < 2:
        raise ValueError(
            f"未找到有效的数据表（有效列 {len(loaded.columns)} < 2）"
            + (f"，已检查 sheet：{loaded.sheet_name or '(csv)'}" if suffix != ".csv" else "")
        )
    if len(loaded.df) < 1:
        raise ValueError("数据表没有数据行")
    return loaded


def sample_for_prompt(df: pd.DataFrame, n: int = 4, cell_limit: int = 40) -> str:
    """渲染表头+前 n 行的紧凑文本（schema 推断 prompt 用，单元格截断）。"""
    head = df.head(n)
    lines = []
    for _, row in head.iterrows():
        cells = []
        for c in df.columns:
            v = str(row[c])
            if len(v) > cell_limit:
                v = v[:cell_limit] + "…"
            cells.append(f"{c}={v}")
        lines.append("  " + " | ".join(cells))
    return "\n".join(lines)
