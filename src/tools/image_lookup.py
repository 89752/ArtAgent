"""
Tool 4 (设计文档 §4.4): Image Lookup Tool

从 SemArt 本地图片资源中查找画作配图。
供场景2（时间线梳理）"配图佐证"使用，也可独立调用。

区别于 image_analysis：本工具只做"查找/定位"，不调用视觉模型。
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

_DATA_DIR = Path(os.getenv("SEMART_DATA_DIR", "./SemArt"))


def _resolve_path(image_file: str) -> str:
    """把 IMAGE_FILE 字段解析成完整本地路径字符串。"""
    p = _DATA_DIR / "Images" / image_file
    return str(p) if p.exists() else ""


def _rows_to_images(df: pd.DataFrame, limit: int) -> list[dict]:
    out = []
    for _, row in df.head(limit).iterrows():
        image_file = str(row.get("IMAGE_FILE", ""))
        out.append(
            {
                "title": row.get("TITLE", ""),
                "author": row.get("AUTHOR", ""),
                "date": str(row.get("DATE", "")),
                "timeframe": str(row.get("TIMEFRAME", "")),
                "image_file": image_file,
                "image_path": _resolve_path(image_file),
            }
        )
    return out


def lookup_images(
    title: Optional[str] = None,
    author: Optional[str] = None,
    timeframe: Optional[str] = None,
    top_k: int = 3,
) -> list[dict]:
    """底层实现，供节点直接调用（绕过 @tool 包装）。"""
    from src.data.loader import get_dataset

    df = get_dataset().all

    if title:
        stripped = title.strip()
        for article in ("the ", "a ", "an "):
            if stripped.lower().startswith(article):
                stripped = stripped[len(article) :]
                break
        df = df[
            df["TITLE"].str.lower().str.contains(stripped.lower(), na=False, regex=False)
        ]
    if author:
        # 取最长词（通常是姓）优先匹配，处理 "Vincent van Gogh" / "Van Gogh"
        tokens = sorted(
            [t for t in author.strip().split() if len(t) > 2], key=len, reverse=True
        )
        mask = None
        for token in tokens:
            cand = df["AUTHOR"].str.lower().str.contains(
                token.lower(), na=False, regex=False
            )
            if cand.any():
                mask = cand
                break
        if mask is not None:
            df = df[mask]
        elif not tokens:
            df = df[
                df["AUTHOR"].str.lower().str.contains(
                    author.lower(), na=False, regex=False
                )
            ]
    if timeframe:
        df = df[
            df["TIMEFRAME"].str.lower().str.contains(
                timeframe.lower(), na=False, regex=False
            )
        ]

    # 只保留图片文件真实存在的记录
    df = df.copy()
    if df.empty:
        return []
    return _rows_to_images(df, top_k)


@tool
def image_lookup(
    title: Optional[str] = None,
    author: Optional[str] = None,
    timeframe: Optional[str] = None,
    top_k: int = 3,
) -> list[dict]:
    """
    从 SemArt 本地图片库查找画作配图（只定位，不做视觉分析）。

    适用场景：
      - 需要为某画家/某时期的叙述配上代表作品图
      - 按标题定位一幅画的图片文件

    Args:
        title:     画作标题（部分匹配）
        author:    画家姓名（部分匹配）
        timeframe: 时期，如 "1851-1900"
        top_k:     返回数量（默认3）

    Returns:
        画作列表，每项含 title / author / date / timeframe / image_file / image_path
    """
    return lookup_images(title, author, timeframe, top_k)
