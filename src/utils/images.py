"""本地画作图片目录与路径解析（core 镜像优先，SemArt 回退）。"""

from __future__ import annotations

import os
from pathlib import Path

CORE_CSV_PATH = Path(os.getenv("CORE_DATA_PATH", "./data/core/artworks_core.csv"))
CORE_IMAGES_DIR = CORE_CSV_PATH.parent / "images"
SEMART_IMAGES_DIR = Path(os.getenv("SEMART_DATA_DIR", "./SemArt")) / "Images"


def artwork_image_bases() -> tuple[Path, Path]:
    """图片目录查找顺序：data/core/images → SemArt/Images。"""
    return (CORE_IMAGES_DIR, SEMART_IMAGES_DIR)


def resolve_artwork_image(image_file: str) -> str:
    """把图片文件名解析成完整本地路径；找不到返回空串。

    安全：只接受纯文件名（basename），且解析结果必须落在允许的图片根目录内，
    防止 LLM 可控参数通过 ../ 读取任意本地文件。
    """
    if not image_file:
        return ""
    name = Path(image_file).name
    if not name or name in (".", ".."):
        return ""
    for base in artwork_image_bases():
        base = base.resolve()
        p = (base / name).resolve()
        if p.parent == base and p.is_file():
            return str(p)
    return ""
