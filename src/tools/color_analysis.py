"""color_analysis：本地图像计算工具（零 LLM / 零 API）。

对已定位的画作图片做确定性结构分析：
- 主色调（K-means 量化 k=5，PIL 内置 MEDIANCUT）
- 明度/对比度（灰度均值 / 标准差）
- 饱和度（HSV 均值）
- 构图网格（3×3 单元亮度标准差的不对称性）

输出只提供"结构性数值/标签"，审美判断留给对话模型组织。
支持本地图片（data/core/images，回退 SemArt/Images）与网络 URL。
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
from langchain_core.tools import tool
from PIL import Image

from src.utils.http import download_bytes


def _dominant_colors(img: Image.Image, k: int = 5) -> list[dict]:
    """缩略图量化取主色调（确定性、可单测）。"""
    small = img.convert("RGB").copy()
    small.thumbnail((160, 160))
    quantized = small.quantize(colors=k, method=Image.MEDIANCUT).convert("RGB")
    pixels = np.array(quantized).reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    order = np.argsort(-counts)
    total = int(counts.sum()) or 1
    out = []
    for i in order[:k]:
        r, g, b = (int(v) for v in colors[i])
        out.append(
            {
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "ratio": round(float(counts[i]) / total, 3),
            }
        )
    return out


def _brightness_contrast(img: Image.Image) -> tuple[str, str]:
    gray = np.asarray(img.convert("L"), dtype=float) / 255.0
    mean, std = float(gray.mean()), float(gray.std())
    brightness = "high" if mean > 0.65 else ("low" if mean < 0.35 else "medium")
    contrast = "high" if std > 0.28 else ("low" if std < 0.12 else "medium")
    return brightness, contrast


def _saturation(img: Image.Image) -> str:
    hsv = np.asarray(img.convert("HSV"), dtype=float)
    s = float(hsv[..., 1].mean()) / 255.0
    return "vivid" if s > 0.5 else ("muted" if s < 0.2 else "moderate")


def _composition_grid(img: Image.Image) -> str:
    """3×3 单元亮度标准差：单元差异大 → dynamic，否则 balanced。"""
    gray = np.asarray(img.convert("L"), dtype=float)
    h, w = gray.shape
    cells = []
    for r in range(3):
        for c in range(3):
            cell = gray[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3]
            if cell.size:
                cells.append(float(cell.std()))
    if not cells:
        return "unknown"
    cells.sort()
    # 显著活跃单元数 vs 平稳单元：用中位数两侧的离散度近似
    spread = cells[-1] - cells[0]
    return "dynamic" if spread > 0.25 else "balanced"


@tool
def color_analysis(
    title: Optional[str] = None,
    author: Optional[str] = None,
    top_k: int = 1,
) -> list[dict]:
    """对画作图片做颜色/明度/饱和度/构图的结构性分析（本地计算，免费快速）。

    适用场景：用户问"这幅画的主色调""色彩偏暖还是偏冷""明暗对比强不强"
    "画面构图是否平衡"等可量化的视觉属性。审美结论由对话模型结合本工具
    的结构化数值给出。

    Args:
        title:  画作标题（部分匹配）
        author: 画家姓名（部分匹配）
        top_k:  分析前几幅匹配画作（默认1）

    Returns:
        每幅画：{title, author, date, success, dominant_colors[{hex,ratio}],
        brightness_contrast, saturation, composition_grid}；
        支持本地图片（data/core/images，回退 SemArt/Images）与网络 URL；
        下载/读取失败时 success=False。
    """
    from src.tools.image_lookup import lookup_images

    located = lookup_images(title=title, author=author, top_k=top_k)
    out: list[dict] = []
    for d in located:
        path = str(d.get("image_path") or "")
        if not path:
            out.append(
                {
                    **{k: d.get(k) for k in ("title", "author", "date")},
                    "success": False,
                    "error": "未找到可用图片",
                }
            )
            continue
        try:
            if path.startswith(("http://", "https://")):
                img = Image.open(BytesIO(download_bytes(path)))
            else:
                img = Image.open(path)
            brightness, contrast = _brightness_contrast(img)
            out.append(
                {
                    **{k: d.get(k) for k in ("title", "author", "date")},
                    "success": True,
                    "dominant_colors": _dominant_colors(img),
                    "brightness_contrast": f"{brightness} / {contrast}",
                    "saturation": _saturation(img),
                    "composition_grid": _composition_grid(img),
                }
            )
        except Exception as e:  # noqa: BLE001
            out.append(
                {
                    **{k: d.get(k) for k in ("title", "author", "date")},
                    "success": False,
                    "error": f"图片分析失败：{e}",
                }
            )
    return out
