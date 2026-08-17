"""本地确定性视觉度量（零 LLM）。复用 color_analysis 并扩展色相关系/明度/视觉重心。"""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image

from src.tools.color_analysis import (
    _brightness_contrast,
    _composition_grid,
    _dominant_colors,
    _saturation,
)


def _hex_to_hsv(hex_color: str) -> tuple[float, float, float]:
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return 0.0, 0.0, 0.0
    try:
        r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return 0.0, 0.0, 0.0
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    return hh * 360.0, ss, vv


def _hue_relationship(colors: list[dict]) -> dict:
    """基于主色在色环上的角度差判定邻近/互补/三角配色。"""
    hues: list[tuple[float, float, str]] = []
    for c in colors[:5]:
        hue, sat, val = _hex_to_hsv(str(c.get("hex") or ""))
        if sat >= 0.12 and val >= 0.12:
            hues.append((hue, float(c.get("ratio") or 0.0), str(c.get("hex") or "")))
    if len(hues) < 2:
        return {"scheme": "单色或近灰", "detail": "未检测到足够的高饱和主色"}
    best: tuple[float, str, str] | None = None
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            d = abs(hues[i][0] - hues[j][0])
            d = min(d, 360.0 - d)
            if best is None or d < best[0]:
                best = (d, hues[i][2], hues[j][2])
    assert best is not None
    d, c1, c2 = best
    if d <= 30:
        scheme = "邻近色"
    elif d >= 150:
        scheme = "互补色"
    elif 115 <= d <= 135:
        scheme = "三角色"
    else:
        scheme = "类似色/其他"
    return {
        "scheme": scheme,
        "detail": f"{c1} 与 {c2} 色环夹角约 {d:.0f}°",
    }


def _value_tiers(img: Image.Image) -> dict:
    gray = np.asarray(img.convert("L"), dtype=float) / 255.0
    light = float((gray > 0.66).mean())
    dark = float((gray < 0.33).mean())
    mid = 1.0 - light - dark
    label = (
        "亮调为主" if light > 0.5 else ("暗调为主" if dark > 0.5 else "中间调为主")
    )
    return {
        "light": round(light, 3),
        "mid": round(mid, 3),
        "dark": round(dark, 3),
        "label": label,
    }


def _visual_weight(img: Image.Image) -> dict:
    """3×3 网格对比度加权重心（近似视觉重心）。"""
    gray = np.asarray(img.convert("L"), dtype=float)
    h, w = gray.shape
    if h < 3 or w < 3:
        return {"x": 0.5, "y": 0.5, "description": "画面过小，无法估计"}
    cells: list[tuple[float, float, float]] = []
    for r in range(3):
        for c in range(3):
            cell = gray[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3]
            if cell.size:
                cells.append(((c + 0.5) / 3.0, (r + 0.5) / 3.0, float(cell.std())))
    total = sum(wt for _, _, wt in cells) or 1.0
    x = sum(cx * wt for cx, _, wt in cells) / total
    y = sum(cy * wt for _, cy, wt in cells) / total
    h_desc = "偏左" if x < 0.4 else ("偏右" if x > 0.6 else "居中")
    v_desc = "偏上" if y < 0.4 else ("偏下" if y > 0.6 else "居中")
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "description": f"{h_desc}、{v_desc}",
    }


def analyze_metrics(img: Image.Image) -> dict:
    """对已预处理（RGB）的图像做本地确定性度量。"""
    colors = _dominant_colors(img)
    return {
        "dominant_colors": colors,
        "brightness_contrast": _brightness_contrast(img),
        "saturation": _saturation(img),
        "composition_grid": _composition_grid(img),
        "hue_relationship": _hue_relationship(colors),
        "value_tiers": _value_tiers(img),
        "visual_weight": _visual_weight(img),
    }
