"""compare_images：两幅画同帧喂视觉模型做结构化对比。

一次视觉调用（复用 image_lookup 的视觉模型），按 focus 分节输出对比；
定位/读取失败返回结构化错误，不中断整轮推理。
"""

from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.utils.http import load_image_bytes


_COMPARE_FOCUS_PROMPTS = {
    "general": (
        "请对比这两幅画的整体风格、构图、色彩与主题，指出最显著的异同。"
    ),
    "brushwork": (
        "请重点对比这两幅画的笔触与肌理：笔触类型（厚涂/薄染/点彩/晕染）、"
        "表面质感、技法差异。"
    ),
    "color": (
        "请重点对比这两幅画的色彩：主色调与冷暖倾向、饱和度、明暗对比、"
        "色彩的功能（写实/情感/象征）。"
    ),
    "composition": (
        "请重点对比这两幅画的构图：空间布局、透视、视觉焦点、平衡与引导线。"
    ),
}


def _locate(title: str) -> tuple[dict, str]:
    """定位画作；返回 (画作字典, 图片路径或错误)。"""
    from src.tools.image_lookup import lookup_images

    hits = lookup_images(title=title, top_k=1)
    if not hits:
        return {}, f"未定位到画作：{title}"
    path = str(hits[0].get("image_path") or "")
    if not path:
        return hits[0], "未定位到可用图片"
    return hits[0], ""


def _image_block(path: str) -> dict:
    data, ext = load_image_bytes(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{ext};base64,{b64}"},
    }


@tool
def compare_images(
    title_a: str,
    title_b: str,
    focus: str = "general",
) -> dict:
    """把两幅画同帧交给视觉模型做结构化对比（一次视觉 API 调用，慢且有成本）。

    适用场景：用户明确要求"对比这两幅画的笔触/色彩/构图"等视觉层面的差异，
    且两幅画都能定位到图片（本地或网络 URL）。仅元数据/背景对比请用
    art_comparison 技能（skill_art_comparison）。

    Args:
        title_a: 第一幅画标题（部分匹配）
        title_b: 第二幅画标题（部分匹配）
        focus:   对比侧重点："general"（默认）/ "brushwork" / "color" / "composition"

    Returns:
        {success, a, b, focus, comparison}；定位/读取失败返回
        {success: false, error}。
    """
    if focus not in _COMPARE_FOCUS_PROMPTS:
        focus = "general"
    a, err_a = _locate(title_a)
    b, err_b = _locate(title_b)
    if err_a or err_b:
        return {
            "success": False,
            "error": err_a or err_b,
            "a": a,
            "b": b,
        }

    prompt = _COMPARE_FOCUS_PROMPTS[focus] + (
        "\n\n第一幅画：{title_a}（{author_a}，{date_a}）\n"
        "第二幅画：{title_b}（{author_b}，{date_b}）\n"
        "请分点对比，最后给出一句话总结。用中文回答。"
    ).format(
        title_a=a.get("title", ""), author_a=a.get("author", ""), date_a=a.get("date", ""),
        title_b=b.get("title", ""), author_b=b.get("author", ""), date_b=b.get("date", ""),
    )

    from src.utils.llm import get_vision_llm

    try:
        msg = HumanMessage(
            content=[
                _image_block(str(a["image_path"])),
                _image_block(str(b["image_path"])),
                {"type": "text", "text": prompt},
            ]
        )
        response = get_vision_llm().invoke([msg])
        comparison = str(response.content)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"视觉对比失败：{e}", "a": a, "b": b}

    return {
        "success": True,
        "a": a,
        "b": b,
        "focus": focus,
        "comparison": comparison,
    }
