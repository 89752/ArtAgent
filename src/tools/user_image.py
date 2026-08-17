"""用户上传图片的通用读图与分析工具（聊天入口的引擎适配器）。"""

from __future__ import annotations

import base64
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.utils.logging_config import get_logger, log_event

logger = get_logger("tools.user_image")


def _guard_image(image_id: str, session_id: str = ""):
    """校验图片存在且（可选）属于当前会话。"""
    from src.analysis.store import get_image

    rec = get_image(image_id)
    if not rec:
        return None, {"success": False, "error": "图片不存在或已删除"}
    if session_id and rec.get("session_id") and session_id != rec.get("session_id"):
        return None, {"success": False, "error": "无权访问该图片"}
    return rec, None


_READ_PROMPTS = {
    "general": "请综合描述这张用户上传的图片：画面主体、内容、风格与技法要点。",
    "content": "请描述画面中的主体、人物/物体、场景与可辨识的内容细节。",
    "style": "请分析画面的艺术风格、笔触与流派倾向。",
    "technique": "请分析画面技法：透视、比例、明暗、结构是否有明显问题。",
    "composition": "请分析构图：主体位置、平衡、留白与视觉引导。",
    "color": "请分析色彩：色相搭配、明度对比、饱和度与冷暖关系。",
}


@tool
def read_user_image(
    image_id: str,
    focus: str = "general",
    session_id: str = "",
) -> dict:
    """用视觉模型读取用户上传图片的实际内容（通用读图）。

    适用场景：用户要求“看看/描述/分析”他自己上传的图片时。
    focus: general / content / style / technique / composition / color。
    image_id 与 会话ID 来自上下文【session】块的“用户图片/会话ID”字段。
    """
    rec, err = _guard_image(image_id, session_id)
    if err:
        return err
    if focus not in _READ_PROMPTS:
        focus = "general"

    from src.utils.http import load_image_bytes

    try:
        data, ext = load_image_bytes(str(rec["file_path"]))
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"图片读取失败：{e}"}

    prompt = _READ_PROMPTS[focus] + "\n用中文回答，客观描述画面本身。"
    b64 = base64.b64encode(data).decode("ascii")
    try:
        from src.utils.llm import get_vision_llm

        llm = get_vision_llm()
        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        )
        description = str(llm.invoke([msg]).content)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"视觉读取失败：{e}"}
    log_event(logger, "read_user_image", image_id=image_id, focus=focus)
    return {
        "success": True,
        "image_id": image_id,
        "focus": focus,
        "description": description,
    }


@tool
def analyze_user_artwork(
    image_id: str,
    focus: str = "all",
    framework_override: Optional[str] = None,
    session_id: str = "",
) -> dict:
    """对用户上传的画作运行结构化技法与美学分析（复用分析引擎）。

    focus=all 产出完整三层报告；定向 focus 只跑相关阶段，省一次视觉调用。
    framework_override 可取 realistic / abstract / childlike / decorative，
    用于用户纠正自动框架判定。结果会写入报告缓存，供后续追问直接引用。
    """
    rec, err = _guard_image(image_id, session_id)
    if err:
        return err

    from src.analysis.engine import run_analysis

    result: dict = {"success": False, "error": "分析未返回结果"}
    for evt in run_analysis(
        image_id, focus=focus, framework_override=framework_override
    ):
        etype = evt.get("type")
        if etype == "error":
            return {"success": False, "error": evt.get("message", "分析失败")}
        if etype == "rejected":
            return {
                "success": False,
                "rejected": True,
                "reason": evt.get("reason", ""),
                "guide": evt.get("guide", ""),
            }
        if etype == "done":
            result = {
                "success": True,
                "image_id": image_id,
                "framework": evt.get("report", {}).get("framework", ""),
                "report": evt.get("report", {}),
                "metrics": evt.get("metrics", {}),
            }
    return result
