"""S1 适用框架门控（视觉模型 #1，结构化输出）。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.analysis.prompts import FRAMEWORKS, GATE_PROMPT
from src.utils.json_utils import parse_json
from src.utils.logging_config import get_logger, log_event

logger = get_logger("analysis.gate")


def classify_framework(image_b64: str, image_ext: str, retries: int = 1) -> dict:
    """对单张用户图片做框架判定；解析失败返回 framework=unknown。"""
    from src.utils.llm import get_vision_llm

    msg = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"},
            },
            {"type": "text", "text": GATE_PROMPT},
        ]
    )
    last_error = ""
    for attempt in range(retries + 1):
        try:
            resp = get_vision_llm().invoke([msg])
            data = parse_json(str(resp.content))
            if isinstance(data, dict) and data.get("framework") in FRAMEWORKS:
                flags = data.get("quality_flags") or []
                if not isinstance(flags, list):
                    flags = []
                try:
                    confidence = float(data.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                out = {
                    "framework": data["framework"],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": str(data.get("reason") or ""),
                    "quality_flags": [str(x) for x in flags],
                    "content_summary": str(data.get("content_summary") or ""),
                }
                log_event(
                    logger, "gate_ok", framework=out["framework"],
                    confidence=out["confidence"],
                )
                return out
            last_error = "输出缺少合法 framework 字段"
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
    log_event(logger, "gate_failed", error=last_error)
    return {
        "framework": "unknown",
        "confidence": 0.0,
        "reason": f"框架判定失败：{last_error}",
        "quality_flags": [],
        "content_summary": "",
    }
