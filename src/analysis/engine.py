"""分析引擎：S0–S4 可组合阶段的编排（生成器，支持 SSE 进度与同步调用）。"""

from __future__ import annotations

import base64
import json
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

from src.analysis.gate import classify_framework
from src.analysis.metrics import analyze_metrics
from src.analysis.report import generate_layered_report
from src.analysis.store import (
    get_image,
    save_analysis,
    update_image_status,
)
from src.utils.logging_config import get_logger, log_event

logger = get_logger("analysis.engine")

USER_IMAGE_ROOT = Path(os.getenv("USER_IMAGES_DIR", "./data/uploads/user_images"))
MAX_ANALYSIS_SIDE = 1600
ALLOWED_FOCUS = {"all", "perspective", "composition", "color", "brushwork", "style"}
ALLOWED_OVERRIDE = {"realistic", "abstract", "childlike", "decorative"}


class AnalysisError(Exception):
    """分析链路中可预期的失败（面向用户展示）。"""


def _safe_image_path(file_path: str) -> Path:
    p = Path(file_path).resolve()
    root = USER_IMAGE_ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise AnalysisError("图片路径不在允许目录内") from e
    if not p.is_file():
        raise AnalysisError("图片文件不存在或已删除")
    return p


def _quality_flags(img: Image.Image) -> list[str]:
    flags: list[str] = []
    w, h = img.size
    if min(w, h) < 400:
        flags.append("low_resolution")
    gray = np.asarray(img.convert("L"), dtype=float) / 255.0
    mean = float(gray.mean())
    if mean > 0.86:
        flags.append("too_bright")
    elif mean < 0.12:
        flags.append("too_dark")
    # 近似模糊检测：边缘幅度方差过低视为模糊
    gy, gx = np.gradient(gray)
    edge_var = float((gx**2 + gy**2).var())
    if edge_var < 0.002:
        flags.append("blurry")
    if w / h > 4 or h / w > 4:
        flags.append("unusual_ratio")
    return flags


def preprocess_image(data: bytes) -> tuple[Image.Image, list[str]]:
    """解码、EXIF 归一化、转 RGB、质量检查（分析用副本降采样）。"""
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise AnalysisError(f"无法识别的图片文件：{e}") from e
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    flags = _quality_flags(img)
    copy_img = img.copy()
    copy_img.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE))
    return copy_img.convert("RGB"), flags


def _image_b64(img: Image.Image) -> tuple[str, str]:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "jpeg"


def run_analysis(
    image_id: str,
    focus: str = "all",
    framework_override: str | None = None,
) -> Iterator[dict]:
    """按 S0–S4 顺序产出事件；最后一条为 done/rejected/error。"""
    focus = focus if focus in ALLOWED_FOCUS else "all"
    if framework_override and framework_override not in ALLOWED_OVERRIDE:
        framework_override = None

    rec = get_image(image_id)
    if not rec:
        yield {"type": "error", "message": "图片不存在或已删除"}
        return
    update_image_status(image_id, "analyzing")

    yield {"type": "stage", "stage": "preprocess", "label": "预处理与质量检查"}
    try:
        path = _safe_image_path(str(rec["file_path"] or ""))
        data = path.read_bytes()
        img, quality_flags = preprocess_image(data)
    except AnalysisError as e:
        update_image_status(image_id, "failed", str(e))
        yield {"type": "error", "message": str(e)}
        return

    yield {"type": "stage", "stage": "metrics", "label": "本地色彩度量"}
    metrics = analyze_metrics(img)
    yield {"type": "metrics", **metrics}

    yield {"type": "stage", "stage": "gate", "label": "判定适用框架"}
    image_b64, image_ext = _image_b64(img)
    if framework_override:
        gate = {
            "framework": framework_override,
            "confidence": 1.0,
            "reason": "用户指定框架",
            "quality_flags": quality_flags,
            "content_summary": "",
        }
    else:
        gate = classify_framework(image_b64, image_ext)
        if gate["framework"] == "unknown":
            update_image_status(image_id, "failed", gate["reason"])
            yield {
                "type": "error",
                "message": f"框架判定失败，请重试：{gate['reason']}",
            }
            return
        gate["quality_flags"] = quality_flags

    if gate["framework"] == "not_painting":
        reason = (
            "这看起来不是一张绘画作品（可能是摄影、截图或图表）。"
            "当前功能只分析绘画/手绘图像。"
        )
        update_image_status(image_id, "rejected", reason)
        yield {
            "type": "rejected",
            "reason": reason,
            "guide": "如果你上传的是自己拍的绘画照片，请确保画面端正、光线均匀后重试；"
            "摄影作品的构图分析模式将在后续版本开放。",
        }
        return

    yield {"type": "stage", "stage": "report", "label": "三层分层分析"}
    report = generate_layered_report(
        image_b64,
        image_ext,
        gate,
        metrics,
        focus=focus,
        quality_flags=quality_flags,
    )

    result_dir = USER_IMAGE_ROOT / image_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "result.json"
    payload = {
        "image_id": image_id,
        "gate": gate,
        "metrics": metrics,
        "report": report,
        "focus": focus,
        "framework_override": framework_override,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    update_image_status(image_id, "done")
    save_analysis(
        image_id,
        str(report.get("framework") or gate["framework"]),
        str(result_path),
        {
            "focus": focus,
            "framework_override": framework_override,
            "version": 1,
            "request_id": uuid.uuid4().hex[:12],
        },
    )
    log_event(logger, "analysis_done", image_id=image_id, focus=focus)
    yield {"type": "done", **payload}
