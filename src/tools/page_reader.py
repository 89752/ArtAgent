"""
Tool: Read Page Image（Qwen-VL 读图，Stage 3 遗留项落地）

当 semantic_search 命中用户上传 PDF 的整页图（source=user_pdf_image，
结果里带 image_path 与 read_hint）时，Agent 调用本工具让视觉模型
（qwen3.5-omni-plus）真正读取页面内容，返回文字描述。

架构分工：对话模型（glm-4.7）是纯文本大脑，负责工具决策；
所有"看见图片"的工作都显式经由视觉模型工具完成——本工具的
存在意义就是视觉读取，不属于"工具内偷藏 LLM 调用"。

成本提示：每次调用 = 一次视觉模型 API 请求，是命中图片路线时的
主要生成开销；log_event 记录调用情况供成本观测。
"""

import base64
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.utils.logging_config import get_logger, log_event

load_dotenv()

logger = get_logger("page_reader")

# 允许读取的根目录（只放行用户上传文档的页面图，防路径穿越）
_UPLOADS_ROOT = Path(os.getenv("UPLOADS_DIR", "./data/uploads")).resolve()
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _validate_image_path(image_path: str) -> tuple[Optional[Path], Optional[str]]:
    """校验路径合法：存在、是图片、位于 uploads 根目录内。"""
    if not image_path:
        return None, "image_path 为空"
    path = Path(image_path).resolve()
    try:
        path.relative_to(_UPLOADS_ROOT)
    except ValueError:
        return None, f"路径不在允许范围内（须位于 {_UPLOADS_ROOT} 下）"
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None, f"不支持的图片类型：{path.suffix}"
    if not path.exists():
        return None, f"图片文件不存在：{path}"
    return path, None


def _resolve_page_path(doc_name: str, page: int) -> tuple[Optional[Path], Optional[str]]:
    """按文档名 + 页码（1 基）定位整页图路径（不依赖语义检索先命中）。"""
    from src.data.documents_store import list_documents

    name = (doc_name or "").strip()
    if not name or not page or page < 1:
        return None, "需要 doc_name 与 page（页码从 1 开始）"
    docs = list_documents()
    candidates = [
        d for d in docs
        if name == (d.get("doc_name") or "").strip()
        or name in (d.get("doc_name") or "")
    ]
    if not candidates:
        return None, f"未找到文档：{doc_name}"
    doc = candidates[0]
    rel = (
        f"{doc.get('kb_id') or 'default'}/{doc.get('doc_id')}/pages/"
        f"page-{page - 1}.png"
    )
    return _validate_image_path(str((_UPLOADS_ROOT / rel).resolve()))


def read_page_image_impl(
    image_path: str = "",
    doc_name: str = "",
    page: int = 0,
    question: str = "",
) -> dict:
    """底层实现（绕过 @tool 包装，供测试直接调用）。"""
    path, error = None, None
    if image_path:
        path, error = _validate_image_path(image_path)
    else:
        path, error = _resolve_page_path(doc_name, page)
    if error:
        return {"success": False, "error": error, "image_path": image_path}

    from src.utils.llm import get_vision_llm

    question_hint = (
        f"\n用户的问题：{question}\n请侧重提取与问题相关的信息。"
        if question
        else ""
    )
    prompt = (
        "这是一份文档的整页图片（来自用户上传的 PDF）。请仔细阅读并输出该页的全部有用信息：\n"
        "1. 页面上的文字内容（转录要点）\n"
        "2. 图像/图版/图表的内容描述（主体、风格、细节）\n"
        f"{question_hint}\n"
        "用中文回答，分小节输出。"
    )

    suffix = path.suffix.lstrip(".").lower()
    if suffix == "jpg":
        suffix = "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    try:
        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{suffix};base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        )
        response = get_vision_llm().invoke([msg])
        description = response.content
    except Exception as e:
        logger.warning("[read_page_image] 视觉读取失败 %s: %s", path, e)
        return {
            "success": False,
            "error": f"视觉读取失败：{e}",
            "image_path": str(path),
        }

    log_event(
        logger, "read_page_image",
        image=path.name, question=question[:40], desc_len=len(description),
    )
    return {
        "success": True,
        "image_path": str(path),
        "page_description": description,
    }


@tool
def read_page_image(
    image_path: Optional[str] = None,
    doc_name: Optional[str] = None,
    page: Optional[int] = None,
    question: str = "",
) -> dict:
    """
    用视觉模型读取用户上传文档的整页图片内容。

    适用场景：用户上传的文档是纯图片/扫描件（无文字索引），或 semantic_search
    命中 source=user_pdf_image（整页图）需要读取图面内容时。两种定位方式任选其一：
    直接给 image_path，或用 doc_name + page（页码 1 基）按文档定位。

    Args:
        image_path: 检索结果中给出的整页图路径
        doc_name:   文档名称（与 page 组合定位页面）
        page:       页码，从 1 开始
        question:   用户的原始问题（可选，提供后视觉模型侧重提取相关信息）

    Returns:
        {"success": bool, "page_description": 页面内容的文字描述（文字转录+图像描述）}
    """
    return read_page_image_impl(image_path or "", doc_name or "", page or 0, question)
