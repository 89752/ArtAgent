"""
Tool 2: Image Analysis Tool

使用 qwen3.5-omni-plus 分析画作图像，提取：
  - 视觉内容描述
  - 风格特征
  - 构图分析
  - 色彩分析
"""

import base64
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


def _get_vision_llm() -> ChatOpenAI:
    """返回支持视觉的模型实例。"""
    return ChatOpenAI(
        model="qwen3.5-omni-plus",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0.3,
    )


def _get_image_path(image_file: str) -> Path:
    """根据 IMAGE_FILE 字段解析完整路径。"""
    data_dir = Path(os.getenv("SEMART_DATA_DIR", "./SemArt"))
    return data_dir / "Images" / image_file


def _image_to_base64(image_path: Path) -> str:
    """将图片文件转为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _find_image_file(artwork_query: str) -> Optional[str]:
    """
    按标题或文件名查找 IMAGE_FILE。
    优先精确匹配文件名，其次模糊匹配标题。
    """
    from src.data.loader import get_dataset

    df = get_dataset().all

    # 1. 精确匹配文件名
    exact = df[df["IMAGE_FILE"].str.lower() == artwork_query.lower()]
    if not exact.empty:
        return exact.iloc[0]["IMAGE_FILE"]

    # 2. 去冠词后模糊匹配标题
    stripped = artwork_query.strip()
    for article in ("the ", "a ", "an "):
        if stripped.lower().startswith(article):
            stripped = stripped[len(article) :]
            break

    contains = df[
        df["TITLE"].str.lower().str.contains(stripped.lower(), na=False, regex=False)
    ]
    if not contains.empty:
        row = contains.iloc[0]
        return row["IMAGE_FILE"]

    return None


@tool
def analyze_image(
    artwork_query: str,
    analysis_focus: str = "general",
) -> dict:
    """
    使用视觉模型分析画作图像。支持按画作标题或文件名查找。

    适用场景：
      - 分析画作的视觉内容和构图（如"分析一下《星夜》的构图"）
      - 识别绘画风格和技法特征（如"这幅画的笔触有什么特点"）
      - 提取色彩、光影等视觉元素
      - 用户要求"看图分析"某幅具体画作时

    Args:
        artwork_query:   画作标题或图片文件名（支持英文、部分匹配）
        analysis_focus:  分析侧重点，可选：
                         "general"     - 综合分析（默认）
                         "style"       - 风格与技法分析
                         "composition" - 构图与色彩分析
                         "content"     - 内容与主题分析

    Returns:
        包含视觉分析结果、元数据和数据集描述的字典
    """
    from src.data.loader import get_dataset

    # 1. 查找 image_file
    image_file = _find_image_file(artwork_query)
    if image_file is None:
        return {
            "success": False,
            "error": f"未在数据集中找到画作：'{artwork_query}'，请尝试英文标题。",
        }

    # 2. 读取元数据
    df = get_dataset().all
    row = df[df["IMAGE_FILE"] == image_file].iloc[0]
    metadata = row.to_dict()

    # 3. 检查图片文件
    image_path = _get_image_path(image_file)
    if not image_path.exists():
        return {
            "success": False,
            "error": f"图片文件不存在：{image_path}",
            "metadata": metadata,
        }

    # 4. 构建 prompt
    focus_prompts = {
        "general": (
            "请综合分析这幅画：\n"
            "1) 画面主题与内容\n"
            "2) 艺术风格与流派\n"
            "3) 构图与空间布局\n"
            "4) 色彩与光影运用\n"
            "5) 整体情感与艺术价值"
        ),
        "style": (
            "请重点分析这幅画的艺术风格与技法：\n"
            "1) 风格归属（如巴洛克、印象派、文艺复兴等）\n"
            "2) 笔触与肌理特征\n"
            "3) 技法特点（厚涂法、明暗法等）\n"
            "4) 与同时期画家风格的关联"
        ),
        "composition": (
            "请重点分析这幅画的构图与视觉元素：\n"
            "1) 构图结构与平衡感\n"
            "2) 透视与空间深度\n"
            "3) 色彩搭配与冷暖对比\n"
            "4) 光影处理（明暗对比）\n"
            "5) 视觉引导与焦点"
        ),
        "content": (
            "请重点分析这幅画的内容与主题：\n"
            "1) 画面中的人物、场景或物体\n"
            "2) 叙事性或故事背景\n"
            "3) 象征元素与寓意\n"
            "4) 历史或神话背景\n"
            "5) 核心主题与艺术家意图"
        ),
    }

    prompt = focus_prompts.get(analysis_focus, focus_prompts["general"])

    # 附上已知元数据引导模型
    metadata_hint = (
        f"已知信息：标题《{metadata.get('TITLE')}》，"
        f"画家：{metadata.get('AUTHOR')}，"
        f"年代：{metadata.get('DATE')}，"
        f"技法：{metadata.get('TECHNIQUE')}，"
        f"流派：{metadata.get('SCHOOL')}。\n\n"
        f"{prompt}\n请用中文回答。"
    )

    # 5. 调用视觉模型
    image_ext = image_path.suffix.lstrip(".").lower()
    if image_ext == "jpg":
        image_ext = "jpeg"

    image_b64 = _image_to_base64(image_path)

    try:
        llm = _get_vision_llm()
        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"},
                },
                {"type": "text", "text": metadata_hint},
            ]
        )
        response = llm.invoke([msg])
        analysis_text = response.content
    except Exception as e:
        analysis_text = f"视觉分析失败：{e}"

    return {
        "success": True,
        "image_file": image_file,
        "title": metadata.get("TITLE"),
        "author": metadata.get("AUTHOR"),
        "date": metadata.get("DATE"),
        "technique": metadata.get("TECHNIQUE"),
        "school": metadata.get("SCHOOL"),
        "dataset_description": metadata.get("DESCRIPTION", ""),
        "analysis_focus": analysis_focus,
        "analysis": analysis_text,
    }
