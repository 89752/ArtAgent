"""
文字路线兜底解析器：pdfplumber（Stage 3）。

定位：MinerU 为主力（版面理解、表格/公式识别），pdfplumber 仅在
MinerU 环境不可用时兜底——它没有版面理解能力，产出全部为 text 块，
section 无法可靠识别（留空）。公式密集页不接受本解析器
（page_classifier 标 force_mineru，pipeline 把它退到多模态整页图）。

与 MinerU 的接口约定：解析器输入 PDF 路径 + 页码列表，产出 list[Block]；
MinerU 接入时实现同签名函数即可替换（pipeline 按可用性选择）。
"""

from __future__ import annotations

import logging
import re

from src.ingestion.blocks import Block
from src.utils.logging_config import get_logger

logger = get_logger("ingestion.pdfplumber")

# pdfminer 对缺 FontBBox 的中文字体会逐条刷 warning，无诊断价值，压到 ERROR
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# 连续两个及以上换行视为段落分隔
_PARA_SPLIT = re.compile(r"\n\s*\n")


def parse_pages(pdf_path: str, page_nos: list[int]) -> list[Block]:
    """用 pdfplumber 抽取指定页的正文段落，产出 text 块。"""
    import pdfplumber

    blocks: list[Block] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no in page_nos:
            if page_no >= len(pdf.pages):
                continue
            text = pdf.pages[page_no].extract_text() or ""
            for para in _PARA_SPLIT.split(text):
                # 段落内换行折叠为空格（pdfplumber 按物理行断行）
                content = re.sub(r"\s*\n\s*", " ", para).strip()
                if content:
                    blocks.append(
                        Block(type="text", content=content, page_idx=page_no)
                    )
    logger.info(
        "[pdfplumber] %s pages=%d → blocks=%d", pdf_path, len(page_nos), len(blocks)
    )
    return blocks
