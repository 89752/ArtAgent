"""
入库流水线的共享数据结构。

Block：解析器（pdfplumber 兜底 / MinerU 主力）产出的语义块，
       带类型、页码、所属小节——chunker 按类型分流归并/拆分。
Chunk：最终入库的检索单元，统一携带 doc_id/page_id/block_type/section/kb_id。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlockType = Literal["text", "table", "equation", "image"]


@dataclass
class Block:
    """解析器产出的一个语义块。"""

    type: BlockType
    content: str  # text: 段落文本；table: HTML；equation: LaTeX；image: 图片路径
    page_idx: int  # 0 基页码
    section: str = ""  # 所属标题/小节（pdfplumber 无法可靠识别，留空；
    #                     MinerU 保留 heading 结构后填充）
    bbox: tuple = ()


@dataclass
class Chunk:
    """入库的检索单元（文字路线）。"""

    content: str
    doc_id: str
    page_idx: int
    block_type: str  # text / table / equation / image_caption
    section: str = ""
    kb_id: str = "default"
    chunk_idx: int = 0  # 文档内序号，做 Chroma id 用

    @property
    def page_id(self) -> str:
        """跨路线去重键：同一页的文字 chunk 与整页图共享同一 page_id。"""
        return f"{self.doc_id}-p{self.page_idx}"

    def chroma_id(self) -> str:
        return f"{self.page_id}-c{self.chunk_idx}"

    def metadata(self, doc_name: str = "") -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_name": doc_name,
            "page_id": self.page_id,
            "page": self.page_idx + 1,  # 展示用 1 基页码
            "block_type": self.block_type,
            "section": self.section,
            "kb_id": self.kb_id,
        }
