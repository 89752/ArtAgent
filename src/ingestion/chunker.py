"""
文字路线分块（Stage 3）：按 block_type 分流，不做纯文本滑动窗口一刀切。

规则（实施方案 §4.3）：
  text     短段向下归并（同页同小节，下限 ~180 字符）；
           单块过长（>500）才拆，滑动窗口 ~12% 重叠；
           300–500 是"该不该拆"的判断阈值，不是硬指标。
  table    整块入库不拆——行列对应关系拆了就没意义。
  equation 整块 + 前后各 ~80 字符正文上下文（孤立的 LaTeX 语义检索价值低）。
  image    （MinerU 内嵌小图）caption 独立成块，不与正文合并。

归并范围限定同页：chunk 只带一个 page_id，跨页归并会让页码归属失真
（MinerU 接入后放宽到"同一小节内跨页"，本节标题变化时 flush）。
"""

from __future__ import annotations

from src.ingestion.blocks import Block, Chunk

MERGE_MIN = 180  # 归并下限（150–200 取中）
SPLIT_MAX = 500  # 拆分判断阈值（300–500 取上界）
OVERLAP_RATIO = 0.12  # 滑动窗口重叠（10–15% 取中）
EQUATION_CTX = 80  # 公式块前后携带的正文上下文长度


def _split_long_text(content: str) -> list[str]:
    """过长文本块滑动窗口拆分（约 12% 重叠）。"""
    overlap = int(SPLIT_MAX * OVERLAP_RATIO)
    step = SPLIT_MAX - overlap
    parts = []
    start = 0
    while start < len(content):
        parts.append(content[start : start + SPLIT_MAX])
        start += step
    return parts


def chunk_blocks(
    blocks: list[Block], doc_id: str, kb_id: str = "default"
) -> list[Chunk]:
    """把解析器产出的语义块归并/拆分成入库 chunk。"""
    raw: list[tuple[str, int, str, str]] = []  # (content, page_idx, block_type, section)

    buf_parts: list[str] = []
    buf_page = -1
    buf_section = ""

    def flush() -> None:
        nonlocal buf_parts
        if buf_parts:
            raw.append(("\n".join(buf_parts), buf_page, "text", buf_section))
            buf_parts = []

    for i, block in enumerate(blocks):
        if block.type == "table":
            flush()
            raw.append((block.content, block.page_idx, "table", block.section))
            continue
        if block.type == "equation":
            flush()
            prev_ctx = ""
            if i > 0 and blocks[i - 1].type == "text":
                prev_ctx = blocks[i - 1].content[-EQUATION_CTX:]
            next_ctx = ""
            if i + 1 < len(blocks) and blocks[i + 1].type == "text":
                next_ctx = blocks[i + 1].content[:EQUATION_CTX]
            content = "\n".join(p for p in (prev_ctx, block.content, next_ctx) if p)
            raw.append((content, block.page_idx, "equation", block.section))
            continue
        if block.type == "image":
            # MinerU 内嵌小图：content 为 caption（由视觉工具预生成）
            flush()
            raw.append((block.content, block.page_idx, "image_caption", block.section))
            continue

        # text 块
        if len(block.content) > SPLIT_MAX:
            flush()
            for part in _split_long_text(block.content):
                raw.append((part, block.page_idx, "text", block.section))
            continue
        # 页或小节变化 → 先 flush（归并只发生在同页同小节内）
        if buf_parts and (block.page_idx != buf_page or block.section != buf_section):
            flush()
        if not buf_parts:
            buf_page, buf_section = block.page_idx, block.section
        buf_parts.append(block.content)
        if sum(len(p) for p in buf_parts) >= MERGE_MIN:
            flush()

    flush()

    return [
        Chunk(
            content=content,
            doc_id=doc_id,
            page_idx=page_idx,
            block_type=block_type,
            section=section,
            kb_id=kb_id,
            chunk_idx=idx,
        )
        for idx, (content, page_idx, block_type, section) in enumerate(raw)
    ]
