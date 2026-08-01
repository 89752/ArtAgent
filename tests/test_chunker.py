# tests/test_chunker.py
"""
分块器（src/ingestion/chunker.py）纯单测：
构造 Block 序列验证归并/拆分/整块保留规则，不依赖真实 PDF，秒级完成。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.blocks import Block
from src.ingestion.chunker import (
    EQUATION_CTX,
    MERGE_MIN,
    SPLIT_MAX,
    chunk_blocks,
)


def _text(content, page=0, section=""):
    return Block(type="text", content=content, page_idx=page, section=section)


# ── text 归并 ────────────────────────────────────────────────────
def test_merge_short_paragraphs_to_min():
    blocks = [_text("短句一。" * 5, page=0), _text("短句二。" * 5, page=0),
              _text("短句三。" * 5, page=0)]
    chunks = chunk_blocks(blocks, "d1")
    # 三段合计 ~45 字符 < MERGE_MIN → 归并为一个 chunk
    assert len(chunks) == 1
    assert "短句一" in chunks[0].content and "短句三" in chunks[0].content


def test_merge_stops_after_reaching_min():
    long_para = "这段已经足够长。" * 30  # ~210 字符
    blocks = [_text(long_para, page=0), _text("后续短段。", page=0)]
    chunks = chunk_blocks(blocks, "d1")
    # 第一段 ≥ MERGE_MIN → 独立成块；尾部短段不足下限但也成块（文档末尾 flush）
    assert len(chunks) == 2
    assert chunks[0].content == long_para
    assert chunks[1].content == "后续短段。"


def test_no_merge_across_pages():
    blocks = [_text("第一页短。", page=0), _text("第二页短。", page=1)]
    chunks = chunk_blocks(blocks, "d1")
    assert len(chunks) == 2
    assert chunks[0].page_idx == 0 and chunks[1].page_idx == 1


def test_no_merge_across_sections():
    blocks = [_text("小节A短。", page=0, section="A"), _text("小节B短。", page=0, section="B")]
    chunks = chunk_blocks(blocks, "d1")
    assert len(chunks) == 2


def test_page_id_and_metadata():
    chunks = chunk_blocks([_text("内容", page=3)], "doc9", kb_id="kb1")
    c = chunks[0]
    assert c.page_id == "doc9-p3"
    meta = c.metadata(doc_name="测试.pdf")
    assert meta["page_id"] == "doc9-p3"
    assert meta["page"] == 4  # 展示用 1 基
    assert meta["kb_id"] == "kb1"
    assert meta["doc_name"] == "测试.pdf"
    assert c.chroma_id() == "doc9-p3-c0"


# ── text 长块拆分 ────────────────────────────────────────────────
def test_split_long_block_with_overlap():
    content = "字" * (SPLIT_MAX * 2 + 100)
    chunks = chunk_blocks([_text(content, page=0)], "d1")
    assert len(chunks) >= 3
    assert all(len(c.content) <= SPLIT_MAX for c in chunks)
    # 相邻窗口有重叠：前块结尾 == 后块开头一段
    overlap_len = int(SPLIT_MAX * 0.12)
    assert chunks[0].content[-overlap_len:] == chunks[1].content[:overlap_len]


def test_exact_threshold_not_split():
    chunks = chunk_blocks([_text("字" * SPLIT_MAX, page=0)], "d1")
    assert len(chunks) == 1


# ── table / equation / image 块 ──────────────────────────────────
def test_table_block_kept_whole():
    html = "<table>" + "<tr><td>x</td></tr>" * 100 + "</table>"  # 超长也不拆
    blocks = [Block(type="table", content=html, page_idx=1)]
    chunks = chunk_blocks(blocks, "d1")
    assert len(chunks) == 1
    assert chunks[0].block_type == "table"
    assert chunks[0].content == html


def test_equation_block_with_context():
    prev_text = "根据能量守恒定律，我们可以得到如下表达式。"
    next_text = "这个公式揭示了质量与能量的等价关系。"
    blocks = [
        _text(prev_text, page=0),
        Block(type="equation", content="E = mc^2", page_idx=0),
        _text(next_text, page=0),
    ]
    chunks = chunk_blocks(blocks, "d1")
    eq = [c for c in chunks if c.block_type == "equation"]
    assert len(eq) == 1
    assert "E = mc^2" in eq[0].content
    assert "能量守恒" in eq[0].content  # 前文上下文
    assert "等价关系" in eq[0].content  # 后文上下文


def test_equation_context_truncated_to_limit():
    long_text = "字" * 500
    blocks = [
        _text(long_text, page=0),
        Block(type="equation", content="x=1", page_idx=0),
        _text(long_text, page=0),
    ]
    eq = [c for c in chunk_blocks(blocks, "d1") if c.block_type == "equation"][0]
    # 上下文各 ≤ EQUATION_CTX，总长有限
    assert len(eq.content) <= EQUATION_CTX * 2 + len("x=1") + 2


def test_image_caption_block_independent():
    blocks = [
        _text("正文段落。", page=0),
        Block(type="image", content="画中是一位戴珍珠耳环的少女。", page_idx=0),
    ]
    chunks = chunk_blocks(blocks, "d1")
    cap = [c for c in chunks if c.block_type == "image_caption"]
    assert len(cap) == 1
    assert "珍珠耳环" in cap[0].content


def test_chunk_idx_sequential():
    blocks = [_text("a" * 200, page=0), Block(type="table", content="<t/>", page_idx=1)]
    chunks = chunk_blocks(blocks, "d1")
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 chunker 全部 {len(fns)} 个单测通过！")
