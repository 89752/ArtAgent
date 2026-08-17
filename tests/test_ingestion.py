# tests/test_ingestion.py
"""
文档摄入（ingestion）集群合并单测：

- src/ingestion：chunker / mineru_parser / pdf_splitter / ocr_route / page_classifier
- src/tools：page_reader（路径安全 + 按文档名定位整页图）
- 数据集构建脚本：01_harvest_wikidata / 02_build_extended / 03_normalize_core /
  04_assemble_descriptions / 05_audit_core / 06_index_core

全部为纯单测：不联网、不调 LLM、不加载向量模型，秒级完成。
"""

import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import fitz
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.ingestion.blocks import Block
from src.ingestion.chunker import (
    EQUATION_CTX,
    MERGE_MIN,
    SPLIT_MAX,
    chunk_blocks,
)
from src.ingestion.mineru_parser import blocks_from_content_list, mineru_available
from src.ingestion.page_classifier import (
    FORMULA_DENSITY_THRESHOLD,
    PageSignals,
    classify_document,
    classify_page,
    collect_page_signals,
)
from src.ingestion.pdf_splitter import split_pdf
from src.ingestion.pipeline import _parse_ocr_route
from src.tools.page_reader import _validate_image_path, read_page_image_impl

import src.tools.page_reader as pr

# 管线脚本带数字前缀，文件名不是合法模块标识符，统一用 importlib 加载
ad = importlib.import_module("04_assemble_descriptions")
ac = importlib.import_module("05_audit_core")
bed = importlib.import_module("02_build_extended_dataset")
hw = importlib.import_module("01_harvest_wikidata")
nc = importlib.import_module("03_normalize_core")
_select_rows_to_index = importlib.import_module("06_index_core")._select_rows_to_index
import core_utils as cutils


# ══════════════════════════════════════════════════════════════════
# 1. chunker（src/ingestion/chunker.py）
# ══════════════════════════════════════════════════════════════════


def _text(content, page=0, section=""):
    return Block(type="text", content=content, page_idx=page, section=section)


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


# ══════════════════════════════════════════════════════════════════
# 2. mineru_parser（src/ingestion/mineru_parser.py）
# ══════════════════════════════════════════════════════════════════

# 固化 fixture：模拟 MinerU v4 zip 内 *_content_list.json 的条目流
FIXTURE = [
    # 第 0 页：标题 → 正文 → 公式 → 表格 → 带图注的图
    {"type": "text", "text": "第二章 印象派的色彩", "text_level": 1,
     "bbox": [62, 80, 400, 120], "page_idx": 0},
    {"type": "text", "text": "莫奈在吉维尼的花园中反复描绘睡莲，捕捉光影的瞬息变化。",
     "bbox": [62, 140, 900, 200], "page_idx": 0},
    {"type": "equation", "text": "$$E = h\\nu$$", "text_format": "latex",
     "img_path": "images/eq1.jpg", "bbox": [300, 220, 700, 260], "page_idx": 0},
    {"type": "table", "table_caption": ["表 2-1 主要画家年表"],
     "table_body": "<html><body><table><tr><td>莫奈</td><td>1840</td></tr></table></body></html>",
     "table_footnote": ["注：年份为出生年。"],
     "img_path": "images/t1.jpg", "bbox": [62, 300, 900, 500], "page_idx": 0},
    {"type": "image", "img_path": "images/p1.jpg",
     "image_caption": ["图 2-3 《睡莲》 克劳德·莫奈 1906 年 布面油画"],
     "image_footnote": ["现藏于芝加哥艺术学院。"],
     "bbox": [62, 520, 900, 800], "page_idx": 0},
    # 第 1 页：无标题正文（沿用第 0 页小节）+ 无图注的图（应被跳过）+ 页面噪声
    {"type": "text", "text": "雷诺阿则偏爱人物与肌肤的暖色调表现。",
     "bbox": [62, 100, 900, 160], "page_idx": 1},
    {"type": "image", "img_path": "images/p2.jpg", "image_caption": [],
     "image_footnote": [], "bbox": [62, 200, 900, 700], "page_idx": 1},
    {"type": "header", "text": "艺术史讲义", "bbox": [62, 20, 900, 40], "page_idx": 1},
    {"type": "footer", "text": "第 17 页", "bbox": [450, 960, 550, 980], "page_idx": 1},
    {"type": "page_number", "text": "17", "bbox": [480, 960, 520, 980], "page_idx": 1},
    # 第 2 页：二级标题、list、code、空文本
    {"type": "text", "text": "2.1 光的物理", "text_level": 2,
     "bbox": [62, 60, 400, 100], "page_idx": 2},
    {"type": "list", "sub_type": "text",
     "list_items": ["色相", "明度", "饱和度"], "bbox": [62, 120, 900, 240], "page_idx": 2},
    {"type": "code", "sub_type": "code", "code_caption": ["示例"],
     "code_body": "mix(red, blue)", "bbox": [62, 260, 900, 300], "page_idx": 2},
    {"type": "text", "text": "   ", "bbox": [62, 320, 900, 340], "page_idx": 2},
]


def test_text_and_heading_section():
    blocks = blocks_from_content_list(FIXTURE)
    text_blocks = [b for b in blocks if b.type == "text"]
    # 标题块自身入库且 section 为自身
    assert text_blocks[0].content == "第二章 印象派的色彩"
    assert text_blocks[0].section == "第二章 印象派的色彩"
    # 标题后的正文继承小节
    assert text_blocks[1].section == "第二章 印象派的色彩"


def test_equation_block():
    blocks = blocks_from_content_list(FIXTURE)
    eq = [b for b in blocks if b.type == "equation"]
    assert len(eq) == 1
    assert "h\\nu" in eq[0].content
    assert eq[0].page_idx == 0
    assert eq[0].section == "第二章 印象派的色彩"


def test_table_block_with_caption_and_footnote():
    blocks = blocks_from_content_list(FIXTURE)
    tbl = [b for b in blocks if b.type == "table"]
    assert len(tbl) == 1
    assert "表 2-1 主要画家年表" in tbl[0].content  # caption 在前
    assert "<table>" in tbl[0].content  # HTML 正文整块保留
    assert "注：年份为出生年。" in tbl[0].content  # footnote 收尾


def test_image_caption_block():
    blocks = blocks_from_content_list(FIXTURE)
    img = [b for b in blocks if b.type == "image"]
    assert len(img) == 1
    assert "《睡莲》" in img[0].content
    assert "芝加哥艺术学院" in img[0].content  # footnote 并入 caption
    assert img[0].section == "第二章 印象派的色彩"


def test_image_without_caption_skipped():
    blocks = blocks_from_content_list(FIXTURE)
    # 第 1 页的无图注图片不产 Block
    assert all("p2" not in b.content for b in blocks)
    assert not any(b.type == "image" and b.page_idx == 1 for b in blocks)


def test_page_noise_discarded():
    blocks = blocks_from_content_list(FIXTURE)
    contents = "\n".join(b.content for b in blocks)
    assert "艺术史讲义" not in contents  # header
    assert "第 17 页" not in contents  # footer
    assert not any(b.content == "17" for b in blocks)  # page_number


def test_list_and_code_become_text():
    blocks = blocks_from_content_list(FIXTURE)
    text_blocks = [b for b in blocks if b.type == "text"]
    list_block = [b for b in text_blocks if "色相" in b.content]
    assert len(list_block) == 1 and "饱和度" in list_block[0].content
    code_block = [b for b in text_blocks if "mix(red, blue)" in b.content]
    assert len(code_block) == 1 and "示例" in code_block[0].content
    # list/code 位于 2.1 标题之后，继承新小节
    assert list_block[0].section == "2.1 光的物理"


def test_empty_text_skipped():
    blocks = blocks_from_content_list(FIXTURE)
    assert all(b.content.strip() for b in blocks)


def test_page_filtering():
    blocks = blocks_from_content_list(FIXTURE, page_nos={0})
    assert blocks and all(b.page_idx == 0 for b in blocks)


def test_page_filtering_empty_result():
    blocks = blocks_from_content_list(FIXTURE, page_nos={99})
    assert blocks == []


def test_no_filter_keeps_all_pages():
    blocks = blocks_from_content_list(FIXTURE, page_nos=None)
    assert {b.page_idx for b in blocks} == {0, 1, 2}


def test_bbox_preserved():
    blocks = blocks_from_content_list(FIXTURE)
    assert blocks[0].bbox == (62, 80, 400, 120)


def test_availability_follows_env_token():
    saved = os.environ.get("MINERU_TOKEN")
    try:
        os.environ["MINERU_TOKEN"] = ""
        assert not mineru_available()
        os.environ["MINERU_TOKEN"] = "  "
        assert not mineru_available()
        os.environ["MINERU_TOKEN"] = "dummy-token"
        assert mineru_available()
    finally:
        if saved is None:
            os.environ.pop("MINERU_TOKEN", None)
        else:
            os.environ["MINERU_TOKEN"] = saved


# ══════════════════════════════════════════════════════════════════
# 3. pdf_splitter（src/ingestion/pdf_splitter.py）
# ══════════════════════════════════════════════════════════════════


def _make_pdf(pages: int) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1} content")
    data = doc.tobytes()
    doc.close()
    return data


def test_split_pdf_creates_multiple_parts():
    data = _make_pdf(5)
    parts = split_pdf(data, max_bytes=1200, filename="big.pdf")
    assert len(parts) >= 2
    assert all(name.startswith("big_part") and name.endswith(".pdf") for name, _ in parts)
    total_pages = 0
    for _, part_bytes in parts:
        doc = fitz.open(stream=part_bytes, filetype="pdf")
        total_pages += doc.page_count
        doc.close()
    assert total_pages == 5


def test_split_pdf_small_file_single_part():
    data = _make_pdf(2)
    parts = split_pdf(data, max_bytes=10 * 1024 * 1024, filename="small.pdf")
    assert len(parts) == 1
    assert parts[0][0] == "small_part1.pdf"


def test_split_pdf_single_oversized_page_still_returned():
    data = _make_pdf(1)
    parts = split_pdf(data, max_bytes=100, filename="one.pdf")
    assert len(parts) == 1  # 单页超限：仍作为独立部分返回，不丢页


# ══════════════════════════════════════════════════════════════════
# 4. ocr_route（src/ingestion/pipeline.py 的 _parse_ocr_route）
# ══════════════════════════════════════════════════════════════════


def _ocr_blocks():
    return [
        Block(type="text", content="手稿正文第一段", page_idx=0),
        Block(type="table", content="<table>…</table>", page_idx=0),
        Block(type="equation", content="x^2", page_idx=1),
        Block(type="image", content="page-1.png", page_idx=1),
    ]


def test_ocr_disabled_returns_empty():
    os.environ["MINERU_OCR"] = "0"
    try:
        out = _parse_ocr_route("x.pdf", [0, 1], None)
    finally:
        os.environ.pop("MINERU_OCR", None)
    assert out == []


def test_ocr_unavailable_returns_empty():
    with patch("src.ingestion.pipeline.mineru_available", return_value=False):
        out = _parse_ocr_route("x.pdf", [0], None)
    assert out == []


def test_ocr_parse_failure_falls_back():
    with patch("src.ingestion.pipeline.mineru_available", return_value=True), \
         patch("src.ingestion.pipeline.mineru_parse_pages", side_effect=RuntimeError("boom")):
        out = _parse_ocr_route("x.pdf", [0], None)
    assert out == []


def test_ocr_keeps_only_text_blocks():
    with patch("src.ingestion.pipeline.mineru_available", return_value=True), \
         patch("src.ingestion.pipeline.mineru_parse_pages", return_value=_ocr_blocks()):
        out = _parse_ocr_route("x.pdf", [0, 1], None)
    assert len(out) == 3  # text + table + equation（image 块不进文字层）
    assert all(b.type != "image" for b in out)


def test_ocr_empty_pages_returns_empty():
    with patch("src.ingestion.pipeline.mineru_available", return_value=True) as m:
        out = _parse_ocr_route("x.pdf", [], None)
    m.assert_not_called()
    assert out == []


# ══════════════════════════════════════════════════════════════════
# 5. page_classifier（src/ingestion/page_classifier.py）
# ══════════════════════════════════════════════════════════════════


class _FakeRect:
    width = 100.0
    height = 100.0  # 页面总面积 10000


class _FakePage:
    """模拟 PyMuPDF page 的最小接口。"""

    def __init__(self, text="", image_bboxes=(), fonts=("F1",)):
        self._text = text
        self._bboxes = list(image_bboxes)
        self._fonts = list(fonts)
        self.rect = _FakeRect()

    def get_text(self):
        return self._text

    def get_image_info(self):
        return [{"bbox": b} for b in self._bboxes]

    def get_fonts(self):
        return self._fonts


def _signals(text_len=0, image_ratio=0.0, has_fonts=True,
             formula_symbols=0, text_len_for_density=None):
    return PageSignals(
        page_no=0,
        text_len=text_len,
        image_ratio=image_ratio,
        has_fonts=has_fonts,
        formula_symbols=formula_symbols,
        formula_density=formula_symbols / max(text_len_for_density or text_len, 1),
    )


def test_collect_text_len_stripped():
    page = _FakePage(text="  abc def  \n")
    s = collect_page_signals(page, 0)
    assert s.text_len == len("abc def")


def test_collect_image_ratio_sums_bboxes():
    # 两张 50x100 图 = 10000 面积 = 占满整页
    page = _FakePage(text="x" * 10, image_bboxes=[(0, 0, 50, 100), (50, 0, 100, 100)])
    s = collect_page_signals(page, 0)
    assert abs(s.image_ratio - 1.0) < 1e-6


def test_collect_image_ratio_capped_at_1():
    # 图片 bbox 溢出页面时占比截断到 1
    page = _FakePage(text="x" * 10, image_bboxes=[(0, 0, 200, 200)])
    s = collect_page_signals(page, 0)
    assert s.image_ratio == 1.0


def test_collect_no_fonts_for_scanned_page():
    s = collect_page_signals(_FakePage(text="", fonts=[]), 0)
    assert s.has_fonts is False


def test_collect_formula_symbols():
    page = _FakePage(text="能量公式 E=mc²，其中 ∑ 表示求和，另有 ∫ 积分符号")
    s = collect_page_signals(page, 0)
    assert s.formula_symbols >= 3  # ² ∑ ∫
    assert s.formula_density > 0


def test_route_text_page():
    # 文字多、图少 → 文字路线
    r = classify_page(_signals(text_len=500, image_ratio=0.1))
    assert r.route == "text" and r.force_mineru is False


def test_route_multimodal_low_text():
    r = classify_page(_signals(text_len=20, image_ratio=0.5))
    assert r.route == "multimodal"


def test_route_multimodal_image_dominated():
    r = classify_page(_signals(text_len=300, image_ratio=0.9))
    assert r.route == "multimodal"


def test_route_dual_middle_ground():
    # 文字不少、图也不少（图文并重）→ 双路线
    r = classify_page(_signals(text_len=300, image_ratio=0.5))
    assert r.route == "dual"


def test_route_boundary_text_200():
    # text_len 恰 200 不满足 >200 → 落双路线（图不多也不触发多模态）
    r = classify_page(_signals(text_len=200, image_ratio=0.1))
    assert r.route == "dual"
    r2 = classify_page(_signals(text_len=201, image_ratio=0.1))
    assert r2.route == "text"


def test_route_boundary_image_30_80():
    # 图占比恰 30%：文字多时不满足 <30% → 双路线
    r = classify_page(_signals(text_len=500, image_ratio=0.30))
    assert r.route == "dual"
    # 图占比恰 80%：不满足 >80%，文字 500 也不 <50 → 双路线
    r2 = classify_page(_signals(text_len=500, image_ratio=0.80))
    assert r2.route == "dual"


def test_route_formula_dense_sets_force_mineru():
    s = _signals(text_len=1000, image_ratio=0.1, formula_symbols=20)
    r = classify_page(s)
    assert r.route == "text" and r.force_mineru is True


def test_route_formula_few_symbols_not_dense():
    # 只有零星公式符号（低于最少个数）不算公式密集
    s = _signals(text_len=1000, image_ratio=0.1, formula_symbols=2)
    assert s.formula_dense is False
    assert classify_page(s).force_mineru is False


def test_formula_density_threshold():
    s = PageSignals(
        page_no=0, text_len=1000, image_ratio=0.1, has_fonts=True,
        formula_symbols=10, formula_density=FORMULA_DENSITY_THRESHOLD,
    )
    assert s.formula_dense is True
    s2 = PageSignals(
        page_no=0, text_len=1000, image_ratio=0.1, has_fonts=True,
        formula_symbols=10, formula_density=FORMULA_DENSITY_THRESHOLD / 2,
    )
    assert s2.formula_dense is False


def test_classify_real_pdf_smoke():
    """用 uploads 里的真实 PDF 跑一遍 classify_document（本地文件，秒级）。"""
    root = PROJECT_ROOT
    pdf = root / "data/uploads/test-session/2c8ca435a6c14b23a03512a453944ae5_ren.pdf"
    if not pdf.exists():
        return  # 素材缺失时跳过（CI 容错）
    plan = classify_document(str(pdf))
    assert len(plan.pages) == 3
    # 每页 110 字符、无图：落双路线（50 < 110 < 200）
    assert plan.distribution == {"text": 0, "multimodal": 0, "dual": 3}


# ══════════════════════════════════════════════════════════════════
# 6. page_reader 路径安全（src/tools/page_reader.py）
# ══════════════════════════════════════════════════════════════════

ROOT = PROJECT_ROOT.resolve()
UPLOADS = (ROOT / "data" / "uploads").resolve()


def test_reject_empty_path():
    path, err = _validate_image_path("")
    assert path is None and "为空" in err


def test_reject_path_outside_uploads():
    path, err = _validate_image_path(str(ROOT / "SemArt" / "Images" / "x.jpg"))
    assert path is None and "允许范围" in err


def test_reject_path_traversal():
    path, err = _validate_image_path(str(UPLOADS / ".." / "SemArt" / "x.jpg"))
    assert path is None and "允许范围" in err


def _cleanup(*paths: Path) -> None:
    """容错清理测试临时文件（沙箱 safe-delete 可能拦截删除，残留无害）。"""
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except OSError:
            pass


def test_reject_non_image_suffix():
    p = UPLOADS / "default" / "fake" / "document.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-fake")
    try:
        path, err = _validate_image_path(str(p))
        assert path is None and "不支持的图片类型" in err
    finally:
        _cleanup(p, p.parent)


def test_reject_missing_file():
    path, err = _validate_image_path(str(UPLOADS / "default" / "nope" / "page-0.png"))
    assert path is None and "不存在" in err


def test_accept_valid_page_image(tmp_path=None):
    p = UPLOADS / "default" / "testdoc" / "pages" / "page-0.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    # 最小合法 PNG（1x1）
    p.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d4944415478da63fcffff3f030005fe02fea72d814d0000000049454e44ae426082"
        )
    )
    try:
        path, err = _validate_image_path(str(p))
        assert err is None and path == p
    finally:
        _cleanup(p, p.parent, p.parent.parent)


def test_impl_returns_error_for_bad_path():
    out = read_page_image_impl("")
    assert out["success"] is False and "error" in out


def test_impl_error_for_outside_path():
    out = read_page_image_impl(str(ROOT / "api.py"))
    assert out["success"] is False


# ══════════════════════════════════════════════════════════════════
# 7. page_reader 按文档名+页码定位整页图
# ══════════════════════════════════════════════════════════════════


def _fake_docs(doc_name="莫奈手稿", doc_id="abc123", kb_id="default"):
    return [{"doc_name": doc_name, "doc_id": doc_id, "kb_id": kb_id, "pages": 16}]


def _setup_uploads():
    tmp = Path(tempfile.mkdtemp())
    pages = tmp / "default" / "abc123" / "pages"
    pages.mkdir(parents=True)
    (pages / "page-0.png").write_bytes(b"fake-png")
    (pages / "page-1.png").write_bytes(b"fake-png")
    return tmp


def test_resolve_page_path_by_doc_name_and_page():
    tmp = _setup_uploads()
    with patch.object(pr, "_UPLOADS_ROOT", tmp.resolve()), \
         patch("src.data.documents_store.list_documents", return_value=_fake_docs()):
        path, err = pr._resolve_page_path("莫奈手稿", 1)
    assert err is None
    assert path is not None and path.name == "page-0.png" and path.exists()


def test_resolve_page_missing_file_returns_error():
    tmp = _setup_uploads()
    with patch.object(pr, "_UPLOADS_ROOT", tmp.resolve()), \
         patch("src.data.documents_store.list_documents", return_value=_fake_docs()):
        path, err = pr._resolve_page_path("莫奈手稿", 99)
    assert err is not None and "不存在" in err


def test_resolve_unknown_doc_returns_error():
    tmp = _setup_uploads()
    with patch.object(pr, "_UPLOADS_ROOT", tmp.resolve()), \
         patch("src.data.documents_store.list_documents", return_value=_fake_docs()):
        path, err = pr._resolve_page_path("不存在的文档", 1)
    assert err is not None and "未找到" in err


def test_read_page_image_impl_requires_locator():
    out = pr.read_page_image_impl()
    assert out["success"] is False and "doc_name 与 page" in out["error"]


# ══════════════════════════════════════════════════════════════════
# 8. harvest_wikidata（scripts/01_harvest_wikidata.py）
# ══════════════════════════════════════════════════════════════════


def test_qid_from_uri():
    assert hw._qid_from_uri("http://www.wikidata.org/entity/Q3305213") == "Q3305213"
    assert hw._qid_from_uri("") == ""


def test_year_from_iso():
    assert hw._year_from_iso("1889-01-01T00:00:00Z") == 1889
    assert hw._year_from_iso("c. 1889") is None
    assert hw._year_from_iso("") is None


def test_clean_label():
    assert hw._clean_label("  A &amp; B  ") == "A & B"


def _binding(**kw):
    base = {
        "artist": {"value": "http://www.wikidata.org/entity/Q5582"},
        "artistLabel": {"value": "Vincent van Gogh"},
        "birth": {"value": "1853-03-30T00:00:00Z"},
        "death": {"value": "1890-07-29T00:00:00Z"},
        "natLabel": {"value": "Netherlands"},
        "movLabel": {"value": "Post-Impressionism"},
        "img": {"value": "http://commons.wikimedia.org/wiki/Special:FilePath/x.jpg"},
    }
    base.update(kw)
    return base


def test_artist_binding_to_row():
    row = hw.artist_binding_to_row(_binding())
    assert row["artist_qid"] == "Q5582"
    assert row["name"] == "Vincent van Gogh"
    assert row["birth"] == 1853
    assert row["death"] == 1890
    assert row["nationality"] == "Netherlands"
    assert row["movement"] == "Post-Impressionism"
    assert row["source_api"] == "wikidata"


def test_artwork_binding_to_row():
    b = _binding(
        w={"value": "http://www.wikidata.org/entity/Q82104"},
        wLabel={"value": "The Bedroom"},
        collLabel={"value": "Art Institute of Chicago"},
        inception={"value": "1889-01-01T00:00:00Z"},
        materialLabel={"value": "oil paint"},
        genreLabel={"value": "interior view"},
        locLabel={"value": "Art Institute of Chicago"},
        seriesLabel={"value": ""},
        desc={"value": "Bedroom in the Yellow House"},
    )
    row = hw.artwork_binding_to_row(b, "Q239303")
    assert row["artwork_id"] == "Q82104"
    assert row["artist_qid"] == "Q5582"
    assert row["collection_qid"] == "Q239303"
    assert row["year"] == 1889
    assert row["year_bucket"] == "1851-1900"
    assert row["material"] == "oil paint"
    assert row["description"] == "Bedroom in the Yellow House"
    assert row["dedup_key"] == "vincent van gogh|the bedroom|1889"


# ══════════════════════════════════════════════════════════════════
# 9. build_extended_dataset（scripts/02_build_extended_dataset.py）
# ══════════════════════════════════════════════════════════════════


def test_clean_desc_strips_html_and_entities():
    raw = "<p>The bedroom &amp; studio</p>\n\n<p>Second version&nbsp;of the scene.</p>"
    out = bed._clean_desc(raw)
    assert "The bedroom & studio" in out
    assert "<" not in out and "&nbsp;" not in out
    assert "\n" not in out


def test_clean_desc_empty():
    assert bed._clean_desc(None) == ""
    assert bed._clean_desc("   ") == ""


def test_parse_dimensions_cm_main():
    raw = "73.6 × 92.3 cm (29 × 36 5/8 in.); Framed: 88.9 × 108 × 8.9 cm (35 × 42 1/2 × 3 1/2 in.)"
    assert bed._parse_dimensions_cm(raw) == (73.6, 92.3)


def test_parse_dimensions_inches_converted():
    assert bed._parse_dimensions_cm("24 × 18 in.") == (60.96, 45.72)


def test_parse_dimensions_none_or_unparsable():
    assert bed._parse_dimensions_cm(None) == (None, None)
    assert bed._parse_dimensions_cm("no dimensions here") == (None, None)


def test_is_fetchable():
    base = {
        "artwork_type_title": "Painting",
        "image_id": "abc",
        "description": "A description.",
    }
    assert bed._is_fetchable(base)
    assert not bed._is_fetchable({**base, "artwork_type_title": "Sculpture"})
    assert not bed._is_fetchable({**base, "image_id": None})
    # 空描述也入库（只进表不进向量），由 index 步骤分流
    assert bed._is_fetchable({**base, "description": ""})


def test_aic_record_to_row_full_mapping():
    rec = {
        "id": 28560,
        "title": "The Bedroom",
        "artist_title": "Vincent van Gogh",
        "date_start": 1889,
        "date_end": 1889,
        "date_display": "1889",
        "medium_display": "Oil on canvas",
        "classification_title": "oil on canvas",
        "style_title": "Post-Impressionism",
        "description": "<p>Perhaps the most famous depiction of a bedroom.</p>",
        "image_id": "6644829f-f292-c5c4-a73c-0356a6fdbf0d",
        "dimensions": "73.6 × 92.3 cm (29 × 36 5/8 in.)",
        "department_title": "Painting and Sculpture of Europe",
        "artwork_type_title": "Painting",
        "copyright_notice": None,
    }
    row = bed.aic_record_to_row(rec)
    assert row["object_id"] == "aic:28560"
    assert row["artist"] == "Vincent van Gogh"
    assert row["year"] == 1889
    assert row["year_bucket"] == "1851-1900"
    assert row["school"] == "Post-Impressionism"
    assert "<p>" not in row["description"]
    assert row["image_url"].startswith("https://www.artic.edu/iiif/2/6644829f")
    assert row["license"] == "CC0/Public domain (AIC)"
    assert row["width_cm"] == 73.6 and row["height_cm"] == 92.3
    assert "Art Institute of Chicago" in row["location"]
    assert row["source_api"] == "aic"
    assert row["dedup_key"] == "vincent van gogh|the bedroom|1889"


# ══════════════════════════════════════════════════════════════════
# 10. normalize_core（scripts/03_normalize_core.py）
# ══════════════════════════════════════════════════════════════════


def test_normalize_artist_name():
    assert nc._normalize_artist_name("GOGH, Vincent van") == "Vincent van GOGH"
    assert nc._normalize_artist_name("Vincent van Gogh") == "Vincent van Gogh"
    assert nc._normalize_artist_name("") == ""


def test_year_from_text():
    assert nc._year_from_text("1526 and after 1528") == 1526
    assert nc._year_from_text("1770-75") == 1770
    assert nc._year_from_text("") is None


def test_semart_row_to_core():
    row = nc.semart_row_to_core({
        "IMAGE_FILE": "18759-guard301.jpg",
        "TITLE": "Landscape with a Fisherman's Tent",
        "AUTHOR": "GUARDI, Francesco",
        "DATE": "1770-75",
        "TIMEFRAME": "1751-1800",
        "TECHNIQUE": "Oil on canvas, 49 x 77 cm",
        "TYPE": "landscape",
        "SCHOOL": "Italian",
        "DESCRIPTION": "A poetic scene with fishing boats.",
    })
    assert row["artwork_id"].startswith("semart:")
    assert row["artist_name"] == "Francesco GUARDI"       # 倒序名已归一
    assert row["year"] == 1770
    assert row["year_bucket"] == "1751-1800"
    assert row["genre"] == "landscape"
    assert row["school"] == "Italian"
    assert row["material"].startswith("Oil on canvas")
    assert row["source_api"] == "semart"
    # 归一化后的 dedup_key 与 Wikidata 显示序（"Francesco Guardi"）一致
    assert row["dedup_key"] == cutils._dedup_key("Francesco Guardi", row["title"], 1770)


def test_merge_semart_with_wikidata():
    semart = {
        "artwork_id": "semart:abc", "title": "The Bedroom", "artist_qid": "",
        "artist_name": "Vincent van GOGH", "year": 1889, "source_api": "semart",
        "description": "A" * 300, "genre": "interior", "school": "Dutch",
        "movement": "", "material": "", "image_url": "img.jpg", "collection_name": "",
        "location": "", "inception": "1889", "series": "", "license": "",
        "dimensions_raw": "", "width_cm": "", "height_cm": "",
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    wd = {
        "artwork_id": "Q82104", "title": "The Bedroom", "artist_qid": "Q5582",
        "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "wikidata",
        "description": "short", "movement": "Post-Impressionism", "genre": "",
        "material": "oil paint", "image_url": "", "collection_name": "AIC",
        "location": "Chicago", "inception": "1889", "series": "", "license": "",
        "dimensions_raw": "", "width_cm": "", "height_cm": "",
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    merged = nc.merge_rows([semart, wd])[0]
    assert merged["artist_qid"] == "Q5582"
    assert len(merged["description"]) == 300          # SemArt 长描述胜出
    assert merged["movement"] == "Post-Impressionism"
    assert merged["school"] == "Dutch"
    assert merged["genre"] == "interior"
    assert merged["source_api"] == "semart;wikidata"


def test_aic_row_to_core():
    row = nc.aic_row_to_core({
        "object_id": "aic:28560",
        "title": "The Bedroom",
        "artist": "Vincent van Gogh",
        "year": 1889,
        "year_display": "1889",
        "year_bucket": "1851-1900",
        "medium": "Oil on canvas",
        "school": "Post-Impressionism",
        "description": "Famous bedroom.",
        "image_url": "http://x/img.jpg",
        "license": "CC0",
        "location": "Europe · Art Institute of Chicago",
        "dimensions_raw": "73.6 × 92.3 cm",
        "width_cm": 73.6,
        "height_cm": 92.3,
    })
    assert row["artwork_id"] == "aic:28560"
    assert row["artist_qid"] == ""
    assert row["collection_name"] == "Art Institute of Chicago"
    assert row["movement"] == "Post-Impressionism"
    assert row["source_api"] == "aic"
    assert row["year"] == 1889


def test_merge_prefers_qid_and_longer_description():
    wd = {
        "artwork_id": "Q82104", "title": "The Bedroom", "artist_qid": "Q5582",
        "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "wikidata",
        "description": "short", "movement": "", "genre": "", "material": "",
        "image_url": "", "collection_name": "", "location": "", "inception": "1889",
        "series": "", "license": "", "dimensions_raw": "", "width_cm": "", "height_cm": "",
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    aic = {
        "artwork_id": "aic:28560", "title": "The Bedroom", "artist_qid": "",
        "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "aic",
        "description": "A much longer curatorial description of the bedroom.", "movement": "Post-Impressionism",
        "genre": "", "material": "oil paint", "image_url": "http://x.jpg",
        "collection_name": "Art Institute of Chicago", "location": "Chicago", "inception": "1889",
        "series": "", "license": "CC0", "dimensions_raw": "73.6 × 92.3 cm",
        "width_cm": 73.6, "height_cm": 92.3,
        "dedup_key": "vincent van gogh|the bedroom|1889",
    }
    merged = nc.merge_rows([wd, aic])[0]
    assert merged["artist_qid"] == "Q5582"          # 保留 Wikidata QID
    assert merged["description"] == aic["description"]  # 长描述胜出
    assert merged["movement"] == "Post-Impressionism"
    assert merged["material"] == "oil paint"
    assert merged["image_url"] == "http://x.jpg"
    assert merged["source_api"] == "wikidata;aic"


def test_merge_distinct_rows_keep_both():
    a = {"artwork_id": "Q1", "title": "A", "artist_qid": "Q9", "artist_name": "X",
         "year": 1800, "description": "d1", "source_api": "wikidata", "dedup_key": "x|a|1800"}
    b = {"artwork_id": "Q2", "title": "B", "artist_qid": "Q8", "artist_name": "Y",
         "year": 1850, "description": "d2", "source_api": "wikidata", "dedup_key": "y|b|1850"}
    assert len(nc.merge_rows([a, b])) == 2


def test_merge_keeps_distinct_qids_for_same_key():
    """同名同题同年但 QID 不同 → 两幅不同作品，不合并（误合并审计修复）。"""
    wd_a = {"artwork_id": "Q18011394", "title": "Portrait of a Man", "artist_qid": "Q41254",
            "artist_name": "Frans Hals", "year": 1650, "source_api": "wikidata",
            "description": "A", "dedup_key": "frans hals|portrait of a man|1650"}
    wd_b = {"artwork_id": "Q18025591", "title": "Portrait of a Man", "artist_qid": "Q41254",
            "artist_name": "Frans Hals", "year": 1650, "source_api": "wikidata",
            "description": "B", "dedup_key": "frans hals|portrait of a man|1650"}
    merged = nc.merge_rows([wd_a, wd_b])
    assert len(merged) == 2
    assert {r["artwork_id"] for r in merged} == {"Q18011394", "Q18025591"}


def test_merge_semart_with_wikidata_still_merges():
    """无 QID 的 semart 行与有 QID 的 Wikidata 行：同作品，仍合并。"""
    semart = {"artwork_id": "semart:abc", "title": "The Bedroom", "artist_qid": "",
              "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "semart",
              "description": "A" * 100, "dedup_key": "vincent van gogh|the bedroom|1889"}
    wd = {"artwork_id": "Q82104", "title": "The Bedroom", "artist_qid": "Q5582",
          "artist_name": "Vincent van Gogh", "year": 1889, "source_api": "wikidata",
          "description": "short", "dedup_key": "vincent van gogh|the bedroom|1889"}
    merged = nc.merge_rows([semart, wd])
    assert len(merged) == 1
    assert merged[0]["artwork_id"] == "Q82104"


def test_join_movement_fills_from_artist_qid():
    artworks = [
        {"artist_qid": "Q5582", "movement": "", "title": "t"},
        {"artist_qid": "Q999", "movement": "Baroque", "title": "u"},
        {"artist_qid": "", "movement": "", "title": "v"},
    ]
    nc.join_movement(artworks, {"Q5582": "Post-Impressionism"})
    assert artworks[0]["movement"] == "Post-Impressionism"
    assert artworks[1]["movement"] == "Baroque"      # 已有值不覆盖
    assert artworks[2]["movement"] == ""             # 无 QID 跳过


# ══════════════════════════════════════════════════════════════════
# 11. index_core（scripts/06_index_core.py）
# ══════════════════════════════════════════════════════════════════


def _index_df():
    return pd.DataFrame(
        {"artwork_id": ["Q1", "Q2", "Q3"], "description": ["a", "b", "c"]}
    )


def test_resume_skips_existing_ids():
    out = _select_rows_to_index(_index_df(), {"Q1", "Q3"}, force=False)
    assert out["artwork_id"].tolist() == ["Q2"]


def test_resume_no_existing_keeps_all():
    out = _select_rows_to_index(_index_df(), set(), force=False)
    assert len(out) == 3


def test_force_keeps_all_even_when_existing():
    out = _select_rows_to_index(_index_df(), {"Q1", "Q2", "Q3"}, force=True)
    assert len(out) == 3


# ══════════════════════════════════════════════════════════════════
# 12. assemble_descriptions（scripts/04_assemble_descriptions.py）
# ══════════════════════════════════════════════════════════════════


def test_split_batches():
    assert ad.split_batches(list("abcdef"), 2) == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert ad.split_batches([], 5) == []


def test_parse_sitelinks():
    payload = {
        "entities": {
            "Q82104": {"sitelinks": {"enwiki": {"title": "The Bedroom (Van Gogh painting)"}}},
            "Q99999": {"sitelinks": {"frwiki": {"title": "X"}}},  # 无 enwiki → 跳过
            "Q88888": {},
        }
    }
    out = ad.parse_sitelinks(payload)
    assert out == {"Q82104": "The Bedroom (Van Gogh painting)"}


def test_parse_extracts():
    payload = {
        "query": {"pages": [
            {"title": "The Bedroom (Van Gogh painting)", "extract": "Painting by Vincent van Gogh."},
            {"title": "No Extract Page"},
            {"title": "Empty", "extract": "   "},
        ]}
    }
    out = ad.parse_extracts(payload)
    assert out == {"The Bedroom (Van Gogh painting)": "Painting by Vincent van Gogh."}


def test_qids_needing_description():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "core.csv"
        pd.DataFrame([
            {"artwork_id": "Q1", "description": "has desc", "title": "a"},
            {"artwork_id": "Q2", "description": "   ", "title": "b"},
            {"artwork_id": "semart:abc", "description": "", "title": "c"},  # 非 QID 跳过
        ]).to_csv(path, index=False, encoding="utf-8-sig")
        assert ad.qids_needing_description(path) == ["Q2"]
        assert ad.qids_needing_description(path, limit=1) == ["Q2"]


def test_apply_descriptions_only_fills_empty():
    df = pd.DataFrame([
        {"artwork_id": "Q1", "description": "has desc", "title": "a"},
        {"artwork_id": "Q2", "description": "", "title": "b"},
        {"artwork_id": "semart:x", "description": "", "title": "c"},
    ])
    filled, total = ad.apply_descriptions(df, {"Q2": "Intro for b.", "Q1": "should not override"})
    assert filled == 1
    assert total == 3
    assert df.loc[1, "description"] == "Intro for b."
    assert df.loc[0, "description"] == "has desc"      # 已有描述不覆盖
    assert df.loc[2, "description"] == ""               # 非 QID 不处理


# ══════════════════════════════════════════════════════════════════
# 13. audit_core（scripts/05_audit_core.py）
# ══════════════════════════════════════════════════════════════════


def _audit_df():
    return pd.DataFrame([
        {"artwork_id": "Q1", "title": "A", "description": "desc", "image_url": "http://x",
         "year": 1800, "movement": "Neoclassicism", "school": "", "artist_qid": "Q9",
         "source_api": "wikidata", "dedup_key": "a|b|1800"},
        {"artwork_id": "Q2", "title": "B", "description": "", "image_url": "",
         "year": "", "movement": "", "school": "Dutch", "artist_qid": "",
         "source_api": "semart", "dedup_key": "c|d|"},
        {"artwork_id": "Q3", "title": "C", "description": "x", "image_url": "http://y",
         "year": 1950, "movement": "Abstract", "school": "", "artist_qid": "Q8",
         "source_api": "wikidata;aic", "dedup_key": "e|f|1950"},
    ])


def test_compute_stats():
    s = ac.compute_stats(_audit_df())
    assert s["total"] == 3
    assert s["described"] == 2
    assert s["image"] == 2
    assert s["year"] == 2
    assert s["year_min"] == 1800 and s["year_max"] == 1950
    assert s["movement"] == 2
    assert s["school"] == 1
    assert s["artist_qid"] == 2
    assert s["dup_artwork_ids"] == 0
    assert s["dedup_collision_groups"] == 0
    assert s["sources"] == {"wikidata": 2, "semart": 1, "aic": 1}


def test_compute_stats_dedup_conflict():
    df = _audit_df()
    df.loc[2, "dedup_key"] = "a|b|1800"  # 与 Q1 同 dedup_key 但不同 artwork_id
    s = ac.compute_stats(df)
    assert s["dedup_collision_groups"] == 1


def test_render_report_contains_key_metrics():
    report = ac.render_report(ac.compute_stats(_audit_df()))
    assert "总作品数" in report and "有描述" in report and "来源分布" in report


def test_collision_groups_and_summary():
    df = _audit_df()
    df.loc[2, "dedup_key"] = "a|b|1800"          # 与 Q1 冲突
    df.loc[2, "image_url"] = df.loc[0, "image_url"]  # 同图 → 疑似重复记录
    groups = ac.collision_groups(df)
    assert len(groups) == 2
    s = ac.collision_summary(groups)
    assert s["groups"] == 1 and s["same_image_groups"] == 1


# ══════════════════════════════════════════════════════════════════
# 14. core_utils（脚本共享工具，原三份复制合并为一）
# ══════════════════════════════════════════════════════════════════


def test_year_bucket_shared():
    assert cutils._year_bucket(1889) == "1851-1900"
    assert cutils._year_bucket(1900) == "1851-1900"
    assert cutils._year_bucket(1901) == "1901-1950"
    assert cutils._year_bucket(None) == ""


def test_dedup_key_shared():
    a = cutils._dedup_key("Vincent van Gogh", "The Bedroom", 1889)
    b = cutils._dedup_key("vincent  van gogh", "the  bedroom", 1889)
    c = cutils._dedup_key("Vincent van Gogh", "The Bedroom", 1890)
    assert a == b
    assert a == "vincent van gogh|the bedroom|1889"
    assert a != c


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] ingestion 集群全部 {len(fns)} 个单测通过")
