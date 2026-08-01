# tests/test_mineru_parser.py
"""
MinerU 解析器（src/ingestion/mineru_parser.py）纯单测：
用固化的 content_list.json 条目流验证块类型映射/小节推进/页过滤，
不联网、不调 LLM、不依赖真实 PDF，秒级完成。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.mineru_parser import blocks_from_content_list, mineru_available

# ── 固化 fixture：模拟 MinerU v4 zip 内 *_content_list.json 的条目流 ──
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


# ── 类型映射 ─────────────────────────────────────────────────────
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


# ── 页过滤与 bbox/页码 ────────────────────────────────────────────
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


# ── 可用性探测 ────────────────────────────────────────────────────
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 mineru_parser 全部 {len(fns)} 个单测通过！")
