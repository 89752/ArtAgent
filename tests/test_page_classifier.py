# tests/test_page_classifier.py
"""
页级路由（src/ingestion/page_classifier.py）纯单测：
fake 页面对象注入信号，不打开真实 PDF、不调 LLM、不联网，秒级完成。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.page_classifier import (
    FORMULA_DENSITY_THRESHOLD,
    PageSignals,
    classify_page,
    collect_page_signals,
)


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


# ── 信号采集 ─────────────────────────────────────────────────────
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


# ── 路由判定 ─────────────────────────────────────────────────────
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


# ── 真文档 smoke：现成素材各页路由合理 ────────────────────────────
def test_classify_real_pdf_smoke():
    """用 uploads 里的真实 PDF 跑一遍 classify_document（本地文件，秒级）。"""
    from src.ingestion.page_classifier import classify_document

    root = Path(__file__).parent.parent
    pdf = root / "data/uploads/test-session/2c8ca435a6c14b23a03512a453944ae5_ren.pdf"
    if not pdf.exists():
        return  # 素材缺失时跳过（CI 容错）
    plan = classify_document(str(pdf))
    assert len(plan.pages) == 3
    # 每页 110 字符、无图：落双路线（50 < 110 < 200）
    assert plan.distribution == {"text": 0, "multimodal": 0, "dual": 3}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 page_classifier 全部 {len(fns)} 个单测通过！")
