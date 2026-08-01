"""
页级自适应路由（Stage 3）：零模型调用的确定性规则。

为什么页级而不是文档级：图文类 PDF（尤其画册/图录）经常页内混排——
正文页排版规整、图版页几乎一整张图，按整份文档二选一必错杀另一半内容。

流程：PyMuPDF 逐页信号采集（几乎无成本）→ 确定性规则判定每页走
文字路线 / 多模态路线 / 双路线（都入库，共享 page_id，检索端去重）。

信号与阈值均为起始值（实施方案 §4.1），需真实 PDF 人工核对后微调；
全部做成模块常量，调参只改这里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.utils.logging_config import get_logger

logger = get_logger("ingestion.classifier")

# ── 路由阈值（起始值，真实 PDF 核对后微调） ─────────────────────────
TEXT_ROUTE_MIN_CHARS = 200  # 可提取字符数 > 此值才可能是文字路线
TEXT_ROUTE_MAX_IMG_RATIO = 0.30  # 图片面积占比 < 此值才走纯文字路线
MM_ROUTE_MAX_CHARS = 50  # 可提取字符数 < 此值 → 多模态候选
MM_ROUTE_MIN_IMG_RATIO = 0.80  # 图片面积占比 > 此值 → 多模态候选

# 公式密度：数学符号数 / 字符数超过此值视为公式密集页
FORMULA_SYMBOLS = "∫∑∏√±≤≥≠≈∂∇∈∀∃λμπσΩ∞∝⊗⊕⟨⟩"
_FORMULA_CHARS = set(FORMULA_SYMBOLS) | set("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ⁺⁻⁼₀₁₂₃₄₅₆₇₈₉₊₋₌")
FORMULA_MIN_SYMBOLS = 5  # 至少这么多个公式符号才参与密度判定（防误报）
FORMULA_DENSITY_THRESHOLD = 0.01

Route = Literal["text", "multimodal", "dual"]


@dataclass
class PageSignals:
    """单页采集到的路由信号。"""

    page_no: int  # 0 基页码
    text_len: int = 0  # 可提取文字字符数（核心信号）
    image_ratio: float = 0.0  # 图片 bbox 面积 / 页面面积
    has_fonts: bool = True  # 是否嵌入字体（扫描页通常没有）
    formula_symbols: int = 0  # 公式符号个数
    formula_density: float = 0.0

    @property
    def formula_dense(self) -> bool:
        """公式密集页：符号够多且密度够高（文字路线内强制 MinerU 的依据）。"""
        return (
            self.formula_symbols >= FORMULA_MIN_SYMBOLS
            and self.formula_density >= FORMULA_DENSITY_THRESHOLD
        )


@dataclass
class PageRoute:
    """单页路由结论。"""

    page_no: int
    route: Route
    force_mineru: bool  # 公式密集：文字路线内不得降级 pdfplumber
    signals: PageSignals


@dataclass
class DocRoutePlan:
    """整份文档的路由计划 + 文档级先验。"""

    producer: str = ""  # 文档级先验：Office 系 vs 扫描仪系
    pages: list[PageRoute] = field(default_factory=list)

    @property
    def distribution(self) -> dict[str, int]:
        """路由分布统计（诊断"为什么检索不到"的第一手信息）。"""
        dist = {"text": 0, "multimodal": 0, "dual": 0}
        for p in self.pages:
            dist[p.route] += 1
        return dist


# ------------------------------------------------------------------ #
# 信号采集（fitz 页面对象 duck-typing，纯单测用 fake page 注入）        #
# ------------------------------------------------------------------ #


def collect_page_signals(page, page_no: int) -> PageSignals:
    """从单个 PyMuPDF page 采集信号（不做任何内容提取，几乎无成本）。"""
    text = page.get_text() or ""
    text_len = len(text.strip())

    page_area = 0.0
    try:
        rect = page.rect
        page_area = max(rect.width * rect.height, 1.0)
    except Exception:
        page_area = 1.0

    image_area = 0.0
    for info in page.get_image_info() or []:
        try:
            bbox = info["bbox"]
            w = max(0.0, bbox[2] - bbox[0])
            h = max(0.0, bbox[3] - bbox[1])
            image_area += w * h
        except (KeyError, IndexError, TypeError):
            continue
    image_ratio = min(image_area / page_area, 1.0)

    has_fonts = bool(page.get_fonts())

    formula_symbols = sum(1 for ch in text if ch in _FORMULA_CHARS)
    formula_density = formula_symbols / max(text_len, 1)

    return PageSignals(
        page_no=page_no,
        text_len=text_len,
        image_ratio=image_ratio,
        has_fonts=has_fonts,
        formula_symbols=formula_symbols,
        formula_density=formula_density,
    )


# ------------------------------------------------------------------ #
# 路由判定（纯函数，零模型调用）                                        #
# ------------------------------------------------------------------ #


def classify_page(signals: PageSignals) -> PageRoute:
    """
    判定单页路由：

      text_len > 200 且 image_ratio < 30%  → 文字路线
      text_len < 50  或 image_ratio > 80%  → 多模态路线
      否则（图文并重的中间地带）            → 双路线（不纠结阈值精度）

    公式密集页标注 force_mineru：文字路线内不得降级 pdfplumber；
    MinerU 不可用时宁可退到多模态路线整页转图（pipeline 负责兑现）。
    """
    force_mineru = signals.formula_dense

    if signals.text_len > TEXT_ROUTE_MIN_CHARS and signals.image_ratio < TEXT_ROUTE_MAX_IMG_RATIO:
        route: Route = "text"
    elif signals.text_len < MM_ROUTE_MAX_CHARS or signals.image_ratio > MM_ROUTE_MIN_IMG_RATIO:
        route = "multimodal"
    else:
        route = "dual"

    return PageRoute(
        page_no=signals.page_no,
        route=route,
        force_mineru=force_mineru,
        signals=signals,
    )


def classify_document(pdf_path: str) -> DocRoutePlan:
    """遍历整份 PDF，产出页级路由计划（含文档级 producer 先验）。"""
    import fitz  # PyMuPDF

    plan = DocRoutePlan()
    with fitz.open(pdf_path) as doc:
        plan.producer = str((doc.metadata or {}).get("producer") or "")
        for page_no, page in enumerate(doc):
            signals = collect_page_signals(page, page_no)
            plan.pages.append(classify_page(signals))
    logger.info(
        "[classify] %s → %s (producer=%s)",
        pdf_path, plan.distribution, plan.producer[:40],
    )
    return plan
