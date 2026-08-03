# tests/test_channel_weights.py
"""
加权 RRF 通道权重纯单测：机制正确性 + 有效性场景（噪声通道降权）。
不加载数据、不联网、不调 LLM。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import _channel_weight, _rrf_fuse


def _hit(content, source="semart", **meta) -> RetrievalResult:
    return RetrievalResult(content=content, source=source, score=0.9, metadata=meta)


def test_default_weights_equal_for_main_sources():
    # 主力通道保持 1.0（与旧等权 RRF 一致）
    assert _channel_weight("semart") == 1.0
    assert _channel_weight("core") == 1.0
    assert _channel_weight("user_table") == 1.0
    assert _channel_weight("user_pdf_text") == 1.0


def test_noise_channel_down_weighted():
    assert _channel_weight("user_pdf_image") == 0.5
    assert _channel_weight("met_museum") == 0.5
    assert _channel_weight("unknown_future_source") == 1.0  # 未知源不意外降权


def test_env_override_wins():
    os.environ["CHANNEL_WEIGHT_MET_MUSEUM"] = "0.3"
    try:
        assert _channel_weight("met_museum") == 0.3
    finally:
        os.environ.pop("CHANNEL_WEIGHT_MET_MUSEUM", None)
    assert _channel_weight("met_museum") == 0.5


def test_equal_weight_matches_old_rrf_order():
    # 全 1.0 时：与旧等权 RRF 逐位一致（a0 与 b0 同 rank 0，稳定序 a0 在前）
    a = [_hit("a0"), _hit("a1")]
    b = [_hit("b0", source="user_pdf_text")]
    fused = _rrf_fuse([a, b])
    assert [h.content for h in fused] == ["a0", "b0", "a1"]


def test_single_source_order_preserved_regardless_of_weight():
    hits = [_hit(f"x{i}") for i in range(3)]
    assert [h.content for h in _rrf_fuse([hits])] == ["x0", "x1", "x2"]


def test_lower_weight_source_loses_tie():
    # 同 rank：权重 1.0 排在权重 0.5 之前
    a = [_hit("semart_rank0", source="semart")]
    b = [_hit("image_rank0", source="user_pdf_image")]
    fused = _rrf_fuse([a, b])
    assert fused[0].content == "semart_rank0"


def test_weight_flips_outcome_noise_channel():
    """有效性场景：噪声通道占满前排时，降权能把相关结果拉回前排。"""
    # 相关结果在 semart（w=1.0）第 5 位；噪声通道 user_pdf_image（w=0.5）占 rank 0-3
    relevant = [_hit("relevant", source="semart")]
    noise = [_hit(f"noise{i}", source="user_pdf_image") for i in range(4)]
    # 相关在 semart 源内排第 5（前面垫 4 个无关 semart 结果）
    semart_full = [_hit(f"unrelated{i}") for i in range(4)] + relevant
    fused = _rrf_fuse([semart_full, noise])
    # 加权后：相关 rank4 → 1/65 ≈ 0.0154 > 噪声 rank0 → 0.5/61 ≈ 0.0082
    assert fused.index(relevant[0]) < fused.index(noise[0])
    # 对照：等权（权重全 1.0）时噪声 rank0 = 1/61 ≈ 0.0164 > 相关 1/65 —— 噪声胜出
    os.environ["CHANNEL_WEIGHT_USER_PDF_IMAGE"] = "1.0"
    try:
        fused_equal = _rrf_fuse([semart_full, noise])
        assert fused_equal.index(relevant[0]) > fused_equal.index(noise[0])
    finally:
        os.environ.pop("CHANNEL_WEIGHT_USER_PDF_IMAGE", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] channel_weights 全部 {len(fns)} 个单测通过！")
