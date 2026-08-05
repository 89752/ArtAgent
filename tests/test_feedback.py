"""反馈闭环存储测试。"""

import tempfile
from pathlib import Path

import pytest

from src.memory import feedback as fb


@pytest.fixture(autouse=True)
def _isolated_db():
    tmp = Path(tempfile.mkdtemp(prefix="artagent_fb_test_"))
    fb._reset_for_tests(tmp / "feedback.db")
    yield
    fb._reset_for_tests()


def test_add_and_list_feedback():
    fb.add_feedback("s1", 1, reason="", comment="很棒")
    fb.add_feedback("s1", -1, reason="引用不充分", comment="")
    items, total = fb.list_feedback()
    assert total == 2
    assert items[0]["rating"] == -1        # 倒序：最新在前
    assert items[0]["reason"] == "引用不充分"
    assert items[1]["rating"] == 1
    assert items[1]["comment"] == "很棒"
    assert fb.count_feedback(1) == 1
    assert fb.count_feedback(-1) == 1


def test_invalid_rating_rejected():
    with pytest.raises(ValueError):
        fb.add_feedback("s1", 0)


def test_export_feedback_jsonl(tmp_path):
    fb.add_feedback("s1", 1)
    fb.add_feedback("s2", -1, reason="不准确")
    out = tmp_path / "feedback.jsonl"
    n = fb.export_feedback(out)
    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"rating": -1' in lines[0]
