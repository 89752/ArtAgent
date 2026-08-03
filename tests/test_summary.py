"""会话滚动摘要单测（临时 SQLite 库 + mock LLM，不落真实数据）。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

import src.memory.summary as summary_mod


def _tmp_db():
    tmp = Path(tempfile.mkdtemp()) / "conversations.db"
    summary_mod._DB_PATH = tmp
    summary_mod._conn = None
    return tmp


def _messages(n_turns: int) -> list:
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"第{i}轮问题"))
        msgs.append(AIMessage(content=f"第{i}轮回答"))
    return msgs


def test_load_save_roundtrip():
    _tmp_db()
    summary_mod._save_summary("c1", "u1", "摘要内容", 10)
    assert summary_mod.load_summary("c1") == "摘要内容"
    assert summary_mod.load_summary("missing") == ""


def test_maybe_summarize_below_trigger_skips_llm():
    _tmp_db()
    with patch.object(summary_mod, "_summarize") as mock:
        out = summary_mod.maybe_summarize(_messages(5), "c1", "u1", llm=lambda p: "x")
    mock.assert_not_called()
    assert out == ""


def test_maybe_summarize_incremental_and_stored():
    _tmp_db()
    fake_llm = lambda prompt: "压缩后的摘要"
    out = summary_mod.maybe_summarize(_messages(10), "c1", "u1", llm=fake_llm)
    assert out == "压缩后的摘要"
    assert summary_mod.load_summary("c1") == "压缩后的摘要"
    # 增量不足：再次调用复用旧摘要，不再调用 LLM
    with patch.object(summary_mod, "_summarize") as mock:
        out2 = summary_mod.maybe_summarize(_messages(11), "c1", "u1", llm=fake_llm)
    mock.assert_not_called()
    assert out2 == "压缩后的摘要"


def test_human_turn_count():
    assert summary_mod._human_turn_count(_messages(4)) == 4
    assert summary_mod._human_turn_count([]) == 0


def test_volume_trigger_fires_below_turn_threshold():
    _tmp_db()
    fake_llm = lambda p: "体积触发的摘要"
    out = summary_mod.maybe_summarize(
        _messages(5), "c1", "u1", llm=fake_llm, volume_chars=20000
    )
    assert out == "体积触发的摘要"
    assert summary_mod.load_summary("c1") == "体积触发的摘要"


def test_low_volume_below_turn_threshold_skips():
    _tmp_db()
    with patch.object(summary_mod, "_summarize") as mock:
        out = summary_mod.maybe_summarize(
            _messages(5), "c1", "u1", llm=lambda p: "x", volume_chars=100
        )
    mock.assert_not_called()
    assert out == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] summary 全部 {len(fns)} 个单测通过")
