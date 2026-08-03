"""查询改写 + 拆分纯单测：归一化、容错回落、历史窗口、开关。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.rewrite import (
    RewriteResult,
    normalize_query,
    rewrite_and_split,
    rewrite_enabled,
)


def test_normalize_query_strips_quotes():
    assert normalize_query('  "这幅画"  ') == "这幅画"
    assert normalize_query("“星月夜”") == "星月夜"
    assert normalize_query("") == ""


def test_rewrite_enabled_default_and_toggle():
    assert rewrite_enabled() is True
    os.environ["REWRITE_ENABLED"] = "0"
    try:
        assert rewrite_enabled() is False
    finally:
        os.environ.pop("REWRITE_ENABLED", None)
    assert rewrite_enabled() is True


def test_llm_rewrite_and_split_success():
    def fake_llm(prompt):
        assert "最新问题" in prompt
        return (
            '{"rewritten_question": "对比莫奈和梵高的色彩，'
            '并推荐类似的画", "sub_questions": ["对比莫奈和梵高的色彩", '
            '"推荐几幅类似莫奈的风景画"]}'
        )

    result = rewrite_and_split("对比莫奈和梵高，顺便推荐类似的画", llm=fake_llm)
    assert result.rewritten_question.startswith("对比莫奈和梵高的色彩")
    assert len(result.sub_questions) == 2


def test_llm_failure_falls_back_to_normalized():
    def boom(prompt):
        raise RuntimeError("llm down")

    result = rewrite_and_split("  找梵高的画  ", llm=boom)
    assert result == RewriteResult("找梵高的画", ["找梵高的画"])


def test_malformed_json_falls_back():
    result = rewrite_and_split("找莫奈的画", llm=lambda p: "not json")
    assert result.rewritten_question == "找莫奈的画"
    assert result.sub_questions == ["找莫奈的画"]


def test_empty_sub_questions_becomes_rewritten():
    raw = '{"rewritten_question": "什么是巴洛克", "sub_questions": []}'
    result = rewrite_and_split("什么是巴洛克", llm=lambda p: raw)
    assert result.rewritten_question == "什么是巴洛克"
    assert result.sub_questions == ["什么是巴洛克"]


def test_key_entities_and_ambiguous_parsed():
    raw = (
        '{"rewritten_question": "莫奈的睡莲有哪些", "sub_questions": [], '
        '"key_entities": ["Monet", "Water Lilies"], "ambiguous": true}'
    )
    result = rewrite_and_split("就是那个，莫奈的睡莲，你懂的", llm=lambda p: raw)
    assert result.key_entities == ["Monet", "Water Lilies"]
    assert result.ambiguous is True


def test_missing_new_fields_default_safe():
    raw = '{"rewritten_question": "什么是巴洛克", "sub_questions": []}'
    result = rewrite_and_split("什么是巴洛克", llm=lambda p: raw)
    assert result.key_entities == []
    assert result.ambiguous is False


def test_rewrite_prompt_asks_compression_and_extraction():
    def fake_llm(prompt):
        assert "压缩" in prompt or "去掉口头禅" in prompt
        assert "key_entities" in prompt
        assert "ambiguous" in prompt
        return ('{"rewritten_question": "q", "sub_questions": [], '
                '"key_entities": [], "ambiguous": false}')

    rewrite_and_split("就是那个，嗯，你懂的", llm=fake_llm)


def test_history_only_keeps_last_two_turns():
    history = [
        HumanMessage(content="第1轮：找梵高的画"),
        AIMessage(content="第1轮回答"),
        HumanMessage(content="第2轮：这幅画呢"),
        AIMessage(content="第2轮回答"),
        HumanMessage(content="第3轮：它现在在哪里"),
    ]

    def fake_llm(prompt):
        assert "第1轮：找梵高的画" not in prompt  # 首轮被丢弃
        assert "第3轮" in prompt
        assert "第2轮" in prompt
        return '{"rewritten_question": "《星月夜》现在收藏在哪里？", "sub_questions": []}'

    result = rewrite_and_split("它现在在哪里", history, llm=fake_llm)
    assert "《星月夜》" in result.rewritten_question


def test_disabled_skips_llm():
    os.environ["REWRITE_ENABLED"] = "0"
    try:
        def boom(prompt):
            raise AssertionError("关闭后不应调用 LLM")

        result = rewrite_and_split(" 找伦勃朗的画 ", llm=boom)
        assert result.rewritten_question == "找伦勃朗的画"
    finally:
        os.environ.pop("REWRITE_ENABLED", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] rewrite 全部 {len(fns)} 个单测通过")
