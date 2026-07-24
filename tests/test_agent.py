"""
ArtAgent 端到端测试。
覆盖：单工具调用、多工具链式调用、图像分析。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import get_graph


def chat(graph, question: str, thread_id: str = "test-default") -> str:
    """向 Agent 发送一条消息，返回最终回答。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "tool_results": [],
            "final_answer": "",
        },
        config=config,
    )

    # 打印工具调用链
    for msg in result["messages"][1:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                print(f"  🔧 {tc['name']}({args_str})")
        elif isinstance(msg, ToolMessage):
            print(f"  📦 {str(msg.content)[:100]}...")

    return result["messages"][-1].content


def run_test(graph, question: str, thread_id: str):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print("=" * 60)
    answer = chat(graph, question, thread_id)
    print(f"A: {answer[:600]}")
    if len(answer) > 600:
        print("  ...")


if __name__ == "__main__":
    graph = get_graph()

    # ── 基础工具测试 ──────────────────────────────────────────
    run_test(graph, "找几幅梵高的画介绍一下", "test-001")
    run_test(graph, "什么是巴洛克风格？给我找几幅代表作", "test-002")
    run_test(graph, "对比一下《星夜》和《向日葵》的风格差异", "test-003")
    run_test(graph, "伦勃朗的绘画有什么特点？", "test-004")

    # ── 图像分析测试 ──────────────────────────────────────────
    run_test(graph, "帮我分析一下《星夜》的构图特点", "test-img-001")
    run_test(graph, "从视觉角度介绍一下卡拉瓦乔的《鞭打基督》", "test-img-002")
    run_test(graph, "分析梵高《戴草帽的自画像》的色彩运用", "test-img-003")

    print("\n\n✅ 所有测试完成！")
