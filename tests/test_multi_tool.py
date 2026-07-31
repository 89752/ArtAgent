"""
测试 ArtAgent 多工具链式调用能力。

观察 Agent 是否能自主决定调用多个工具完成复杂任务。
每个测试用例后打印完整的工具调用链。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import get_graph


def chat_with_trace(graph, thread_id: str, question: str) -> str:
    """发送消息并打印完整工具调用链。"""
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

    # 打印工具调用链（跳过第一条 HumanMessage）
    print("\n── 工具调用链 ──────────────────────────────")
    for msg in result["messages"][1:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                print(f"  🔧 {tc['name']}({args_str})")
        elif isinstance(msg, ToolMessage):
            content_preview = str(msg.content)[:120].replace("\n", " ")
            print(f"  📦 结果: {content_preview}...")
    print("────────────────────────────────────────────")

    return result["messages"][-1].content


def test_case(graph, thread_id: str, title: str, question: str):
    print(f"\n{'='*60}")
    print(f"场景: {title}")
    print(f"Q: {question}")
    print("=" * 60)
    answer = chat_with_trace(graph, thread_id, question)
    print(f"\nA: {answer[:3000]}...")


if __name__ == "__main__":
    graph = get_graph()

    # 场景1：找画 + 分析画家（需要 semantic_search + query_painter_knowledge）
    test_case(
        graph,
        thread_id="multi-tool-001",
        title="找印象派风景画并分析画家风格",
        question="找几幅印象派的风景画，并介绍其中一位画家的艺术特点",
    )

    # 场景2：找画 + 对比（exact_lookup 取元数据，Agent 自行组织对比）
    test_case(
        graph,
        thread_id="multi-tool-002",
        title="找梵高两幅画并对比",
        question="找两幅梵高的自画像，然后对比它们的风格差异",
    )

    # 场景3：三工具联动（semantic_search + exact_lookup + query_painter_knowledge）
    test_case(
        graph,
        thread_id="multi-tool-003",
        title="巴洛克画家深度分析",
        question="找几幅卡拉瓦乔的作品，并结合他的生平介绍他的艺术风格",
    )

    print("\n🎉 多工具链式调用测试完成！")
