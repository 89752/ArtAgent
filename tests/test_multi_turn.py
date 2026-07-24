"""
测试 ArtAgent 多轮对话记忆能力。

每个 thread_id 代表一个独立会话，同一 thread_id 内 Agent 记住上下文。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from src.agent.graph import get_graph


def chat(graph, thread_id: str, question: str) -> str:
    """向指定会话发送一条消息，返回 Agent 回复。"""
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
    return result["messages"][-1].content


def test_multi_turn():
    graph = get_graph()
    thread_id = "test-session-001"

    print("\n" + "=" * 60)
    print("多轮对话测试（同一 thread_id）")
    print("=" * 60)

    # 第1轮：找画
    q1 = "找几幅梵高的画"
    print(f"\n[第1轮] Q: {q1}")
    a1 = chat(graph, thread_id, q1)
    print(f"A: {a1[:300]}...")

    # 第2轮：引用上文（不提梵高，测试记忆）
    q2 = "刚才提到的第一幅画，能详细介绍一下它的创作背景吗？"
    print(f"\n[第2轮] Q: {q2}")
    a2 = chat(graph, thread_id, q2)
    print(f"A: {a2[:300]}...")

    # 第3轮：继续追问
    q3 = "这幅画现在收藏在哪里？"
    print(f"\n[第3轮] Q: {q3}")
    a3 = chat(graph, thread_id, q3)
    print(f"A: {a3[:300]}...")

    print("\n" + "=" * 60)
    print("隔离测试：新 thread_id 应该没有上下文记忆")
    print("=" * 60)

    # 新会话，不应该知道之前说的梵高
    q4 = "刚才提到的第一幅画是什么？"
    print(f"\n[新会话] Q: {q4}")
    a4 = chat(graph, "test-session-002", q4)
    print(f"A: {a4[:300]}...")

    print("\n✅ 多轮对话测试完成！")


if __name__ == "__main__":
    test_multi_turn()
