"""
混合架构四条分支的端到端冒烟测试。
每条打印意图、节点执行链、最终答案摘要。
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import get_graph
from src.memory.store import clear_preferences


def run(graph, question, thread_id, user_id="smoke_user"):
    print(f"\n{'='*64}\nQ: {question}\n{'='*64}")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "user_id": user_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"  intent        = {result.get('intent')}")
    print(f"  last step     = {result.get('current_step')}")
    print(f"  reflection    = {result.get('reflection_notes')}")
    print(f"  #artworks     = {len(result.get('artworks', []))}")
    print(f"  #images       = {len(result.get('images', []))}")
    if result.get("subjects"):
        print(f"  subjects      = {result.get('subjects')}")
    if result.get("extracted_features"):
        print(f"  features      = {result['extracted_features'][:120]}")
    if result.get("candidates"):
        print(f"  candidates    = {[c['author'] for c in result['candidates']]}")
    # tool calls (general branch)
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  🔧 {tc['name']}({tc['args']})")
    ans = result.get("final_answer") or result["messages"][-1].content
    print(f"\n  A: {ans[:500]}{'...' if len(ans) > 500 else ''}")
    return result


if __name__ == "__main__":
    graph = get_graph()
    clear_preferences("smoke_user")

    # 场景1：跨维度对比
    run(graph, "帮我对比一下莫奈和梵高在色彩运用上的差异", "cmp-1")

    # 场景3：偏好推荐（核心亮点）
    run(graph, "我喜欢梵高那种浓烈奔放的风格，还有什么画家我可能会喜欢？", "rec-1")

    # 场景2：时间线（用数据集覆盖较好的画家）
    run(graph, "梳理一下透纳的风格演变", "tl-1")

    # general：单一事实查询
    run(graph, "找几幅伦勃朗的画介绍一下", "gen-1")

    # 场景5：跨会话记忆——再问推荐，应带出之前记住的偏好
    print("\n\n### 复用同一 user_id，验证长期记忆 ###")
    run(graph, "再给我推荐一位画家", "rec-2")

    clear_preferences("smoke_user")
    print("\n\n✅ 冒烟测试完成")
