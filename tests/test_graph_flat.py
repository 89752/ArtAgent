"""扁平化图结构单测：无子管线分支节点、主路径线性直达 general_agent。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import get_graph

_BRANCH_NODES = {
    "comp_decompose", "comp_retrieve", "comp_synthesize",
    "tl_subject", "tl_periods", "tl_synthesize",
    "rec_extract", "rec_search", "rec_filter", "rec_synthesize",
}
_CORE_NODES = {
    "load_memory", "rewrite_split", "classify", "rag_gate", "direct_answer",
    "ask_user", "multi_retrieve", "general_agent", "general_tools",
    "reflection", "save_memory",
}


def _edges_of(graph) -> list[tuple[str, str]]:
    out = []
    for e in graph.edges:
        start = getattr(e, "start", None) or getattr(e, "source", None)
        end = getattr(e, "end", None) or getattr(e, "target", None)
        if start and end:
            out.append((start, end))
    return out


def test_graph_has_no_branch_nodes():
    graph = get_graph().get_graph()
    nodes = set(graph.nodes.keys())
    assert nodes & _BRANCH_NODES == set()
    assert _CORE_NODES <= nodes


def test_classify_flows_to_ask_user_then_multi_retrieve_then_general_agent():
    edges = _edges_of(get_graph().get_graph())
    assert ("classify", "rag_gate") in edges
    assert ("rag_gate", "ask_user") in edges
    assert ("rag_gate", "direct_answer") in edges
    assert ("direct_answer", "reflection") in edges
    assert ("ask_user", "multi_retrieve") in edges
    assert ("multi_retrieve", "general_agent") in edges
    # 旧分支入口不再存在
    assert not any(start.startswith(("comp_", "tl_", "rec_")) for start, _ in edges)


def test_general_react_loop_present():
    edges = _edges_of(get_graph().get_graph())
    assert ("general_tools", "general_agent") in edges


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] graph_flat 全部 {len(fns)} 个单测通过")
