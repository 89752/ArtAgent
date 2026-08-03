# tests/test_tools.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_semantic_search():
    print("\n=== Test 1: Semantic Search ===")
    from src.tools.retrieval import semantic_search

    results = semantic_search.invoke(
        {"query": "Renaissance painting with golden background", "top_k": 3}
    )
    # Stage 3 起语义检索融合用户 PDF（带 source 键的文档片段，无 author 字段），
    # 画作形状只对 semart 结果保证；断言只针对画作，文档片段不影响测试。
    semart_hits = [r for r in results if "author" in r]
print(f"  core hits: {len(semart_hits)}/{len(results)}")
    for r in semart_hits:
        print(f"  - {r['title']} by {r['author']} ({r['date']})")
    assert semart_hits, "语义检索应至少返回 1 条画作结果"
    print("✅ semantic_search OK")


def test_exact_lookup():
    print("\n=== Test 2: Exact Lookup ===")
    from src.tools.retrieval import exact_lookup

    results = exact_lookup.invoke({"author": "GOGH", "top_k": 3})
    for r in results:
        print(f"  - {r['title']} by {r['author']}")
    print("✅ exact_lookup OK")


def test_knowledge():
    print("\n=== Test 3: Painter Knowledge（结构化统计，去 LLM 化） ===")
    from src.tools.knowledge import query_painter_knowledge

    result = query_painter_knowledge.invoke({"painter_name": "Vincent van Gogh"})
    print(f"  Painter: {result['painter']} | found={result['found']}")
    print(f"  Works: {result['works_count']} | Schools: {result['main_schools']}")
    print(f"  Timeframes: {result['active_timeframes']}")
    print(f"  Sample: {result['sample_works'][:2]}")
    assert result["found"] and result["works_count"] > 0
    print("✅ query_painter_knowledge OK")


def test_image_lookup():
    print("\n=== Test 4: Image Lookup（查找模式，不调视觉模型） ===")
    from src.tools.image_lookup import image_lookup

    results = image_lookup.invoke({"author": "Turner", "top_k": 2})
    for r in results:
        print(f"  - {r['title']} by {r['author']} → {r['image_path']}")
    assert results and all("image_path" in r for r in results)
    print("✅ image_lookup OK")


if __name__ == "__main__":
    test_semantic_search()
    test_exact_lookup()
    test_knowledge()
    test_image_lookup()
    print("\n🎉 All tools passed!")
