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
    for r in results:
        print(f"  - {r['title']} by {r['author']} ({r['date']})")
    print("✅ semantic_search OK")


def test_exact_lookup():
    print("\n=== Test 2: Exact Lookup ===")
    from src.tools.retrieval import exact_lookup

    results = exact_lookup.invoke({"author": "GOGH", "top_k": 3})
    for r in results:
        print(f"  - {r['title']} by {r['author']}")
    print("✅ exact_lookup OK")


def test_knowledge():
    print("\n=== Test 3: Painter Knowledge ===")
    from src.tools.knowledge import query_painter_knowledge

    result = query_painter_knowledge.invoke(
        {
            "painter_name": "Vincent van Gogh",
            "question": "What are the main characteristics of his painting style?",
        }
    )
    print(f"  Painter: {result['painter']}")
    print(f"  Answer (first 200 chars): {result['answer'][:200]}...")
    print("✅ query_painter_knowledge OK")


def test_style_comparison():
    print("\n=== Test 4: Style Comparison ===")
    from src.tools.style_comparison import compare_artwork_styles

    result = compare_artwork_styles.invoke(
        {
            "artwork1_title": "The Starry Night",
            "artwork2_title": "Sunflowers",
            "comparison_aspects": "style",
        }
    )
    print(f"  Comparing: {result['artwork1']} vs {result['artwork2']}")
    print(f"  Comparison (first 200 chars): {result['comparison'][:200]}...")
    print("✅ compare_artwork_styles OK")


if __name__ == "__main__":
    test_semantic_search()
    test_exact_lookup()
    test_knowledge()
    test_style_comparison()
    print("\n🎉 All tools passed!")
