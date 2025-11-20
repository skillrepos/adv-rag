#!/usr/bin/env python3
"""
Benchmark script for RAG system performance
"""

import time
import statistics
from rag_skeleton import KnowledgeBase

def benchmark_search(kb: KnowledgeBase, queries: list, iterations: int = 3):
    """Benchmark search performance"""
    print("\n" + "="*60)
    print("Benchmarking Search Performance")
    print("="*60)

    all_times = []

    for query in queries:
        query_times = []
        print(f"\nQuery: '{query}'")
        print("  Running", iterations, "iterations...")

        for i in range(iterations):
            start_time = time.time()
            results = kb.search(query, max_results=3)
            end_time = time.time()

            elapsed = end_time - start_time
            query_times.append(elapsed)

            print(f"    Iteration {i+1}: {elapsed:.4f}s ({len(results)} results)")

        avg_time = statistics.mean(query_times)
        std_dev = statistics.stdev(query_times) if len(query_times) > 1 else 0

        print(f"  Average: {avg_time:.4f}s (±{std_dev:.4f}s)")
        all_times.extend(query_times)

    return all_times

def main():
    """Run benchmark tests on the RAG system"""
    print("="*60)
    print("RAG System Performance Benchmark")
    print("="*60)

    # Initialize and time the knowledge base loading
    print("\n[Phase 1] Knowledge Base Initialization")
    print("-"*40)

    start_time = time.time()
    kb = KnowledgeBase("../knowledge_base_pdfs")
    load_time = time.time() - start_time

    print(f"✓ Knowledge base loaded in {load_time:.2f} seconds")

    # Get statistics
    stats = kb.get_statistics()
    if stats:
        print(f"✓ Loaded {stats['total_documents']} documents")
        print(f"✓ Total size: {stats['total_characters']:,} characters")

    # Benchmark different query types
    print("\n[Phase 2] Search Performance Testing")
    print("-"*40)

    test_queries = [
        # Simple queries
        "return policy",
        "shipping",
        "password reset",

        # More complex queries
        "How do I return a defective product?",
        "What are my shipping options for international orders?",
        "I forgot my password and can't access my email",

        # Edge cases
        "xyz123notfound",
        "a",
        "the quick brown fox jumps over the lazy dog"
    ]

    # Run benchmarks
    search_times = benchmark_search(kb, test_queries, iterations=3)

    # Calculate overall statistics
    print("\n" + "="*60)
    print("Overall Performance Summary")
    print("="*60)

    if search_times:
        print(f"Total searches performed: {len(search_times)}")
        print(f"Average search time: {statistics.mean(search_times):.4f}s")
        print(f"Median search time: {statistics.median(search_times):.4f}s")
        print(f"Min search time: {min(search_times):.4f}s")
        print(f"Max search time: {max(search_times):.4f}s")

        if len(search_times) > 1:
            print(f"Standard deviation: {statistics.stdev(search_times):.4f}s")

    # Test memory efficiency
    print("\n[Phase 3] Memory and Efficiency")
    print("-"*40)

    # Perform rapid consecutive searches
    rapid_start = time.time()
    for _ in range(10):
        kb.search("test query", max_results=1)
    rapid_time = time.time() - rapid_start

    print(f"✓ 10 rapid searches completed in {rapid_time:.2f}s")
    print(f"✓ Average time per rapid search: {rapid_time/10:.4f}s")

    # Performance grade
    print("\n" + "="*60)
    print("Performance Grade")
    print("="*60)

    avg_search = statistics.mean(search_times) if search_times else 0

    if avg_search < 0.1:
        grade = "A+ (Excellent)"
    elif avg_search < 0.3:
        grade = "A (Very Good)"
    elif avg_search < 0.5:
        grade = "B (Good)"
    elif avg_search < 1.0:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Optimization)"

    print(f"Grade: {grade}")
    print(f"Based on average search time of {avg_search:.3f} seconds")

    print("\n✅ Benchmark completed successfully!")

if __name__ == "__main__":
    main()