#!/usr/bin/env python3
"""
Test script for the RAG system
"""

from rag_skeleton import KnowledgeBase

def main():
    # Initialize the knowledge base
    print("Initializing Knowledge Base...")
    kb = KnowledgeBase("../knowledge_base_pdfs")

    # Test queries
    queries = [
        "How do I return a product?",
        "What are the shipping options?",
        "How can I reset my password?",
        "My device won't turn on",
        "What payment methods are accepted?"
    ]

    print("\n" + "="*60)
    print("Testing RAG System with Various Queries")
    print("="*60)

    for query in queries:
        print(f"\n[Query]: {query}")
        print("-" * 50)

        results = kb.search(query, max_results=2)

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Category: {result['category']}")
                print(f"    Score: {result['score']:.3f}")
                print(f"    Source: {result['source']}")
                print(f"    Content Preview: {result['content'][:150]}...")
        else:
            print("  No results found")

    # Show statistics
    print("\n" + "="*60)
    print("Knowledge Base Statistics")
    print("="*60)

    stats = kb.get_statistics()
    if stats:
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Characters: {stats['total_characters']}")
        print("\nDocuments by Category:")
        for category, count in stats['categories'].items():
            print(f"  - {category}: {count} document(s)")

if __name__ == "__main__":
    main()