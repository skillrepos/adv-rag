#!/usr/bin/env python3
"""
Test persistence of the RAG knowledge base
"""

import os
import time
from rag_skeleton import KnowledgeBase

def test_persistence():
    """Test that the knowledge base persists correctly"""
    print("="*60)
    print("Testing Knowledge Base Persistence")
    print("="*60)

    # Phase 1: Initial load and search
    print("\n[Phase 1] Initial Load")
    print("-"*40)

    kb1 = KnowledgeBase("../knowledge_base_pdfs")
    stats1 = kb1.get_statistics()

    print(f"✓ Loaded {stats1['total_documents']} documents")
    print(f"✓ Categories: {', '.join(stats1['categories'].keys())}")

    # Perform some searches and save results
    test_query = "return policy"
    print(f"\nSearching for: '{test_query}'")
    results1 = kb1.search(test_query, max_results=2)

    if results1:
        print(f"✓ Found {len(results1)} results")
        first_result_id = results1[0]['id']
        first_result_score = results1[0]['score']
        print(f"  First result ID: {first_result_id}")
        print(f"  First result score: {first_result_score:.3f}")
    else:
        print("⚠ No results found")
        first_result_id = None
        first_result_score = None

    # Phase 2: Create new instance and verify
    print("\n[Phase 2] Creating New Instance")
    print("-"*40)

    # Small delay to simulate restart
    time.sleep(1)

    kb2 = KnowledgeBase("../knowledge_base_pdfs")
    stats2 = kb2.get_statistics()

    print(f"✓ Loaded {stats2['total_documents']} documents")

    # Verify document count matches
    if stats1['total_documents'] == stats2['total_documents']:
        print("✓ Document count matches original")
    else:
        print(f"⚠ Document count mismatch: {stats1['total_documents']} vs {stats2['total_documents']}")

    # Phase 3: Verify search consistency
    print("\n[Phase 3] Verifying Search Consistency")
    print("-"*40)

    results2 = kb2.search(test_query, max_results=2)

    if results2:
        print(f"✓ Found {len(results2)} results")

        if first_result_id and results2[0]['id'] == first_result_id:
            print("✓ First result ID matches")
        else:
            print("⚠ First result ID differs")

        if first_result_score and abs(results2[0]['score'] - first_result_score) < 0.01:
            print("✓ First result score matches (within tolerance)")
        else:
            print(f"⚠ First result score differs: {first_result_score:.3f} vs {results2[0]['score']:.3f}")
    else:
        print("⚠ No results found in second instance")

    # Phase 4: Test multiple queries
    print("\n[Phase 4] Testing Multiple Query Consistency")
    print("-"*40)

    test_queries = [
        "shipping costs",
        "password reset",
        "device troubleshooting"
    ]

    all_match = True
    for query in test_queries:
        r1 = kb1.search(query, max_results=1)
        r2 = kb2.search(query, max_results=1)

        if len(r1) == len(r2):
            if r1 and r2 and r1[0]['id'] == r2[0]['id']:
                print(f"✓ Query '{query}' returns consistent results")
            elif not r1 and not r2:
                print(f"✓ Query '{query}' consistently returns no results")
            else:
                print(f"⚠ Query '{query}' returns different results")
                all_match = False
        else:
            print(f"⚠ Query '{query}' returns different number of results")
            all_match = False

    # Phase 5: Test document retrieval
    print("\n[Phase 5] Document Integrity Check")
    print("-"*40)

    # Check that categories are consistent
    cats1 = set(stats1['categories'].keys())
    cats2 = set(stats2['categories'].keys())

    if cats1 == cats2:
        print("✓ Categories are consistent")
    else:
        print(f"⚠ Category mismatch:")
        print(f"  Only in first: {cats1 - cats2}")
        print(f"  Only in second: {cats2 - cats1}")

    # Summary
    print("\n" + "="*60)
    print("Persistence Test Summary")
    print("="*60)

    tests_passed = 0
    tests_total = 5

    if stats1['total_documents'] == stats2['total_documents']:
        tests_passed += 1
    if results1 and results2 and results1[0]['id'] == results2[0]['id']:
        tests_passed += 1
    if all_match:
        tests_passed += 1
    if cats1 == cats2:
        tests_passed += 1
    if stats1['total_characters'] == stats2['total_characters']:
        tests_passed += 1

    print(f"Tests Passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("✅ All persistence tests passed!")
        print("The knowledge base maintains consistency across instances.")
    elif tests_passed >= 3:
        print("⚠ Most persistence tests passed with minor issues.")
        print("The system is mostly consistent but may have small variations.")
    else:
        print("❌ Persistence tests failed.")
        print("The knowledge base may not be persisting correctly.")

    return tests_passed == tests_total

if __name__ == "__main__":
    success = test_persistence()
    exit(0 if success else 1)