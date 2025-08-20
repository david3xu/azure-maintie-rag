#!/usr/bin/env python3
"""
Query Generation Showcase - Phase 5 Integration
===============================================

Demonstrates advanced query capabilities with tri-modal search integration.
Shows real-world scenarios using Vector + Graph + GNN search modalities.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import run_universal_search


async def showcase_query_scenarios():
    """Showcase various query types and search capabilities"""
    print("🔍 QUERY GENERATION SHOWCASE")
    print("=" * 50)
    print("Demonstrating tri-modal search capabilities")
    print("(Vector + Graph + GNN integration)")
    print()

    # Define test queries from different complexity levels
    test_queries = [
        {
            "id": "basic_concept",
            "query": "What are Azure AI Language services?",
            "description": "Basic concept query - tests vector search",
            "expected_modalities": ["Vector"]
        },
        {
            "id": "relationship_query",
            "query": "How do Azure AI services connect to machine learning workflows?",
            "description": "Relationship query - tests graph search",
            "expected_modalities": ["Vector", "Graph"]
        },
        {
            "id": "complex_reasoning",
            "query": "What are the best practices for integrating Azure AI Language with enterprise architectures?",
            "description": "Complex reasoning - tests GNN capabilities",
            "expected_modalities": ["Vector", "Graph", "GNN"]
        }
    ]

    results = []
    total_queries = len(test_queries)

    for i, test_case in enumerate(test_queries, 1):
        print(f"📋 Query {i}/{total_queries}: {test_case['id']}")
        print(f"   ❓ Query: {test_case['query']}")
        print(f"   📝 Type: {test_case['description']}")
        print(f"   🎯 Expected modalities: {', '.join(test_case['expected_modalities'])}")

        start_time = time.time()

        try:
            # Run the universal search
            result = await run_universal_search(
                test_case['query'],
                max_results=5
            )

            query_time = time.time() - start_time

            # Extract results information
            search_results = result.get('search_results', [])
            modalities_used = result.get('modalities_used', [])

            print(f"   ✅ Success: {len(search_results)} results in {query_time:.2f}s")
            print(f"   🔍 Modalities used: {', '.join(modalities_used) if modalities_used else 'None specified'}")

            if search_results:
                top_result = search_results[0]
                print(f"   📄 Top result: {top_result.get('title', 'N/A')[:80]}...")
                print(f"   📊 Relevance: {top_result.get('score', 'N/A')}")

            results.append({
                "query_id": test_case['id'],
                "query": test_case['query'],
                "success": True,
                "results_count": len(search_results),
                "query_time": query_time,
                "modalities_used": modalities_used,
                "expected_modalities": test_case['expected_modalities'],
                "top_result_title": search_results[0].get('title', '') if search_results else ''
            })

        except Exception as e:
            query_time = time.time() - start_time
            print(f"   ❌ Failed: {str(e)}")

            results.append({
                "query_id": test_case['id'],
                "query": test_case['query'],
                "success": False,
                "error": str(e),
                "query_time": query_time,
                "modalities_used": [],
                "expected_modalities": test_case['expected_modalities']
            })

        print()

    # Summary results
    print("📊 SHOWCASE SUMMARY")
    print("=" * 30)

    successful_queries = sum(1 for r in results if r['success'])
    total_time = sum(r['query_time'] for r in results)
    avg_time = total_time / len(results) if results else 0

    print(f"✅ Successful queries: {successful_queries}/{total_queries}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⏱️  Average time per query: {avg_time:.2f}s")

    # Modality analysis
    all_modalities = set()
    for r in results:
        if r['success']:
            all_modalities.update(r['modalities_used'])

    print(f"🔍 Modalities demonstrated: {', '.join(sorted(all_modalities)) if all_modalities else 'None'}")

    # Save detailed results
    results_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "total_time": total_time,
            "average_time": avg_time,
            "modalities_used": list(all_modalities)
        },
        "query_results": results
    }

    # Save to results directory
    from scripts.dataflow.utilities.path_utils import get_results_dir
    results_dir = get_results_dir()
    results_file = results_dir / "query_showcase_results.json"

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"📄 Detailed results saved to: {results_file}")

    if successful_queries == total_queries:
        print(f"\n🎉 SHOWCASE COMPLETED SUCCESSFULLY")
        print("All query types demonstrated successfully")
        return True
    else:
        print(f"\n⚠️  SHOWCASE PARTIALLY COMPLETED")
        print(f"Some queries failed - check individual results above")
        return successful_queries > 0


async def main():
    """Main execution function"""
    print("🚀 Starting Query Generation Showcase...")
    print("This demonstrates the full range of search capabilities")
    print("available in the Azure RAG system.\n")

    success = await showcase_query_scenarios()
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print("\n✅ Query showcase completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Query showcase had issues")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Query showcase error: {e}")
        sys.exit(1)
