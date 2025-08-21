#!/usr/bin/env python3
"""
Simple Query Pipeline - CODING_STANDARDS Compliant
Clean query processing script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import run_universal_search


async def process_query(query: str, domain: str = "general"):
    """Simple query processing pipeline"""
    print(f"🔍 Processing Query: '{query}'")

    try:
        # Use the same approach as 04_01_query_analysis.py that works
        result = await run_universal_search(
            query=query,
            max_results=10,
            use_domain_analysis=True
        )

        print(f"✅ Found {result.total_results_found} total results")
        print(f"🎯 Search confidence: {result.search_confidence:.2f}")
        print(f"📊 Strategy used: {result.search_strategy_used}")

        if result.unified_results:
            print(f"📋 Top results:")
            for i, result_item in enumerate(result.unified_results[:3], 1):
                title = result_item.title[:50] if result_item.title else "No title"
                score = result_item.score
                print(f"   {i}. {title}... (score: {score:.3f})")

        return result.total_results_found > 0

    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple query pipeline")
    parser.add_argument("query", help="Query to process")
    parser.add_argument("--domain", default="general", help="Query domain")
    args = parser.parse_args()

    result = asyncio.run(process_query(args.query, args.domain))
    sys.exit(0 if result else 1)
