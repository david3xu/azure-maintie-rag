#!/usr/bin/env python3
"""
Simple Query Analysis - CODING_STANDARDS Compliant
Clean query analysis script without over-engineering complex patterns.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import UniversalSearchAgent
from agents.core.azure_service_container import ConsolidatedAzureServices


async def analyze_query(query: str):
    """Simple query analysis"""
    print(f"🔍 Query Analysis: '{query}'")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Initialize search agent
        search_agent = UniversalSearchAgent(azure_services)

        # Basic query analysis
        query_info = {
            "query": query,
            "length": len(query),
            "word_count": len(query.split()),
            "type": "question" if query.endswith("?") else "statement",
        }

        print(f"📊 Analysis: {query_info['word_count']} words, {query_info['type']}")

        return query_info

    except Exception as e:
        print(f"❌ Query analysis failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple query analysis")
    parser.add_argument("query", help="Query to analyze")
    args = parser.parse_args()

    result = asyncio.run(analyze_query(args.query))
    sys.exit(0 if result else 1)
