#!/usr/bin/env python3
"""
Simple Unified Search - CODING_STANDARDS Compliant
Clean search script using UniversalSearchAgent.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import UniversalSearchAgent
from agents.core.azure_service_container import ConsolidatedAzureServices


async def unified_search(query: str):
    """Simple unified search across all modalities"""
    print(f"🔍 Unified Search: '{query}'")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Initialize search agent
        search_agent = UniversalSearchAgent(azure_services)

        # Execute search
        result = await search_agent.process_query({"query": query, "limit": 10})

        if result.get("success"):
            results = result.get("results", [])
            print(f"✅ Found {len(results)} results")

            for i, item in enumerate(results[:3], 1):
                title = item.get("title", "No title")[:40]
                score = item.get("score", 0)
                print(f"   {i}. {title}... (score: {score:.2f})")

            return results
        else:
            print(f"❌ Search failed: {result.get('error')}")
            return []

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple unified search")
    parser.add_argument("query", help="Search query")
    args = parser.parse_args()

    results = asyncio.run(unified_search(args.query))
    sys.exit(0 if results else 1)
