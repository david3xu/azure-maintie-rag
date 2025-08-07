#!/usr/bin/env python3
"""
Simple Context Retrieval - CODING_STANDARDS Compliant
Clean context retrieval script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import UniversalSearchAgent
from infrastructure.azure_openai.openai_client import AzureOpenAIClient
from infrastructure.azure_storage.storage_client import SimpleStorageClient


async def retrieve_context(query: str, limit: int = 5):
    """Simple context retrieval for query"""
    print(f"📚 Context Retrieval: '{query}'")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Initialize search agent
        search_agent = UniversalSearchAgent(azure_services)

        # Get relevant context
        result = await search_agent.process_query({"query": query, "limit": limit})

        if result.get("success"):
            results = result.get("results", [])
            context_text = ""

            for item in results:
                content = item.get("content", "")[:200]  # First 200 chars
                context_text += f"{content}... "

            print(f"✅ Retrieved context from {len(results)} sources")
            return {"context": context_text.strip(), "sources": len(results)}
        else:
            print(f"❌ Context retrieval failed: {result.get('error')}")
            return None

    except Exception as e:
        print(f"❌ Context retrieval failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple context retrieval")
    parser.add_argument("query", help="Query for context retrieval")
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of context sources"
    )
    args = parser.parse_args()

    result = asyncio.run(retrieve_context(args.query, args.limit))
    sys.exit(0 if result else 1)
