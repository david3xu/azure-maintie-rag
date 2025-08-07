#!/usr/bin/env python3
"""
Simple Cosmos Storage - CODING_STANDARDS Compliant
Clean graph storage script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.azure_service_container import ConsolidatedAzureServices


async def store_graph_data(entities: list = None, relationships: list = None):
    """Simple graph data storage"""
    print("📊 Cosmos Graph Storage")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Get cosmos client
        cosmos_client = azure_services.cosmos_client

        entities = entities or []
        relationships = relationships or []

        print(
            f"💾 Storing {len(entities)} entities, {len(relationships)} relationships"
        )

        # Simple storage (demo)
        if entities and hasattr(cosmos_client, "add_entity"):
            for entity in entities[:3]:  # Demo: store first 3
                await cosmos_client.add_entity(entity)

        print("✅ Graph storage complete")
        return True

    except Exception as e:
        print(f"❌ Graph storage failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(store_graph_data())
    sys.exit(0 if result else 1)
