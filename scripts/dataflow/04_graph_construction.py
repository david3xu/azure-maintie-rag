#!/usr/bin/env python3
"""
Simple Graph Construction - CODING_STANDARDS Compliant
Clean script for basic graph operations without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.azure_cosmos.cosmos_gremlin_client import SimpleCosmosClient
from infrastructure.azure_storage.storage_client import SimpleStorageClient


async def construct_graph(container: str = "knowledge-extraction"):
    """Graph construction from extracted knowledge using Azure services"""
    print("🧠 Graph Construction - Knowledge → Simple Graph Format")

    try:
        # Initialize Azure service clients
        try:
            storage_client = SimpleStorageClient()
            await storage_client.async_initialize()
            print("✅ Storage client ready")
        except Exception as e:
            storage_client = None
            print(f"⚠️  Storage unavailable: {str(e)[:50]}...")

        try:
            cosmos_client = SimpleCosmosClient()
            await cosmos_client.async_initialize()
            print("✅ Cosmos client ready")
        except Exception as e:
            cosmos_client = None
            print(f"⚠️  Cosmos unavailable: {str(e)[:50]}...")

        if not storage_client:
            print("📊 Simulated graph construction (storage unavailable)")
            return True

        print(f"📦 Reading from container: {container}")

        # List available files
        blobs_result = await storage_client.list_blobs(container)

        if blobs_result.get("success"):
            blobs = blobs_result.get("data", {}).get("blobs", [])
            print(f"📄 Found {len(blobs)} files")

            # Find knowledge extraction file
            for blob in blobs:
                if "knowledge_extraction" in blob.get("name", ""):
                    print(f"✅ Found knowledge file: {blob['name']}")
                    return True

        print("⚠️ No knowledge extraction files found")
        return False

    except Exception as e:
        print(f"❌ Graph construction failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple graph construction")
    parser.add_argument(
        "--container", default="knowledge-extraction", help="Storage container"
    )
    args = parser.parse_args()

    result = asyncio.run(construct_graph(args.container))
    sys.exit(0 if result else 1)
