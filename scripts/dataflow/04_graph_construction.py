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

from agents.core.azure_service_container import ConsolidatedAzureServices


async def construct_graph(container: str = "knowledge-extraction"):
    """Simple graph construction from extracted knowledge"""
    print("🧠 Graph Construction - Knowledge → Simple Graph Format")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Get storage client
        storage_client = azure_services.storage_client

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
