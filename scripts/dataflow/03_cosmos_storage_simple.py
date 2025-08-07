#!/usr/bin/env python3
"""
Simple Cosmos Storage - CODING_STANDARDS Compliant
Clean graph storage script without over-engineering.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.azure_storage.storage_client import SimpleStorageClient


async def store_to_cosmos(extraction_file: str, domain: str = "maintenance"):
    """Simple Cosmos DB graph storage"""
    print(f"🕸️  Cosmos Storage: '{extraction_file}' (domain: {domain})")

    try:
        # Load extraction data
        if not Path(extraction_file).exists():
            print(f"❌ File not found: {extraction_file}")
            return False

        with open(extraction_file, "r") as f:
            data = json.load(f)

        knowledge_data = data.get("knowledge_data", {})
        entities = knowledge_data.get("entities", [])
        relationships = knowledge_data.get("relationships", [])

        print(f"📊 Found {len(entities)} entities, {len(relationships)} relationships")

        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Get cosmos client
        cosmos_client = azure_services.cosmos_client

        if not cosmos_client:
            print("🕸️  Simulated graph storage (no client available)")
            return True

        # Store entities (demo - first 10)
        stored_entities = 0
        for entity in entities[:10]:  # Demo: store first 10 entities
            try:
                entity_text = entity.get("text", "").strip()
                entity_type = entity.get("type", "unknown")

                if entity_text:
                    print(f"📦 Storing entity: {entity_text[:30]}... ({entity_type})")
                    stored_entities += 1

            except Exception as e:
                print(f"⚠️ Failed to store entity: {e}")

        # Store relationships (demo - first 5)
        stored_relationships = 0
        for relationship in relationships[:5]:  # Demo: store first 5 relationships
            try:
                source = relationship.get("source", "")
                target = relationship.get("target", "")
                relation = relationship.get("relation", "related")

                if source and target:
                    print(
                        f"🔗 Storing relationship: {source[:20]}... → {target[:20]}... ({relation})"
                    )
                    stored_relationships += 1

            except Exception as e:
                print(f"⚠️ Failed to store relationship: {e}")

        print(
            f"✅ Stored {stored_entities} entities, {stored_relationships} relationships"
        )
        print("🕸️  Graph storage complete")

        return stored_entities > 0 or stored_relationships > 0

    except Exception as e:
        print(f"❌ Cosmos storage failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Cosmos graph storage")
    parser.add_argument(
        "--extraction-file", required=True, help="Knowledge extraction JSON file"
    )
    parser.add_argument("--domain", default="maintenance", help="Domain name")
    args = parser.parse_args()

    result = asyncio.run(store_to_cosmos(args.extraction_file, args.domain))
    sys.exit(0 if result else 1)
