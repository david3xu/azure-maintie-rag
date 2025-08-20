#!/usr/bin/env python3
"""
Cleanup script to remove duplicate GNN endpoints before deployment.
This prevents resource waste and confusion.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.azure_ml.endpoint_manager import get_endpoint_manager


async def main():
    """Clean up duplicate endpoints."""
    print("🧹 Cleaning up duplicate GNN endpoints...")

    try:
        endpoint_manager = await get_endpoint_manager()

        # List current endpoints
        endpoints = await endpoint_manager.list_gnn_endpoints()
        print(f"📊 Found {len(endpoints)} GNN endpoints:")
        for ep in endpoints:
            print(f"   - {ep.name} (created: {ep.creation_context.created_at if ep.creation_context else 'unknown'})")

        if len(endpoints) <= 1:
            print("✅ No cleanup needed - only 1 or fewer endpoints")
            return

        # Clean up duplicates (keep only 1)
        result = await endpoint_manager.cleanup_duplicate_endpoints(keep_count=1)

        print(f"✅ Cleanup complete:")
        print(f"   - Deleted: {result['cleaned']} endpoints")
        print(f"   - Kept: {result['kept']} endpoints")

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
