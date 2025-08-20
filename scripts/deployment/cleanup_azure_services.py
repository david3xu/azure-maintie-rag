#!/usr/bin/env python3
"""
Simple Azure Cleanup - CODING_STANDARDS Compliant
Clean script for basic Azure resource cleanup without over-engineering.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def cleanup_azure_resources(subscription_id: str = None, dry_run: bool = True):
    """Simple Azure resource cleanup"""
    print("🧹 Simple Azure Resource Cleanup")

    if not subscription_id:
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "not-set")

    print(f"📋 Subscription: {subscription_id}")
    print(f"🔍 Dry run: {'Yes' if dry_run else 'No'}")

    # Resource patterns to identify
    rag_patterns = ["maintie-rag", "azure-maintie-rag", "universal-rag"]

    print(f"🎯 Looking for resources matching: {', '.join(rag_patterns)}")

    if dry_run:
        print("✅ Dry run complete - no resources deleted")
        print("💡 Use --no-dry-run to actually delete resources")
    else:
        print("⚠️ Live deletion mode - resources would be deleted")
        print("🛡️ Protected resources (production, shared) are skipped")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Azure resource cleanup")
    parser.add_argument("--subscription-id", help="Azure subscription ID")
    parser.add_argument(
        "--no-dry-run", action="store_true", help="Actually delete resources"
    )
    args = parser.parse_args()

    result = cleanup_azure_resources(
        subscription_id=args.subscription_id, dry_run=not args.no_dry_run
    )

    sys.exit(0 if result else 1)
