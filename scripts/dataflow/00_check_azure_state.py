#!/usr/bin/env python3
"""
Simple Azure State Check - CODING_STANDARDS Compliant
Clean Azure state validation script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.azure_service_container import ConsolidatedAzureServices


async def check_azure_state(domain: str = "maintenance"):
    """Simple Azure services state check"""
    print(f"🔍 Azure State Check (domain: {domain})")

    try:
        # Initialize services
        azure_services = ConsolidatedAzureServices()
        await azure_services.initialize_all_services()

        # Check service status
        service_status = azure_services.get_service_status()

        print(f"📊 Azure Services Status:")
        print(f"   Overall Health: {service_status['overall_health']}")
        print(
            f"   Services Ready: {service_status['successful_services']}/{service_status['total_services']}"
        )

        # Check raw data availability
        raw_data_dir = Path("data/raw")
        if raw_data_dir.exists():
            md_files = list(raw_data_dir.glob("**/*.md"))
            print(f"   📁 Raw data files: {len(md_files)} found")
        else:
            print(f"   📁 Raw data directory: Not found")

        # Simple recommendations
        if service_status["successful_services"] >= 2:
            print("✅ System ready for processing")
            return True
        else:
            print("⚠️ Some services not available - check Azure configuration")
            return False

    except Exception as e:
        print(f"❌ Azure state check failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Azure state check")
    parser.add_argument("--domain", default="maintenance", help="Domain to check")
    args = parser.parse_args()

    result = asyncio.run(check_azure_state(args.domain))
    sys.exit(0 if result else 1)
