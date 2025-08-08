#!/usr/bin/env python3
"""
Simple Azure Services Setup - CODING_STANDARDS Compliant
Clean script to validate Azure services without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps


async def validate_azure_services():
    """Simple Azure services validation"""
    print("🔍 Validating Azure Services...")

    try:
        # Initialize universal dependencies
        universal_deps = await get_universal_deps()

        # Test all services
        service_status = await universal_deps.initialize_all_services()

        print("\n📊 Results:")
        for service, status in service_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {service}")

        # Get overall health
        available_services = universal_deps.get_available_services()
        total_services = len(service_status)
        print(
            f"\n🎯 Overall: {len(available_services)}/{total_services} services ready"
        )

        return len(available_services) >= (total_services // 2)

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(validate_azure_services())
    sys.exit(0 if result else 1)
