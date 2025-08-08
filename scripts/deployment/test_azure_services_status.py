#!/usr/bin/env python3
"""
Simple Azure Services Status Test - CODING_STANDARDS Compliant
Clean script to test Azure services without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps


async def test_azure_services_status():
    """Simple Azure services status test"""
    print("🔧 Testing Azure Services Status...")

    try:
        # Initialize services using UniversalDeps
        universal_deps = await get_universal_deps()

        # Test all services
        service_status = await universal_deps.initialize_all_services()

        print("\n📊 Service Status:")
        healthy_count = 0

        for service, status in service_status.items():
            if status:
                print(f"   ✅ {service}: Healthy")
                healthy_count += 1
            else:
                print(f"   ❌ {service}: Failed")

        # Get available services
        available_services = universal_deps.get_available_services()
        total_services = len(service_status)

        print(f"\n🎯 Overall Health:")
        print(f"   Services Ready: {len(available_services)}/{total_services}")
        
        overall_health = len(available_services) >= (total_services // 2)  # At least half services healthy
        print(f"   Overall Status: {'Healthy' if overall_health else 'Degraded'}")

        return overall_health

    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_azure_services_status())
    sys.exit(0 if result else 1)
