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

from agents.core.azure_service_container import ConsolidatedAzureServices


async def validate_azure_services():
    """Simple Azure services validation"""
    print("🔍 Validating Azure Services...")
    
    try:
        # Initialize consolidated services
        azure_services = ConsolidatedAzureServices()
        
        # Test all services
        service_status = await azure_services.initialize_all_services()
        
        print("\n📊 Results:")
        for service, status in service_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {service}")
        
        # Get overall health
        health = azure_services.get_service_status()
        print(f"\n🎯 Overall: {health['successful_services']}/{health['total_services']} services ready")
        
        return health['overall_health']
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(validate_azure_services())
    sys.exit(0 if result else 1)