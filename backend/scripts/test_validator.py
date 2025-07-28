#!/usr/bin/env python3
"""
Testing Tool
Consolidated script for test execution and validation
Replaces: test_*.py files, validate_*.py files, verify_*.py files
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService

async def main():
    """Main testing tool entry point"""
    infrastructure = InfrastructureService()
    
    print("🧪 Testing Tool")
    print("="*50)
    
    # Run comprehensive system tests
    print("1. Testing Azure service connectivity...")
    health_results = infrastructure.check_all_services_health()
    
    healthy_services = health_results['summary']['healthy_services']
    total_services = health_results['summary']['total_services']
    
    print(f"   Services: {healthy_services}/{total_services} healthy")
    
    if healthy_services == total_services:
        print("✅ All connectivity tests passed")
        return 0
    else:
        print("❌ Some connectivity tests failed")
        
        # Show failed services
        for service_name, service_health in health_results['services'].items():
            if service_health['status'] != 'healthy':
                print(f"   ❌ {service_name}: {service_health.get('error', 'Unknown error')}")
        
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))