#!/usr/bin/env python3
"""
Azure Configuration Tool
Consolidated script for Azure configuration validation and setup
Replaces: azure_config_validator.py, azure_credentials_setup.sh, load_env_and_setup_azure.py
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService

async def main():
    """Main configuration tool entry point"""
    infrastructure = InfrastructureService()
    
    print("🔧 Azure Configuration Tool")
    print("="*50)
    
    # Initialize and validate all Azure services
    await infrastructure.initialize()
    health_results = infrastructure.check_all_services_health()
    
    print(f"Overall Status: {health_results['overall_status']}")
    print(f"Services Health: {health_results['summary']['healthy_services']}/{health_results['summary']['total_services']}")
    
    for service_name, service_health in health_results['services'].items():
        status_icon = "✅" if service_health['status'] == 'healthy' else "❌"
        print(f"  {status_icon} {service_name}: {service_health['status']}")
    
    return 0 if health_results['overall_status'] == 'healthy' else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))