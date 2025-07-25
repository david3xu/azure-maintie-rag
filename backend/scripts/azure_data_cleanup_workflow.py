#!/usr/bin/env python3
"""
Enterprise Azure Data Cleanup Workflow
=====================================
Azure Universal RAG data lifecycle management orchestration
Implements clean data reset across all Azure services while preserving infrastructure
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from integrations.azure_services import AzureServicesManager
from config.settings import azure_settings

async def main():
    """Execute enterprise Azure data cleanup workflow"""

    print("🧹 Enterprise Azure Universal RAG - Data Cleanup Workflow")
    print("=" * 60)
    print("📊 Purpose: Clean all data from Azure services while preserving infrastructure")
    print("☁️  Azure Services: Blob Storage, Cognitive Search, Cosmos DB")
    print("⏱️  Enterprise Operation: Automated data lifecycle management")
    print()

    workflow_start = time.time()
    domain = "general"

    try:
        # Initialize Azure services using existing enterprise patterns
        print("📝 Initializing Azure services manager...")
        azure_services = AzureServicesManager()

        # Enterprise validation before cleanup
        print("🔍 Validating Azure services configuration...")
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            print(f"❌ Azure services validation failed: {validation}")
            return 1

        print("✅ Azure services validated successfully")

        # Execute automated cleanup across all Azure services
        print(f"\n🧹 Step 1: Executing automated data cleanup for domain '{domain}'...")
        cleanup_result = await azure_services.cleanup_all_azure_data(domain)

        # Enterprise cleanup reporting
        print(f"\n📊 Azure Service Cleanup Results:")
        for service_name, service_result in cleanup_result["cleanup_results"].items():
            if service_result["success"]:
                status = "✅"
                details = ""

                # Service-specific cleanup metrics
                if service_name == "blob_storage":
                    details = f"({service_result.get('blobs_deleted', 0)} blobs deleted)"
                elif service_name == "cognitive_search":
                    details = f"({service_result.get('documents_deleted', 0)} documents deleted)"
                elif service_name == "cosmos_db":
                    details = f"({service_result.get('entities_deleted', 0)} entities deleted)"

                print(f"   {status} {service_name.replace('_', ' ').title()}: Data cleaned {details}")
            else:
                print(f"   ❌ {service_name.replace('_', ' ').title()}: {service_result.get('error', 'Unknown error')}")

        # Enterprise workflow summary
        workflow_duration = time.time() - workflow_start
        print(f"\n📈 Enterprise Data Cleanup Summary:")
        print(f"   ⏱️  Total Duration: {workflow_duration:.2f} seconds")
        print(f"   🏗️  Infrastructure Status: Preserved")
        print(f"   📊 Data State: Reset to clean slate")
        print(f"   🚀 Readiness: Ready for fresh data processing")

        if cleanup_result["success"]:
            print(f"\n✅ Enterprise Azure data cleanup completed successfully!")
            print(f"💡 Azure infrastructure preserved - ready for data-prep-enterprise")
            return 0
        else:
            print(f"\n❌ Azure data cleanup encountered issues")
            return 1

    except Exception as e:
        print(f"❌ Enterprise Azure data cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)