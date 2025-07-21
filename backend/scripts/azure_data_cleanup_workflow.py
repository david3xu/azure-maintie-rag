#!/usr/bin/env python3
"""
Enterprise Azure Data Cleanup Workflow
=====================================
Automated cleanup of Azure Universal RAG data across all services
Preserves infrastructure while cleaning all data
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
    """Execute automated Azure data cleanup workflow"""

    print("🧹 Enterprise Azure Universal RAG - Data Cleanup Workflow")
    print("=" * 60)
    print("📊 Purpose: Clean all data from Azure services while preserving infrastructure")
    print("☁️  Azure Services: Blob Storage, Cognitive Search, Cosmos DB")
    print("⏱️  Enterprise Operation: Automated data lifecycle management")
    print()

    workflow_start = time.time()
    domain = "general"  # From your existing data state patterns

    try:
        # Initialize Azure services using existing patterns
        print("📝 Initializing Azure services manager...")
        azure_services = AzureServicesManager()

        # Validate services before cleanup
        print("🔍 Validating Azure services configuration...")
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            print(f"❌ Azure services validation failed: {validation}")
            return 1

        print("✅ Azure services validated successfully")

        # Execute automated cleanup
        print(f"\n🧹 Step 1: Executing automated data cleanup for domain '{domain}'...")
        cleanup_result = await azure_services.cleanup_all_azure_data(domain)

        # Report cleanup results
        print(f"\n📊 Cleanup Results:")
        for service_name, service_result in cleanup_result["cleanup_results"].items():
            if service_result["success"]:
                status = "✅"
                details = ""

                # Service-specific cleanup metrics
                if service_name == "blob_storage":
                    details = f"({service_result['blobs_deleted']} blobs deleted)"
                elif service_name == "cognitive_search":
                    details = f"({service_result['documents_deleted']} documents deleted)"
                elif service_name == "cosmos_db":
                    details = f"({service_result['total_entities_deleted']} entities deleted)"

                print(f"   {status} {service_name.replace('_', ' ').title()}: {details}")
            else:
                print(f"   ❌ {service_name.replace('_', ' ').title()}: {service_result.get('error', 'Unknown error')}")

        # Enterprise metrics
        metrics = cleanup_result["enterprise_metrics"]
        workflow_time = time.time() - workflow_start

        print(f"\n📈 Enterprise Metrics:")
        print(f"   🔹 Services cleaned: {metrics['services_cleaned']}/{metrics['total_services']}")
        print(f"   🔹 Cleanup efficiency: {metrics['cleanup_efficiency']*100:.1f}%")
        print(f"   🔹 Total cleanup time: {workflow_time:.2f}s")
        print(f"   🔹 Azure infrastructure: Preserved")

        if cleanup_result["success"]:
            print(f"\n✅ Azure data cleanup completed successfully!")
            print(f"🏗️ Azure infrastructure preserved and ready for new data")
            return 0
        else:
            print(f"\n⚠️ Partial cleanup completed - some services had issues")
            return 1

    except Exception as e:
        print(f"\n❌ Azure data cleanup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)