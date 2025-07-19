#!/usr/bin/env python3
"""
Workflow Analysis Script with Azure Services
==========================================

Analyzes Azure service usage patterns in both workflows.
Provides insights into Azure architecture and service responsibilities.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def analyze_azure_services():
    """Analyze Azure service usage patterns"""

    azure_services = {
        "Azure Blob Storage": {
            "data_prep": True,
            "query_runtime": True,
            "purpose": "Both workflows - stores documents + retrieves content"
        },
        "Azure Cognitive Search": {
            "data_prep": True,
            "query_runtime": True,
            "purpose": "Both workflows - builds indices + searches documents"
        },
        "Azure OpenAI": {
            "data_prep": True,
            "query_runtime": True,
            "purpose": "Both workflows - processes documents + generates responses"
        },
        "Azure Cosmos DB": {
            "data_prep": True,
            "query_runtime": True,
            "purpose": "Both workflows - stores metadata + tracks queries"
        },
        "Azure Machine Learning": {
            "data_prep": False,
            "query_runtime": False,
            "purpose": "Future enhancement - custom ML models"
        }
    }

    print("📊 AZURE SERVICES USAGE ANALYSIS")
    print("=" * 80)

    data_prep_count = sum(1 for s in azure_services.values() if s["data_prep"])
    query_runtime_count = sum(1 for s in azure_services.values() if s["query_runtime"])
    both_count = sum(1 for s in azure_services.values() if s["data_prep"] and s["query_runtime"])

    print(f"📋 Summary:")
    print(f"   🔸 Total Azure services analyzed: {len(azure_services)}")
    print(f"   🔸 Data preparation workflow: {data_prep_count} services")
    print(f"   🔸 Query processing workflow: {query_runtime_count} services")
    print(f"   🔸 Used by both workflows: {both_count} services")
    print(f"   🔸 Service utilization: {(data_prep_count + query_runtime_count - both_count) / len(azure_services) * 100:.1f}%")

    print(f"\n📊 Detailed Analysis:")

    print(f"\n🔹 DATA PREPARATION WORKFLOW SERVICES:")
    for service_name, info in azure_services.items():
        if info["data_prep"]:
            shared = "📍 SHARED" if info["query_runtime"] else ""
            print(f"   ✅ {service_name} {shared}")
            print(f"      └─ {info['purpose']}")

    print(f"\n🔸 QUERY PROCESSING WORKFLOW SERVICES:")
    for service_name, info in azure_services.items():
        if info["query_runtime"]:
            shared = "📍 SHARED" if info["data_prep"] else ""
            print(f"   ✅ {service_name} {shared}")
            print(f"      └─ {info['purpose']}")

    print(f"\n🎯 Azure Architecture Benefits:")
    print(f"   ✅ Fully managed cloud services")
    print(f"   ✅ Automatic scaling and high availability")
    print(f"   ✅ Built-in security and compliance")
    print(f"   ✅ Pay-per-use pricing model")
    print(f"   ✅ Global distribution and low latency")
    print(f"   ✅ Integrated monitoring and logging")

    print(f"\n☁️  Azure Service Integration:")
    print(f"   🔹 Azure Blob Storage: Document storage and retrieval")
    print(f"   🔹 Azure Cognitive Search: Semantic search and indexing")
    print(f"   🔹 Azure OpenAI: Natural language processing")
    print(f"   🔹 Azure Cosmos DB: Metadata and query tracking")
    print(f"   🔹 Azure Machine Learning: Future ML model deployment")

    print(f"\n📈 Performance Characteristics:")
    print(f"   ⚡ Data preparation: One-time setup with Azure services")
    print(f"   ⚡ Query processing: Real-time with Azure Cognitive Search")
    print(f"   ⚡ Response generation: Fast with Azure OpenAI")
    print(f"   ⚡ Scalability: Automatic with Azure cloud services")


def analyze_azure_integration_files():
    """Analyze Azure integration file structure"""

    azure_files = {
        "config/azure_settings.py": {
            "purpose": "Azure configuration and environment settings",
            "usage": "Both workflows - provides Azure service credentials"
        },
        "azure/storage_client.py": {
            "purpose": "Azure Blob Storage client operations",
            "usage": "Both workflows - document storage and retrieval"
        },
        "azure/search_client.py": {
            "purpose": "Azure Cognitive Search client operations",
            "usage": "Both workflows - search indexing and querying"
        },
        "azure/cosmos_client.py": {
            "purpose": "Azure Cosmos DB client operations",
            "usage": "Both workflows - metadata storage and tracking"
        },
        "azure/ml_client.py": {
            "purpose": "Azure Machine Learning client operations",
            "usage": "Future enhancement - custom ML model deployment"
        },
        "integrations/azure_services.py": {
            "purpose": "Unified Azure services manager",
            "usage": "Both workflows - coordinates all Azure services"
        },
        "integrations/azure_openai.py": {
            "purpose": "Azure OpenAI integration",
            "usage": "Both workflows - document processing and response generation"
        }
    }

    print(f"\n📁 AZURE INTEGRATION FILES ANALYSIS")
    print("=" * 80)

    print(f"📋 Summary:")
    print(f"   🔸 Total Azure integration files: {len(azure_files)}")
    print(f"   🔸 Configuration files: 1")
    print(f"   🔸 Service client files: 4")
    print(f"   🔸 Integration manager files: 2")

    print(f"\n📊 File Structure:")
    for filepath, info in azure_files.items():
        print(f"   ✅ {filepath}")
        print(f"      └─ {info['purpose']}")
        print(f"      └─ Usage: {info['usage']}")

    print(f"\n🏗️  Infrastructure Files:")
    print(f"   ✅ infrastructure/azure-resources.bicep - Azure resource templates")
    print(f"   ✅ infrastructure/parameters.json - Deployment parameters")
    print(f"   ✅ infrastructure/provision.py - Automated provisioning script")


if __name__ == "__main__":
    """Execute workflow analysis"""
    analyze_azure_services()
    analyze_azure_integration_files()