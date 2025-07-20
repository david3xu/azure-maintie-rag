#!/usr/bin/env python3
"""Test script to verify storage factory integration in Azure services manager"""

import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_storage_integration():
    """Test the storage factory integration in Azure services manager"""
    try:
        from integrations.azure_services import AzureServicesManager

        # Create Azure services manager
        azure_services = AzureServicesManager()

        print("✅ Azure Services Manager Test Results:")

        # Test storage factory integration
        storage_factory = azure_services.get_storage_factory()
        print(f"Storage factory available: {storage_factory is not None}")

        # Test individual storage clients
        rag_storage = azure_services.get_rag_storage_client()
        ml_storage = azure_services.get_ml_storage_client()
        app_storage = azure_services.get_app_storage_client()

        print(f"RAG storage client: {rag_storage is not None}")
        print(f"ML storage client: {ml_storage is not None}")
        print(f"App storage client: {app_storage is not None}")

        # Test storage status
        status = storage_factory.get_storage_status()
        print(f"Storage status: {status}")

        # Test available clients
        clients = storage_factory.list_available_clients()
        print(f"Available clients: {clients}")

        return True

    except Exception as e:
        print(f"❌ Storage Integration Test Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_storage_integration()
    if success:
        print("\n🎉 Storage factory integration is working correctly!")
    else:
        print("\n💥 Storage factory integration needs configuration or dependencies")