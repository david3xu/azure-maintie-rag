#!/usr/bin/env python3
"""Test script to verify storage factory integration in Azure services manager"""

import sys
from pathlib import Path
import pytest

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def test_storage_integration():
    """Test the storage factory integration in Azure services manager"""
    from integrations.azure_services import AzureServicesManager

    # Create Azure services manager
    azure_services = AzureServicesManager()

    # Test storage factory integration
    storage_factory = azure_services.get_storage_factory()
    assert storage_factory is not None, "Storage factory is not available."

    # Test individual storage clients
    rag_storage = azure_services.get_rag_storage_client()
    ml_storage = azure_services.get_ml_storage_client()
    app_storage = azure_services.get_app_storage_client()

    assert rag_storage is not None, "RAG storage client is not available."
    assert ml_storage is not None, "ML storage client is not available."
    assert app_storage is not None, "App storage client is not available."

    # Test storage status
    status = storage_factory.get_storage_status()
    assert status is not None, "Storage status could not be retrieved."

    # Test available clients
    clients = storage_factory.list_available_clients()
    assert clients is not None and len(clients) > 0, "No clients available from storage factory."