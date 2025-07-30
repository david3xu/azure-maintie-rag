#!/usr/bin/env python3
"""
Debug Azure Settings - Check what values are actually loaded
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import azure_settings

def debug_azure_settings():
    """Debug Azure settings values"""
    print("🔍 Azure Settings Debug")
    print("=" * 40)
    
    print(f"Storage Account: {repr(azure_settings.azure_storage_account)}")
    print(f"Storage Connection String: {repr(azure_settings.azure_storage_connection_string)}")
    print(f"Blob Container: {repr(azure_settings.azure_blob_container)}")
    print(f"Storage Container: {repr(azure_settings.azure_storage_container)}")
    print(f"Use Managed Identity: {azure_settings.use_managed_identity}")
    
    print(f"\nSearch Endpoint: {repr(azure_settings.azure_search_endpoint)}")
    print(f"Search Index: {repr(azure_settings.azure_search_index)}")
    print(f"Search Key: {repr(azure_settings.azure_search_key)}")
    
    print(f"\nCosmos Endpoint: {repr(azure_settings.azure_cosmos_endpoint)}")
    print(f"Cosmos Database: {repr(azure_settings.cosmos_database_name)}")
    print(f"Cosmos Graph: {repr(azure_settings.cosmos_graph_name)}")
    print(f"Cosmos Key: {repr(azure_settings.azure_cosmos_key)}")
    
    print(f"\nML Workspace: {repr(azure_settings.azure_ml_workspace_name)}")
    
    print(f"\nContainer Name Mappings:")
    print(f"  AZURE_BLOB_CONTAINER: {repr(azure_settings.azure_blob_container)}")
    print(f"  AZURE_STORAGE_CONTAINER: {repr(azure_settings.azure_storage_container)}")
    print(f"  STORAGE_CONTAINER_NAME: {repr(getattr(azure_settings, 'storage_container_name', 'not found'))}")

if __name__ == "__main__":
    debug_azure_settings()