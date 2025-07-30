#!/usr/bin/env python3
"""
Debug Cosmos Gremlin Client - Check authentication format
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import azure_settings

def debug_cosmos_settings():
    """Debug Cosmos Gremlin settings"""
    print("🔍 Cosmos Gremlin Debug")
    print("=" * 40)
    
    print(f"Cosmos Endpoint: {repr(azure_settings.azure_cosmos_endpoint)}")
    print(f"Database Name: {repr(azure_settings.cosmos_database_name)}")
    print(f"Graph Name: {repr(azure_settings.cosmos_graph_name)}")
    print(f"Container Name (azure_cosmos_container): {repr(azure_settings.azure_cosmos_container)}")
    print(f"Container Name (cosmos_graph_name): {repr(azure_settings.cosmos_graph_name)}")
    print(f"Use Managed Identity: {azure_settings.use_managed_identity}")
    print(f"Cosmos Use Managed Identity: {getattr(azure_settings, 'cosmos_use_managed_identity', 'not found')}")
    
    # Test username format construction
    database = azure_settings.cosmos_database_name
    container = azure_settings.cosmos_graph_name
    expected_username = f"/dbs/{database}/colls/{container}"
    print(f"\nExpected Username Format: {repr(expected_username)}")
    print(f"Username components valid: {bool(database and container)}")

if __name__ == "__main__":
    debug_cosmos_settings()