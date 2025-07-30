#!/usr/bin/env python3
"""Test Azure clients when completely disconnected from Azure"""

import sys
import os
import asyncio
import subprocess
from unittest.mock import patch

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def logout_azure():
    """Logout from Azure CLI to simulate no credentials"""
    try:
        result = subprocess.run(['az', 'logout'], capture_output=True, text=True)
        print(f"Azure logout: {result.returncode == 0}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to logout from Azure: {e}")
        return False

def login_azure():
    """Re-login to Azure (will be interactive, skip for now)"""
    print("Note: Manual re-login required after test")
    return True

async def test_clients_without_credentials():
    """Test all clients after Azure logout"""
    print("Testing clients after Azure logout...")
    
    # Clear all possible credential sources
    credential_env_vars = {
        'AZURE_CLIENT_ID': '',
        'AZURE_CLIENT_SECRET': '',
        'AZURE_TENANT_ID': '',
        'AZURE_USERNAME': '',
        'AZURE_PASSWORD': '',
        'MSI_ENDPOINT': '',
        'MSI_SECRET': '',
        'AZURE_CLIENT_CERTIFICATE_PATH': '',
        'AZURE_AUTHORITY_HOST': '',
        'AZURE_CLOUD': ''
    }
    
    with patch.dict(os.environ, credential_env_vars, clear=False):
        # Test storage client
        try:
            from core.azure_storage.storage_client import UnifiedStorageClient
            client = UnifiedStorageClient()
            result = await client.test_connection()
            if result.get('success', False):
                print(f"❌ Storage client connected without credentials")
                return False
            else:
                print(f"✅ Storage client properly failed: {result.get('error', 'Authentication failed')[:100]}...")
        except Exception as e:
            print(f"✅ Storage client properly failed: {str(e)[:100]}...")
        
        # Test search client
        try:
            from core.azure_search.search_client import UnifiedSearchClient
            client = UnifiedSearchClient()
            result = await client.test_connection()
            if result.get('success', False):
                print(f"❌ Search client connected without credentials")
                return False
            else:
                print(f"✅ Search client properly failed: {result.get('error', 'Authentication failed')[:100]}...")
        except Exception as e:
            print(f"✅ Search client properly failed: {str(e)[:100]}...")
        
        # Test cosmos client
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            client = AzureCosmosGremlinClient()
            result = await client.test_connection()
            if result.get('success', False):
                print(f"❌ Cosmos client connected without credentials")
                return False
            else:
                print(f"✅ Cosmos client properly failed: {result.get('error', 'Authentication failed')[:100]}...")
        except Exception as e:
            print(f"✅ Cosmos client properly failed: {str(e)[:100]}...")
        
        return True

async def main():
    print("Testing Azure clients with NO credentials available...")
    print("=" * 70)
    
    # First, logout from Azure
    if not logout_azure():
        print("Failed to logout from Azure, continuing with environment variable clearing...")
    
    try:
        # Test clients
        success = await test_clients_without_credentials()
        
        if success:
            print("\n✅ All clients properly enforce Azure authentication requirements")
            print("✅ Azure-only authentication modifications are working correctly")
            return 0
        else:
            print("\n❌ Some clients connected without proper Azure credentials")
            return 1
    finally:
        print("\nNote: You may need to run 'az login' to restore Azure access")

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)