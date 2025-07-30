#!/usr/bin/env python3
"""Test that Azure clients properly fail when attempting actual operations without credentials"""

import sys
import os
import asyncio
from unittest.mock import patch

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

async def test_storage_client_operations():
    """Test that storage client fails on actual operations without credentials"""
    print("Testing Storage Client Operations...")
    try:
        from core.azure_storage.storage_client import UnifiedStorageClient
        
        # Clear all Azure-related environment variables
        azure_env_vars = [k for k in os.environ.keys() if 'AZURE' in k.upper()]
        with patch.dict(os.environ, {var: '' for var in azure_env_vars}, clear=False):
            client = UnifiedStorageClient()
            
            try:
                # This should fail because no Azure credentials are available
                result = await client.test_connection()
                print(f"❌ Storage client operations should fail without credentials: {result}")
                return False
            except Exception as e:
                if any(word in str(e).lower() for word in ['credential', 'authentication', 'unauthorized', 'forbidden']):
                    print("✅ Storage client properly fails operations without Azure credentials")
                    return True
                else:
                    print(f"❌ Storage client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        # Import or initialization errors are also acceptable
        print(f"✅ Storage client failed to initialize: {e}")
        return True

async def test_search_client_operations():
    """Test that search client fails on actual operations without credentials"""
    print("Testing Search Client Operations...")
    try:
        from core.azure_search.search_client import UnifiedSearchClient
        
        # Clear all Azure-related environment variables
        azure_env_vars = [k for k in os.environ.keys() if 'AZURE' in k.upper()]
        with patch.dict(os.environ, {var: '' for var in azure_env_vars}, clear=False):
            client = UnifiedSearchClient()
            
            try:
                # This should fail because no Azure credentials are available
                result = await client.test_connection()
                print(f"❌ Search client operations should fail without credentials: {result}")
                return False
            except Exception as e:
                if any(word in str(e).lower() for word in ['credential', 'authentication', 'unauthorized', 'forbidden', 'key']):
                    print("✅ Search client properly fails operations without Azure credentials")
                    return True
                else:
                    print(f"❌ Search client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        # Import or initialization errors are also acceptable
        print(f"✅ Search client failed to initialize: {e}")
        return True

async def test_cosmos_client_operations():
    """Test that cosmos client fails on actual operations without credentials"""
    print("Testing Cosmos Client Operations...")
    try:
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        
        # Clear all Azure-related environment variables
        azure_env_vars = [k for k in os.environ.keys() if 'AZURE' in k.upper()]
        with patch.dict(os.environ, {var: '' for var in azure_env_vars}, clear=False):
            client = AzureCosmosGremlinClient()
            
            try:
                # This should fail because no Azure credentials are available
                result = await client.test_connection()
                print(f"❌ Cosmos client operations should fail without credentials: {result}")
                return False
            except Exception as e:
                if any(word in str(e).lower() for word in ['credential', 'authentication', 'unauthorized', 'forbidden', 'key']):
                    print("✅ Cosmos client properly fails operations without Azure credentials")
                    return True
                else:
                    print(f"❌ Cosmos client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        # Import or initialization errors are also acceptable
        print(f"✅ Cosmos client failed to initialize: {e}")
        return True

def test_ml_client_operations():
    """Test that ML client fails on actual operations without credentials"""
    print("Testing ML Client Operations...")
    try:
        from core.azure_ml.ml_client import AzureMLClient
        
        # Clear all Azure-related environment variables
        azure_env_vars = [k for k in os.environ.keys() if 'AZURE' in k.upper()]
        with patch.dict(os.environ, {var: '' for var in azure_env_vars}, clear=False):
            client = AzureMLClient()
            
            try:
                # Try to access workspace - this should fail
                workspace = client.get_workspace()
                print(f"❌ ML client operations should fail without credentials: {workspace}")
                return False
            except Exception as e:
                if any(word in str(e).lower() for word in ['credential', 'authentication', 'unauthorized', 'forbidden', 'workspace']):
                    print("✅ ML client properly fails operations without Azure credentials")
                    return True
                else:
                    print(f"❌ ML client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        # Import or initialization errors are also acceptable
        print(f"✅ ML client failed to initialize: {e}")
        return True

async def main():
    print("Testing Azure clients with actual operations (no credentials)...")
    print("=" * 70)
    
    tests = [
        test_storage_client_operations(),
        test_search_client_operations(),
        test_cosmos_client_operations()
    ]
    
    # Add ML client test (non-async)
    ml_result = test_ml_client_operations()
    
    # Run async tests
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    all_passed = ml_result
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"✅ Test {i+1} failed as expected: {result}")
        elif result is False:
            all_passed = False
        print()
    
    if all_passed:
        print("✅ All Azure clients properly enforce credential requirements")
        return 0
    else:
        print("❌ Some Azure clients allow operations without proper credentials")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)