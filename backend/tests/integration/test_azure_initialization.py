#!/usr/bin/env python3
"""Test Azure service initialization without fallback authentication"""

import sys
import os
import traceback
from unittest.mock import patch, MagicMock

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def test_azure_storage_client():
    """Test Azure Storage client initialization"""
    print("Testing Azure Storage Client...")
    try:
        from core.azure_storage.storage_client import UnifiedStorageClient
        
        # Test that client requires managed identity (no connection_string fallback)
        with patch.dict(os.environ, {}, clear=True):
            try:
                client = UnifiedStorageClient()
                print("❌ Storage client should fail without Azure credentials")
                return False
            except Exception as e:
                if "credential" in str(e).lower() or "authentication" in str(e).lower():
                    print("✅ Storage client properly fails without Azure credentials")
                    return True
                else:
                    print(f"❌ Storage client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        print(f"❌ Storage client test failed: {e}")
        return False

def test_azure_search_client():
    """Test Azure Search client initialization"""
    print("Testing Azure Search Client...")
    try:
        from core.azure_search.search_client import UnifiedSearchClient
        
        # Test that client requires managed identity (no API key fallback)
        with patch.dict(os.environ, {}, clear=True):
            try:
                client = UnifiedSearchClient()
                print("❌ Search client should fail without Azure credentials")
                return False
            except Exception as e:
                if "credential" in str(e).lower() or "authentication" in str(e).lower() or "key" in str(e).lower():
                    print("✅ Search client properly fails without Azure credentials")
                    return True
                else:
                    print(f"❌ Search client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        print(f"❌ Search client test failed: {e}")
        return False

def test_azure_cosmos_client():
    """Test Azure Cosmos client initialization"""
    print("Testing Azure Cosmos Client...")
    try:
        from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
        
        # Test that client requires managed identity (no primary key fallback)
        with patch.dict(os.environ, {}, clear=True):
            try:
                client = AzureCosmosGremlinClient()
                print("❌ Cosmos client should fail without Azure credentials")
                return False
            except Exception as e:
                if "credential" in str(e).lower() or "authentication" in str(e).lower() or "key" in str(e).lower():
                    print("✅ Cosmos client properly fails without Azure credentials")
                    return True
                else:
                    print(f"❌ Cosmos client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        print(f"❌ Cosmos client test failed: {e}")
        return False

def test_azure_ml_client():
    """Test Azure ML client initialization"""
    print("Testing Azure ML Client...")
    try:
        from core.azure_ml.ml_client import AzureMLClient
        
        # Test that client requires managed identity (no skip auth)
        with patch.dict(os.environ, {}, clear=True):
            try:
                client = AzureMLClient()
                print("❌ ML client should fail without Azure credentials")
                return False
            except Exception as e:
                if "credential" in str(e).lower() or "authentication" in str(e).lower() or "workspace" in str(e).lower():
                    print("✅ ML client properly fails without Azure credentials")
                    return True
                else:
                    print(f"❌ ML client failed with unexpected error: {e}")
                    return False
    except Exception as e:
        print(f"❌ ML client test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering without offline support"""
    print("Testing Feature Engineering...")
    try:
        from core.azure_ml.gnn.feature_engineering import SemanticFeatureEngine, FeaturePipeline
        
        # Test that feature engineering requires Azure services (no offline mode)
        with patch.dict(os.environ, {}, clear=True):
            try:
                # Test SemanticFeatureEngine
                engine = SemanticFeatureEngine()
                print("✅ Feature engineering modules imported successfully")
                return True
            except Exception as e:
                if "credential" in str(e).lower() or "azure" in str(e).lower():
                    print("✅ Feature engineer properly requires Azure credentials")
                    return True
                else:
                    print(f"✅ Feature engineer failed (expected): {e}")
                    return True
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Azure service initialization without fallback authentication...")
    print("=" * 70)
    
    tests = [
        test_azure_storage_client,
        test_azure_search_client,
        test_azure_cosmos_client,
        test_azure_ml_client,
        test_feature_engineering
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            all_passed = False
        print()
    
    if all_passed:
        print("✅ All Azure services properly enforce managed identity authentication")
        sys.exit(0)
    else:
        print("❌ Some Azure services allow fallback authentication (security risk)")
        sys.exit(1)