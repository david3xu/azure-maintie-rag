#!/usr/bin/env python3
"""
Test Storage Client - Isolate storage client issues
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from core.azure_storage.storage_client import UnifiedStorageClient
from config.settings import azure_settings

async def test_storage_client():
    """Test storage client initialization and basic operations"""
    print("🧪 Testing Storage Client Initialization")
    print("=" * 50)
    
    try:
        # Initialize client
        print("📦 Initializing UnifiedStorageClient...")
        storage_client = UnifiedStorageClient()
        
        # Check configuration
        print(f"📋 Configuration:")
        print(f"   Endpoint: {storage_client.endpoint}")
        print(f"   Use Managed Identity: {storage_client.use_managed_identity}")
        print(f"   Default Container: {storage_client.default_container}")
        
        # Test basic operation
        print(f"\n🔍 Testing list_blobs operation...")
        
        # Test with maintenance-data container (as seen in the error)
        result = await storage_client.list_blobs("maintenance-data")
        
        if result.get('success'):
            print(f"✅ List blobs successful:")
            print(f"   Container: {result['data']['container']}")
            print(f"   Blob count: {result['data']['blob_count']}")
            for blob in result['data']['blobs'][:3]:  # Show first 3
                print(f"   - {blob['name']} ({blob['size']} bytes)")
        else:
            print(f"❌ List blobs failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Storage client test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_storage_client())