#!/usr/bin/env python3
"""
Debug Actual Storage State - Check what's really in Azure Blob Storage
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from core.azure_storage.storage_client import UnifiedStorageClient
from config.settings import azure_settings

async def debug_actual_storage():
    """Debug what's actually in Azure Blob Storage"""
    print("🔍 Actual Azure Blob Storage Debug")
    print("=" * 50)
    
    print(f"📋 Configuration:")
    print(f"   Storage Account: {azure_settings.azure_storage_account}")
    print(f"   Container: {azure_settings.azure_blob_container}")
    print(f"   Use Managed Identity: {azure_settings.use_managed_identity}")
    
    try:
        # Test with the maintenance-data container that showed errors
        print(f"\n🧪 Testing container: maintenance-data")
        storage_client = UnifiedStorageClient()
        
        # Test list_blobs on maintenance-data (the container with permission errors)
        result1 = await storage_client.list_blobs("maintenance-data")
        print(f"📦 maintenance-data container:")
        if result1.get('success'):
            print(f"   ✅ Success: {result1['data']['blob_count']} blobs")
            for blob in result1['data']['blobs'][:5]:  # Show first 5
                print(f"   - {blob['name']} ({blob['size']} bytes)")
        else:
            print(f"   ❌ Failed: {result1.get('error')}")
        
        # Test list_blobs on default container (maintie-staging-data)
        print(f"\n🧪 Testing container: {azure_settings.azure_blob_container}")
        result2 = await storage_client.list_blobs()  # Use default container
        print(f"📦 {azure_settings.azure_blob_container} container:")
        if result2.get('success'):
            print(f"   ✅ Success: {result2['data']['blob_count']} blobs")
            for blob in result2['data']['blobs'][:5]:  # Show first 5
                print(f"   - {blob['name']} ({blob['size']} bytes)")
        else:
            print(f"   ❌ Failed: {result2.get('error')}")
            
        # Try to upload a test file to see what happens
        print(f"\n🧪 Testing actual upload to {azure_settings.azure_blob_container}")
        test_data = "Test data from debug script"
        upload_result = await storage_client.upload_data(test_data, "debug_test.txt")
        print(f"📤 Upload test:")
        if upload_result.get('success'):
            print(f"   ✅ Upload reported success")
            print(f"   Blob URL: {upload_result['data'].get('blob_url', 'not provided')}")
        else:
            print(f"   ❌ Upload failed: {upload_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_actual_storage())