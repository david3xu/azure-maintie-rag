#!/usr/bin/env python3
"""
Azure Storage Cleanup Script
Clean all blobs from all containers in Azure Storage account while keeping containers.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from config.settings import azure_settings


async def clean_azure_storage():
    """Clean all blobs from Azure Storage account"""
    print("🧹 AZURE STORAGE CLEANUP")
    print("=" * 40)
    
    try:
        # Initialize Azure Storage client
        credential = DefaultAzureCredential()
        storage_account_url = f"https://{azure_settings.azure_storage_account}.blob.core.windows.net"
        
        print(f"🔧 Connecting to: {storage_account_url}")
        blob_service_client = BlobServiceClient(
            account_url=storage_account_url,
            credential=credential
        )
        
        # List all containers
        print("📊 Listing all containers...")
        containers = blob_service_client.list_containers()
        container_list = list(containers)
        
        print(f"📁 Found {len(container_list)} containers:")
        for container in container_list:
            print(f"   - {container.name}")
        
        # Clean each container
        total_blobs_deleted = 0
        
        for container in container_list:
            container_name = container.name
            print(f"\n🧹 Cleaning container: {container_name}")
            
            try:
                container_client = blob_service_client.get_container_client(container_name)
                
                # List all blobs in container
                blobs = container_client.list_blobs()
                blob_list = list(blobs)
                
                if blob_list:
                    print(f"   🗑️  Found {len(blob_list)} blobs to delete")
                    
                    # Delete each blob
                    for blob in blob_list:
                        try:
                            blob_client = container_client.get_blob_client(blob.name)
                            blob_client.delete_blob()
                            total_blobs_deleted += 1
                            print(f"     ✅ Deleted: {blob.name}")
                        except Exception as e:
                            print(f"     ❌ Failed to delete {blob.name}: {e}")
                    
                    print(f"   ✅ Container {container_name}: {len(blob_list)} blobs deleted")
                else:
                    print(f"   ✅ Container {container_name}: Already clean (0 blobs)")
                    
            except Exception as e:
                print(f"   ❌ Failed to clean container {container_name}: {e}")
        
        print(f"\n🎉 STORAGE CLEANUP COMPLETE!")
        print(f"📊 Summary:")
        print(f"   - Containers processed: {len(container_list)}")
        print(f"   - Total blobs deleted: {total_blobs_deleted}")
        print(f"   - Containers preserved: {len(container_list)} (structure intact)")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(clean_azure_storage())
    if result:
        print("\n✅ Azure Storage cleanup completed successfully")
    else:
        print("\n❌ Azure Storage cleanup failed")
        sys.exit(1)