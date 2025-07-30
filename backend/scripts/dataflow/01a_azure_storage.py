#!/usr/bin/env python3
"""
Azure Blob Storage Test - Step 01a
Raw Text Data → Azure Blob Storage ONLY

This script tests ONLY Azure Blob Storage upload functionality:
- Uses real Azure Storage client with managed identity
- Uploads files from data/raw to Azure containers
- Provides detailed logging and verification
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from config.domain_patterns import DomainPatternManager
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureStorageTestStage:
    """Step 01a: Raw Text Data → Azure Blob Storage ONLY"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        
    async def execute(self, source_path: str, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Execute Azure Blob Storage upload test
        
        Args:
            source_path: Path to raw text files
            domain: Domain for processing
            
        Returns:
            Dict with storage upload results
        """
        print("📦 Step 01a: Azure Blob Storage Test")
        print("=" * 40)
        
        start_time = asyncio.get_event_loop().time()
        
        results = {
            "stage": "01a_azure_storage",
            "source_path": str(source_path),
            "domain": domain,
            "uploaded_files": 0,
            "container_name": "",
            "success": False
        }
        
        try:
            storage_client = self.infrastructure.storage_client
            if not storage_client:
                raise RuntimeError("❌ Azure Storage client not initialized")
            
            # Test connectivity
            print("🔗 Testing Azure Storage connectivity...")
            containers = await storage_client.list_containers()
            print(f"✅ Storage connectivity verified - found {len(containers)} containers")
            
            # Get container name
            container_name = DomainPatternManager.get_container_name(domain, azure_settings.azure_storage_container)
            results["container_name"] = container_name
            
            # Ensure container exists
            print(f"📦 Creating/verifying container: {container_name}")
            await storage_client._ensure_container_exists(container_name)
            
            # Upload files
            source_path_obj = Path(source_path)
            if not source_path_obj.exists():
                raise FileNotFoundError(f"Source path not found: {source_path}")
            
            uploaded_files = []
            failed_uploads = []
            
            if source_path_obj.is_file():
                files_to_upload = [source_path_obj]
            else:
                files_to_upload = list(source_path_obj.rglob("*.md"))
            
            print(f"📤 Uploading {len(files_to_upload)} files...")
            
            for file_path in files_to_upload:
                try:
                    relative_path = file_path.relative_to(source_path_obj) if source_path_obj.is_dir() else file_path.name
                    blob_name = f"{domain}/{relative_path}"
                    
                    print(f"📤 Uploading: {file_path.name} → {blob_name}")
                    upload_result = await storage_client.upload_file(
                        str(file_path), blob_name, container_name
                    )
                    
                    if upload_result.get('success'):
                        uploaded_files.append({
                            "file_path": str(file_path),
                            "blob_name": blob_name,
                            "size": file_path.stat().st_size,
                            "blob_url": upload_result.get('data', {}).get('blob_url', '')
                        })
                        print(f"✅ Upload successful: {blob_name}")
                    else:
                        failed_uploads.append(str(file_path))
                        print(f"❌ Upload failed: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    failed_uploads.append(str(file_path))
                    print(f"❌ Upload error: {file_path} - {e}")
            
            # Results
            duration = asyncio.get_event_loop().time() - start_time
            results.update({
                "uploaded_files": len(uploaded_files),
                "failed_uploads": len(failed_uploads),
                "duration_seconds": round(duration, 2),
                "success": len(failed_uploads) == 0,
                "details": uploaded_files
            })
            
            print(f"\n✅ Step 01a Complete:")
            print(f"   📦 Container: {container_name}")
            print(f"   📤 Files uploaded: {results['uploaded_files']}")
            print(f"   ❌ Failed uploads: {results['failed_uploads']}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")
            
            if results['success']:
                print(f"🎉 All uploads successful!")
            else:
                print(f"⚠️  Some uploads failed")
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Step 01a Failed: {e}")
            logger.error(f"Azure Storage test failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for Azure Storage test"""
    parser = argparse.ArgumentParser(
        description="Step 01a: Azure Blob Storage Test"
    )
    parser.add_argument(
        "--source", 
        required=True,
        help="Path to raw text files"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Domain for processing"
    )
    
    args = parser.parse_args()
    
    # Execute stage
    stage = AzureStorageTestStage()
    results = await stage.execute(
        source_path=args.source,
        domain=args.domain
    )
    
    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))