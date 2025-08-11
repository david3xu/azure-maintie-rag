#!/usr/bin/env python3
"""
Simple Azure Storage - CODING_STANDARDS Compliant
Clean storage upload script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.azure_storage.storage_client import SimpleStorageClient


async def upload_to_storage(source_path: str, container: str = "raw-data"):
    """Azure Blob Storage upload with real storage client"""
    print(f"📦 Azure Storage Upload: '{source_path}' → {container}")

    try:
        # Initialize storage client
        try:
            storage_client = SimpleStorageClient()
            await storage_client.async_initialize()
            print(f"✅ Connected to Azure Blob Storage")
            storage_available = True
        except Exception as storage_error:
            # NO FALLBACKS - Azure Storage required for production
            print(f"❌ Azure Storage connection failed: {storage_error}")
            raise Exception(f"Azure Storage is required for production data uploads: {storage_error}")

        # Find files to upload
        source_dir = Path(source_path)
        if not source_dir.exists():
            print(f"❌ Source path not found: {source_path}")
            return False

        # Find files
        files = list(source_dir.glob("**/*.md"))

        if not files:
            print(f"❌ No .md files found in {source_path}")
            return False

        print(f"📂 Found {len(files)} files to upload")

        # Upload files to Azure Blob Storage
        uploaded = 0
        failed = 0

        for file_path in files[:5]:  # Upload first 5 files
            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                blob_name = f"raw/{file_path.name}"
                print(f"📤 Uploading: {file_path.name} → {blob_name}")

                if storage_available:
                    try:
                        # Real Azure Blob Storage upload
                        upload_result = await storage_client.upload_text_content(
                            content=content, blob_name=blob_name, container=container
                        )
                        uploaded += 1
                        print(f"   ✅ Upload successful")

                    except Exception as upload_error:
                        print(f"   ❌ Azure upload failed: {upload_error}")
                        # NO FALLBACKS - Azure Storage required for production
                        raise upload_error
                else:
                    # NO SIMULATIONS - Azure Storage required for production
                    raise Exception("Azure Storage client is required for production uploads")

            except Exception as e:
                print(f"⚠️ Failed to process {file_path.name}: {e}")
                failed += 1

        print(f"✅ Uploaded {uploaded}/{len(files)} files to container '{container}'")
        if failed > 0:
            print(f"⚠️  Failed: {failed} files")
        return uploaded > 0

    except Exception as e:
        print(f"❌ Storage upload failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Azure storage upload")
    parser.add_argument("--source", required=True, help="Source directory path")
    parser.add_argument("--container", default="data", help="Storage container name")
    args = parser.parse_args()

    result = asyncio.run(upload_to_storage(args.source, args.container))
    sys.exit(0 if result else 1)
