#!/usr/bin/env python3
"""
Simple Data Ingestion - CODING_STANDARDS Compliant
Clean data ingestion script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps


async def ingest_data(source_path: str, container_name: str = "raw-data"):
    """Simple data ingestion to Azure Blob Storage"""
    print(f"📁 Data Ingestion: '{source_path}' → Azure Blob Storage")

    try:
        # Initialize universal dependencies
        deps = await get_universal_deps()
        storage_client = deps.storage_client
        print(f"✅ Connected to Azure Blob Storage")

        # Find files to process
        source_directory = Path(source_path)
        if not source_directory.exists():
            print(f"❌ Source path not found: {source_path}")
            return None

        # Find markdown files
        md_files = list(source_directory.glob("**/*.md"))

        if not md_files:
            print(f"❌ No .md files found in {source_path}")
            return None

        print(f"📂 Found {len(md_files)} files to process")

        # Process all files
        processed = 0
        total_size = 0

        for file_path in md_files:  # Process all available files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                file_size = len(content.encode("utf-8"))
                total_size += file_size

                # Upload to Azure Blob Storage
                blob_name = file_path.name  # Just the file name, container is set in client

                try:
                    # Use storage client to upload (correct signature)
                    data_bytes = content.encode('utf-8')
                    upload_result = await storage_client.upload_blob(
                        blob_name=blob_name, data=data_bytes
                    )
                    print(
                        f"📝 Uploaded: {file_path.name} ({file_size} bytes) → {blob_name}"
                    )
                    processed += 1

                except Exception as upload_error:
                    # NO FALLBACKS - Azure storage MUST work for production
                    print(f"❌ Upload failed: {file_path.name}")
                    print(f"   Azure storage error: {upload_error}")
                    raise upload_error

            except Exception as e:
                print(f"⚠️ Failed to process {file_path.name}: {e}")

        print(
            f"✅ Processed {processed}/{len(md_files)} files ({total_size/1024:.1f} KB)"
        )
        return {"processed": processed, "total": len(md_files), "size": total_size}

    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure Universal RAG - Data Ingestion")
    parser.add_argument(
        "--source",
        default="/workspace/azure-maintie-rag/data/raw",
        help="Source directory path",
    )
    parser.add_argument(
        "--container", default="raw-data", help="Azure Blob Storage container name"
    )
    args = parser.parse_args()

    print("📁 Azure Universal RAG - Data Ingestion")
    print("=" * 50)
    print(f"Source: {args.source}")
    print(f"Container: {args.container}")
    print("")

    result = asyncio.run(ingest_data(args.source, args.container))

    if result:
        print(f"\n✅ SUCCESS: Ingested {result['processed']} files")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Data ingestion encountered issues")
        sys.exit(1)
