#!/usr/bin/env python3
"""
Azure Cognitive Search Test - Step 01b
Text → Vector Embeddings → Azure Search Index ONLY

This script tests ONLY Azure Cognitive Search indexing functionality:
- Uses real Azure Search client with managed identity
- Creates search index and indexes documents
- Provides detailed logging and verification
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.async_pattern_manager import get_pattern_manager
from config.discovery_infrastructure_naming import get_discovery_naming
from config.dynamic_ml_config import get_dynamic_ml_config
from config.settings import azure_settings
from services.infrastructure_service import InfrastructureService

logger = logging.getLogger(__name__)


class AzureSearchTestStage:
    """Step 01b: Text → Azure Cognitive Search Index ONLY"""

    def __init__(self):
        self.infrastructure = InfrastructureService()

    async def execute(
        self, source_path: str, domain: str = "maintenance"
    ) -> Dict[str, Any]:
        """
        Execute Azure Cognitive Search indexing test

        Args:
            source_path: Path to raw text files
            domain: Domain for processing

        Returns:
            Dict with search indexing results
        """
        print("🔍 Step 01b: Azure Cognitive Search Test")
        print("=" * 45)

        start_time = asyncio.get_event_loop().time()

        results = {
            "stage": "01b_azure_search",
            "source_path": str(source_path),
            "domain": domain,
            "documents_indexed": 0,
            "index_name": "",
            "success": False,
        }

        try:
            search_service = self.infrastructure.search_service
            if not search_service:
                raise RuntimeError("❌ Azure Search service not initialized")

            # Test connectivity
            print("🔗 Testing Azure Search connectivity...")
            indexes = await search_service.list_indexes()
            print(f"✅ Search connectivity verified - found {len(indexes)} indexes")

            # Get index name
            index_name = await (await get_discovery_naming()).get_discovered_index_name(
                domain, azure_settings.azure_search_index
            )
            results["index_name"] = index_name

            # Create index if needed
            print(f"🔍 Creating/verifying index: {index_name}")
            try:
                index_result = await search_service.create_index(index_name, domain)
                if index_result.get("success"):
                    print(f"✅ Index created: {index_name}")
                else:
                    print(f"ℹ️  Index already exists: {index_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"ℹ️  Index already exists: {index_name}")
                else:
                    raise e

            # Process and index documents
            source_path_obj = Path(source_path)
            if not source_path_obj.exists():
                raise FileNotFoundError(f"Source path not found: {source_path}")

            documents_indexed = 0
            failed_documents = []

            if source_path_obj.is_file():
                files_to_process = [source_path_obj]
            else:
                files_to_process = list(source_path_obj.rglob("*.md"))

            print(f"📄 Processing {len(files_to_process)} files...")

            for file_path in files_to_process:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    print(f"📄 Processing: {file_path.name}")

                    # Parse content into documents
                    if "<id>" in content:
                        # Structured format with <id> markers
                        maintenance_items = content.split("<id>")
                        documents = []

                        for i, item in enumerate(
                            maintenance_items[1:], 1
                        ):  # Skip first empty split
                            if item.strip():
                                document = {
                                    "id": f"{domain}-maintenance-{i}",
                                    "content": item.strip(),
                                    "title": f"{domain.title()} Item {i}",
                                    "domain": domain,
                                    "source_file": str(file_path),
                                    "document_type": "maintenance_item",
                                    "timestamp": datetime.now().isoformat(),
                                }
                                documents.append(document)

                        print(f"📄 Found {len(documents)} maintenance items")

                        # Index in batches
                        BATCH_SIZE = 10
                        for i in range(0, len(documents), BATCH_SIZE):
                            batch = documents[i : i + BATCH_SIZE]
                            print(
                                f"📤 Indexing batch {i//BATCH_SIZE + 1}: {len(batch)} documents"
                            )

                            index_result = await search_service.index_documents(
                                batch, index_name
                            )
                            if index_result.get("success"):
                                documents_indexed += len(batch)
                                print(f"✅ Batch indexed successfully")
                            else:
                                failed_documents.extend([f"batch-{i//BATCH_SIZE + 1}"])
                                print(f"❌ Batch indexing failed")
                    else:
                        # Regular document
                        document = {
                            "id": f"{domain}-{file_path.stem}",
                            "content": content,
                            "title": f"{domain.title()} Document: {file_path.stem}",
                            "domain": domain,
                            "source_file": str(file_path),
                            "document_type": "document",
                            "timestamp": datetime.now().isoformat(),
                        }

                        print(f"📤 Indexing document: {file_path.stem}")
                        index_result = await search_service.index_documents(
                            [document], index_name
                        )
                        if index_result.get("success"):
                            documents_indexed += 1
                            print(f"✅ Document indexed successfully")
                        else:
                            failed_documents.append(str(file_path))
                            print(f"❌ Document indexing failed")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    failed_documents.append(str(file_path))
                    print(f"❌ Processing error: {file_path} - {e}")

            # Results
            duration = asyncio.get_event_loop().time() - start_time
            results.update(
                {
                    "documents_indexed": documents_indexed,
                    "failed_documents": len(failed_documents),
                    "duration_seconds": round(duration, 2),
                    "success": len(failed_documents) == 0,
                }
            )

            print(f"\n✅ Step 01b Complete:")
            print(f"   🔍 Index: {index_name}")
            print(f"   📄 Documents indexed: {results['documents_indexed']}")
            print(f"   ❌ Failed documents: {results['failed_documents']}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")

            if results["success"]:
                print(f"🎉 All documents indexed successfully!")
            else:
                print(f"⚠️  Some documents failed to index")

            return results

        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(
                asyncio.get_event_loop().time() - start_time, 2
            )
            print(f"❌ Step 01b Failed: {e}")
            logger.error(f"Azure Search test failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for Azure Search test"""
    parser = argparse.ArgumentParser(
        description="Step 01b: Azure Cognitive Search Test"
    )
    parser.add_argument("--source", required=True, help="Path to raw text files")
    parser.add_argument("--domain", default="maintenance", help="Domain for processing")

    args = parser.parse_args()

    # Execute stage
    stage = AzureSearchTestStage()
    results = await stage.execute(source_path=args.source, domain=args.domain)

    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
