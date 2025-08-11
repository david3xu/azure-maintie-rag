#!/usr/bin/env python3
"""
Simple Azure Search - CODING_STANDARDS Compliant
Clean search indexing script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps


async def index_in_search(source_path: str, domain: str = "universal"):
    """Azure Cognitive Search indexing with real search client"""
    print(f"🔍 Azure Search Indexing: '{source_path}' (domain: {domain})")

    try:
        # Initialize universal dependencies
        deps = await get_universal_deps()
        search_client = deps.search_client
        openai_client = deps.openai_client
        print("✅ Azure Search and OpenAI clients ready")

        if not search_client or not openai_client:
            # NO SIMULATIONS - Azure services MUST work for production
            raise Exception("Azure Search or OpenAI client not available - cannot proceed")

        # Find files to index
        source_dir = Path(source_path)
        if not source_dir.exists():
            print(f"❌ Source path not found: {source_path}")
            return False

        # Find files
        files = list(source_dir.glob("**/*.md"))

        if not files:
            print(f"❌ No .md files found in {source_path}")
            return False

        print(f"📂 Found {len(files)} files to index")

        # Index all files
        indexed = 0
        for file_path in files:  # Index all available files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                doc_id = f"{domain}-{file_path.stem}"
                print(f"📄 Indexing: {file_path.name} → {doc_id}")

                # Simple document structure
                document = {
                    "id": doc_id,
                    "content": content,
                    "title": f"{domain.title()} Document: {file_path.stem}",
                    "domain": domain,
                }

                # REAL Azure Search indexing - no simulations
                index_result = await search_client.index_document(document)
                if index_result.get("success"):
                    indexed += 1
                    print(f"   ✅ Indexed: {file_path.name}")
                else:
                    raise Exception(f"Search indexing failed: {index_result.get('error')}")

            except Exception as e:
                print(f"⚠️ Indexing failed for {file_path.name}: {e}")

        print(f"✅ Indexed {indexed}/{len(files)} documents in search index")
        return indexed > 0

    except Exception as e:
        print(f"❌ Search indexing failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Azure search indexing")
    parser.add_argument("--source", required=True, help="Source directory path")
    parser.add_argument(
        "--domain", default="discovered_content", help="Domain for processing"
    )
    args = parser.parse_args()

    result = asyncio.run(index_in_search(args.source, args.domain))
    sys.exit(0 if result else 1)
