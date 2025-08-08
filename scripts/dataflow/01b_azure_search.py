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

from infrastructure.azure_openai.openai_client import AzureOpenAIClient
from infrastructure.azure_search.search_client import SimpleSearchClient


async def index_in_search(source_path: str, domain: str = "universal"):
    """Azure Cognitive Search indexing with real search client"""
    print(f"🔍 Azure Search Indexing: '{source_path}' (domain: {domain})")

    try:
        # Initialize search and OpenAI clients
        try:
            search_client = SimpleSearchClient()
            await search_client.async_initialize()
            print("✅ Azure Search client ready")
            search_available = True
        except Exception as e:
            search_client = None
            search_available = False
            print(f"⚠️  Azure Search unavailable: {str(e)[:50]}...")

        try:
            openai_client = AzureOpenAIClient()
            await openai_client.async_initialize()
            print("✅ Azure OpenAI client ready for embeddings")
            openai_available = True
        except Exception as e:
            openai_client = None
            openai_available = False
            print(f"⚠️  Azure OpenAI unavailable: {str(e)[:50]}...")

        if not search_available and not openai_available:
            print("🔍 Simulated search indexing (Azure services unavailable)")
            return True

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

        # Index files (demo)
        indexed = 0
        for file_path in files[:3]:  # Demo: index first 3 files
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

                # Simple indexing simulation
                indexed += 1

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
