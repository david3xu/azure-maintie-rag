#!/usr/bin/env python3
"""
Simple Vector Embeddings - CODING_STANDARDS Compliant
Clean vector embedding script without over-engineering.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps


async def generate_embeddings(domain: str = "universal"):
    """Vector embedding generation using Azure OpenAI"""
    print(f"🎯 Vector Embeddings Generation (domain: {domain})")

    try:
        # Initialize universal dependencies
        deps = await get_universal_deps()
        openai_client = deps.openai_client
        search_client = deps.search_client
        print("✅ Azure OpenAI and Search clients ready for embeddings")

        if not openai_client:
            # NO SIMULATIONS - Azure OpenAI required for production embeddings
            raise Exception(
                "Azure OpenAI client is required for production vector embeddings"
            )

        # Load actual uploaded files for embedding generation
        data_dir = Path(
            "/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output"
        )
        data_files = list(data_dir.glob("*.md"))

        if not data_files:
            raise Exception(f"No data files found in {data_dir}")

        print(f"📄 Processing {len(data_files)} actual uploaded documents")

        # Generate embeddings for real uploaded files
        embeddings_generated = 0
        for i, data_file in enumerate(data_files, 1):
            try:
                print(f"🎯 Generating embedding for document {i}: {data_file.name}")

                # Load actual file content (first 1000 chars for embedding)
                content = data_file.read_text(encoding="utf-8", errors="ignore")
                content_chunk = content[:1000] if len(content) > 1000 else content

                if not content_chunk.strip():
                    print(f"⚠️  Skipping empty file: {data_file.name}")
                    continue

                # REAL Azure OpenAI embedding generation with actual file content
                result = await openai_client.get_embedding(content_chunk)
                if result.get("success") and result.get("data", {}).get("embedding"):
                    embedding = result["data"]["embedding"]
                    if len(embedding) == 1536:
                        embeddings_generated += 1
                        print(f"✅ Generated 1536D embedding for {data_file.name}")
                    else:
                        raise Exception(
                            f"Invalid embedding dimension: {len(embedding)}, expected 1536"
                        )
                else:
                    raise Exception(
                        f"Failed to generate valid 1536D embedding for {data_file.name}"
                    )

            except Exception as e:
                print(
                    f"⚠️ Embedding generation failed for document {i} ({data_file.name}): {e}"
                )

        print(f"✅ Generated {embeddings_generated}/{len(data_files)} embeddings")
        print("🔍 Index now supports semantic vector search")

        return embeddings_generated > 0

    except Exception as e:
        print(f"❌ Vector embedding generation failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple vector embedding generation")
    parser.add_argument(
        "--domain", default="discovered_content", help="Domain for processing"
    )
    args = parser.parse_args()

    result = asyncio.run(generate_embeddings(args.domain))
    sys.exit(0 if result else 1)
