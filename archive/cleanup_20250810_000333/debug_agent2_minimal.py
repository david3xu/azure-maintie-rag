#!/usr/bin/env python3
"""
Minimal Agent 2 Debug Script
=============================

Deep debugging of Agent 2 Knowledge Extraction to find why it extracts 0 entities/relationships.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.knowledge_extraction.agent import run_knowledge_extraction


async def debug_agent2_minimal():
    """Debug Agent 2 with minimal input to isolate the issue."""

    print("🔍 AGENT 2 MINIMAL DEBUG")
    print("=" * 50)

    # Test 1: Minimal content with obvious entities
    print("\n1️⃣ Test with minimal obvious content:")
    minimal_content = "Azure Cosmos DB is a database service. Microsoft created Azure."
    print(f"   Content: {minimal_content}")

    try:
        result = await run_knowledge_extraction(
            minimal_content, use_domain_analysis=False  # Skip Agent 1 first
        )
        print(
            f"   ✅ Result: {len(result.entities)} entities, {len(result.relationships)} relationships"
        )
        print(f"   📊 Confidence: {result.extraction_confidence}")
        print(f"   📝 Signature: {result.processing_signature}")

        if result.entities:
            for i, entity in enumerate(result.entities):
                print(
                    f"      Entity {i+1}: {entity.text} ({entity.entity_type}) - {entity.confidence:.2f}"
                )

        if result.relationships:
            for i, rel in enumerate(result.relationships):
                print(
                    f"      Relationship {i+1}: {rel.source} -[{rel.relation}]-> {rel.target} - {rel.confidence:.2f}"
                )

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: With Agent 1 (domain analysis)
    print("\n2️⃣ Test with Agent 1 domain analysis:")
    try:
        result = await run_knowledge_extraction(
            minimal_content, use_domain_analysis=True  # Enable Agent 1
        )
        print(
            f"   ✅ Result: {len(result.entities)} entities, {len(result.relationships)} relationships"
        )
        print(f"   📊 Confidence: {result.extraction_confidence}")
        print(f"   📝 Signature: {result.processing_signature}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Check what tool is being called
    print("\n3️⃣ Test direct tool call:")
    try:
        from agents.core.universal_deps import get_universal_deps
        from agents.knowledge_extraction.agent import knowledge_extraction_agent

        deps = await get_universal_deps()

        # Direct agent call with explicit tool request
        result = await knowledge_extraction_agent.run(
            "Use the extract_entities_and_relationships tool to extract from: Azure Cosmos DB is a database service.",
            deps=deps,
        )

        print(
            f"   ✅ Direct tool call result: {len(result.output.entities)} entities, {len(result.output.relationships)} relationships"
        )

    except Exception as e:
        print(f"   ❌ Direct tool call error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_agent2_minimal())
