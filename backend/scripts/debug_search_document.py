#!/usr/bin/env python3
"""
Debug script to test Azure Search document structure
"""

import asyncio
import json
import sys
import os

# Add the current directory to the Python path
sys.path.append('.')

from core.azure_search.search_client import AzureCognitiveSearchClient

async def test_document_structure():
    """Test what document structure is being sent to Azure Search"""

    # Initialize search client
    search_client = AzureCognitiveSearchClient()

    # Create a test document (similar to what's in data_preparation_workflow.py)
    test_document = {
        "id": "test_chunk_1_maintenance_all_texts_1",
        "content": "This is a test maintenance record about air conditioner thermostat not working",
        "title": "maintenance_all_texts.md - Part 1",
        "domain": "maintenance",
        "source": "data/raw/maintenance_all_texts.md",
        "metadata": json.dumps({
            "original_filename": "maintenance_all_texts.md",
            "chunk_index": 1,
            "chunk_type": "text_segment",
            "total_chunks": 122,
            "processing_method": "intelligent_chunking",
            "file_size": 215000,
            "processing_timestamp": "2025-07-27T00:53:51.694",
            "content_type": "text/markdown",
            "azure_openai_processed": True,
            "intelligent_chunking": True
        })
    }

    print("🔍 Testing document structure:")
    print(f"Document ID: {test_document['id']}")
    print(f"Document keys: {list(test_document.keys())}")
    print(f"Metadata keys: {list(json.loads(test_document['metadata']).keys())}")

    # Check if any direct fields contain chunk_type
    for key, value in test_document.items():
        if isinstance(value, str) and 'chunk_type' in value:
            print(f"⚠️  Found 'chunk_type' in field '{key}': {value}")

    print("\n✅ Document structure looks correct - no direct chunk_type field")

    # Try to index the document
    try:
        result = await search_client.index_document("rag-index-general", test_document)
        print(f"\n📊 Index result: {result}")
    except Exception as e:
        print(f"\n❌ Index error: {e}")

if __name__ == "__main__":
    asyncio.run(test_document_structure())
