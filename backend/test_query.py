#!/usr/bin/env python3
"""
Test script to isolate the query processing issue
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestration.enhanced_rag_universal import get_enhanced_rag_instance

async def test_query():
    """Test the query processing"""
    try:
        print("Testing Enhanced Universal RAG query processing...")

        # Get the enhanced RAG instance
        print("Getting enhanced RAG instance...")
        enhanced_rag = get_enhanced_rag_instance("general")
        print(f"Enhanced RAG instance: {enhanced_rag}")
        print(f"Components initialized: {enhanced_rag.components_initialized}")

        # Test initialization if needed
        if not enhanced_rag.components_initialized:
            print("Initializing components...")
            init_results = await enhanced_rag.initialize_components()
            print(f"Initialization results: {init_results}")

            if not init_results.get("success", False):
                print(f"Initialization failed: {init_results.get('error', 'Unknown error')}")
                return

        # Test query processing
        print("Processing test query...")
        results = await enhanced_rag.process_query(
            query="what are common issues?",
            max_results=10,
            include_explanations=True,
            enable_safety_warnings=True
        )

        print(f"Query results: {results}")

        if results.get("success", False):
            print("✅ Query processing successful!")
        else:
            print(f"❌ Query processing failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_query())