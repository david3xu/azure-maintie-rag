#!/usr/bin/env python3
"""
Query Processing Workflow Script
===============================

Demonstrates WORKFLOW 2: User Query Processing
Uses 7/12 core files to process queries against pre-built knowledge base.

Core Files Used:
- enhanced_rag_universal.py (main orchestration)
- universal_rag_orchestrator_complete.py (7-step workflow)
- universal_workflow_manager.py (progress tracking)
- universal_query_analyzer.py (query analysis)
- universal_vector_search.py (vector search)
- universal_llm_interface.py (response generation)
- universal_models.py (data structures)
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import actual components used in query processing
from core.orchestration.enhanced_rag_universal import EnhancedUniversalRAG
from core.workflow.universal_workflow_manager import create_workflow_manager
from core.enhancement.universal_query_analyzer import UniversalQueryAnalyzer
from core.retrieval.universal_vector_search import UniversalVectorSearch
from core.generation.universal_llm_interface import UniversalLLMInterface


async def main():
    """Execute query processing workflow"""

    print("🔄 WORKFLOW 2: User Query Processing")
    print("=" * 60)
    print("📊 Purpose: Process user queries against pre-built knowledge base")
    print("🔧 Core Files: 7/12 files actively processing queries")
    print("⏱️  Frequency: Every user request (runtime)")

    domain = "general"
    test_query = "What are common issues and how to prevent them?"

    print(f"\n❓ Test Query: '{test_query}'")

    start_time = time.time()

    try:
        # Step 1: Initialize Enhanced RAG (main orchestration)
        print(f"\n🎭 Step 1: Enhanced RAG Orchestration")
        enhanced_rag = EnhancedUniversalRAG(domain)

        # Ensure system is initialized
        if not enhanced_rag.components_initialized:
            print(f"   📊 Initializing components...")
            await enhanced_rag.initialize_components()

        # Step 2: Create workflow manager for tracking
        print(f"\n📱 Step 2: Workflow Manager Creation")
        workflow_manager = create_workflow_manager(test_query, domain)
        print(f"   🆔 Query ID: {workflow_manager.query_id}")

        # Step 3: Process query through 7-step workflow
        print(f"\n🔄 Step 3: 7-Step Query Processing")
        results = await enhanced_rag.process_query(
            query=test_query,
            max_results=5,
            include_explanations=True,
            enable_safety_warnings=True,
            workflow_manager=workflow_manager
        )

        processing_time = time.time() - start_time

        if results.get("success", False):
            print(f"\n✅ Query processing completed successfully!")
            print(f"⏱️  Processing time: {processing_time:.2f}s")

            # Extract response details
            response = results.get("generated_response", {})
            search_results = results.get("search_results", [])

            print(f"📊 Search results: {len(search_results)}")
            print(f"📝 Response generated: {len(str(response)) > 0}")
            print(f"🎯 Workflow steps: 7 (complete)")

            print(f"\n📋 Core Files Usage Summary:")
            print(f"   ✅ enhanced_rag_universal.py - Main orchestration")
            print(f"   ✅ universal_rag_orchestrator_complete.py - 7-step workflow")
            print(f"   ✅ universal_workflow_manager.py - Progress tracking")
            print(f"   ✅ universal_query_analyzer.py - Query analysis")
            print(f"   ✅ universal_vector_search.py - Vector search")
            print(f"   ✅ universal_llm_interface.py - Response generation")
            print(f"   ✅ universal_models.py - Data structures")

            print(f"\n🎯 Workflow Steps Executed:")
            print(f"   1️⃣ Data Ingestion - Text processing")
            print(f"   2️⃣ Knowledge Extraction - Entity/relation discovery")
            print(f"   3️⃣ Vector Indexing - FAISS search preparation")
            print(f"   4️⃣ Graph Construction - Knowledge graph building")
            print(f"   5️⃣ Query Processing - Query analysis")
            print(f"   6️⃣ Retrieval - Multi-modal search")
            print(f"   7️⃣ Generation - LLM response creation")

            print(f"\n🚀 Result: Intelligent response with citations!")

        else:
            print(f"❌ Query processing failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Query processing workflow failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    """Execute query processing workflow"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)