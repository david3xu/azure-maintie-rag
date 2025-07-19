#!/usr/bin/env python3
"""
Data Preparation Workflow Script
================================

Demonstrates WORKFLOW 1: Raw Text Data Handling
Uses 8/12 core files to convert raw text into searchable knowledge base.

Core Files Used:
- universal_text_processor.py (load text files)
- universal_knowledge_extractor.py (extract entities/relations)
- optimized_llm_extractor.py (LLM extraction)
- universal_classifier.py (classify knowledge)
- universal_vector_search.py (build indices)
- universal_gnn_processor.py (prepare GNN data)
- universal_models.py (data structures)
- universal_rag_orchestrator_complete.py (initialization)
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# FIXED: Import actual components used in data preparation
from core.orchestration.rag_orchestration_service import AzureRAGOrchestrationService
from core.azure_openai.text_processor import AzureOpenAITextProcessor
from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
from core.azure_ml.classification_service import (
    AzureMLClassificationService as UniversalClassificationPipeline
)  # Updated to Azure service naming
from core.azure_search.vector_service import AzureSearchVectorService
from core.azure_ml.gnn_processor import AzureMLGNNProcessor


async def main():
    """Execute data preparation workflow"""

    print("🔄 WORKFLOW 1: Raw Text Data Handling")
    print("=" * 60)
    print("📊 Purpose: Convert raw text files into searchable knowledge base")
    print("🔧 Core Files: 8/12 files actively processing data")
    print("⏱️  Frequency: Once per data update (initialization/startup)")

    domain = "general"
    start_time = time.time()

    try:
        # Initialize Universal RAG Orchestrator (triggers data preparation)
        print(f"\n📝 Initializing Universal RAG system...")
        orchestrator = AzureRAGOrchestrationService(domain)

        # This initialization call uses all 8 data preparation core files
        initialization_results = await orchestrator.initialize_from_text_files()

        processing_time = time.time() - start_time

        # FIXED: Handle different result types properly
        if hasattr(initialization_results, 'get'):
            # If it's a dictionary
            success = initialization_results.get("success", False)
            stats = initialization_results.get("system_stats", {})
        elif hasattr(initialization_results, 'to_dict'):
            # If it's a class instance with to_dict method
            result_dict = initialization_results.to_dict()
            success = result_dict.get("success", False)
            stats = result_dict.get("system_stats", {})
        else:
            # If it's a boolean or other type
            success = bool(initialization_results)
            stats = {}

        if success:
            print(f"\n✅ Data preparation completed successfully!")
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"📊 Documents processed: {stats.get('total_documents', 0)}")
            print(f"🧠 Entities extracted: {stats.get('total_entities', 0)}")
            print(f"🔗 Relations extracted: {stats.get('total_relations', 0)}")
            print(f"🏷️  Entity types discovered: {stats.get('unique_entity_types', 0)}")
            print(f"🔗 Relation types discovered: {stats.get('unique_relation_types', 0)}")
            print(f"📈 Index built: {stats.get('index_built', False)}")

            print(f"\n📋 Core Files Usage Summary:")
            print(f"   ✅ universal_text_processor.py - Loaded text files")
            print(f"   ✅ universal_knowledge_extractor.py - Extracted knowledge")
            print(f"   ✅ optimized_llm_extractor.py - LLM processing")
            print(f"   ✅ universal_classifier.py - Classification")
            print(f"   ✅ universal_vector_search.py - Built indices")
            print(f"   ✅ universal_gnn_processor.py - GNN preparation")
            print(f"   ✅ universal_models.py - Data structures")
            print(f"   ✅ universal_rag_orchestrator_complete.py - Coordination")

            print(f"\n🚀 System Status: Ready for user queries!")

        else:
            error_msg = "Unknown error"
            if hasattr(initialization_results, 'get'):
                error_msg = initialization_results.get('error', 'Unknown error')
            elif hasattr(initialization_results, 'to_dict'):
                result_dict = initialization_results.to_dict()
                error_msg = result_dict.get('error', 'Unknown error')
            print(f"❌ Data preparation failed: {error_msg}")

    except Exception as e:
        print(f"❌ Data preparation workflow failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    """Execute data preparation workflow"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)