#!/usr/bin/env python3
"""
Workflow Analysis Script
=======================

Analyzes core file usage patterns in both workflows.
Provides insights into system architecture and file responsibilities.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def analyze_core_files():
    """Analyze core file usage patterns"""

    core_files = {
        "orchestration/enhanced_rag_universal.py": {
            "data_prep": False,
            "query_runtime": True,
            "purpose": "Runtime orchestration only - coordinates query processing"
        },
        "orchestration/universal_rag_orchestrator_complete.py": {
            "data_prep": True,
            "query_runtime": True,
            "purpose": "Both workflows - initializes system + processes queries"
        },
        "workflow/universal_workflow_manager.py": {
            "data_prep": False,
            "query_runtime": True,
            "purpose": "Runtime tracking only - tracks query progress"
        },
        "extraction/universal_knowledge_extractor.py": {
            "data_prep": True,
            "query_runtime": False,
            "purpose": "Data prep only - extracts knowledge from raw text"
        },
        "extraction/optimized_llm_extractor.py": {
            "data_prep": True,
            "query_runtime": False,
            "purpose": "Data prep only - LLM calls for extraction"
        },
        "extraction/llm_knowledge_extractor.py": {
            "data_prep": True,
            "query_runtime": False,
            "purpose": "Data prep only - base extraction functionality"
        },
        "knowledge/universal_text_processor.py": {
            "data_prep": True,
            "query_runtime": False,
            "purpose": "Data prep only - loads and processes raw text files"
        },
        "classification/universal_classifier.py": {
            "data_prep": True,
            "query_runtime": False,
            "purpose": "Data prep only - classifies extracted entities/relations"
        },
        "retrieval/universal_vector_search.py": {
            "data_prep": True,
            "query_runtime": True,
            "purpose": "Both workflows - builds indices + searches"
        },
        "enhancement/universal_query_analyzer.py": {
            "data_prep": False,
            "query_runtime": True,
            "purpose": "Runtime only - analyzes user queries"
        },
        "generation/universal_llm_interface.py": {
            "data_prep": False,
            "query_runtime": True,
            "purpose": "Runtime only - generates responses"
        },
        "gnn/universal_gnn_processor.py": {
            "data_prep": True,
            "query_runtime": False,
            "purpose": "Data prep only - prepares GNN data structures"
        }
    }

    print("📊 UNIVERSAL RAG CORE FILES USAGE ANALYSIS")
    print("=" * 80)

    data_prep_count = sum(1 for f in core_files.values() if f["data_prep"])
    query_runtime_count = sum(1 for f in core_files.values() if f["query_runtime"])
    both_count = sum(1 for f in core_files.values() if f["data_prep"] and f["query_runtime"])

    print(f"📋 Summary:")
    print(f"   🔸 Total core files analyzed: {len(core_files)}")
    print(f"   🔸 Data preparation workflow: {data_prep_count} files")
    print(f"   🔸 Query processing workflow: {query_runtime_count} files")
    print(f"   🔸 Used by both workflows: {both_count} files")
    print(f"   🔸 System utilization: {(data_prep_count + query_runtime_count - both_count) / len(core_files) * 100:.1f}%")

    print(f"\n📊 Detailed Analysis:")

    print(f"\n🔹 DATA PREPARATION WORKFLOW FILES:")
    for filepath, info in core_files.items():
        if info["data_prep"]:
            shared = "📍 SHARED" if info["query_runtime"] else ""
            print(f"   ✅ {filepath} {shared}")
            print(f"      └─ {info['purpose']}")

    print(f"\n🔸 QUERY PROCESSING WORKFLOW FILES:")
    for filepath, info in core_files.items():
        if info["query_runtime"]:
            shared = "📍 SHARED" if info["data_prep"] else ""
            print(f"   ✅ {filepath} {shared}")
            print(f"      └─ {info['purpose']}")

    print(f"\n🎯 Architecture Benefits:")
    print(f"   ✅ Clear separation of concerns")
    print(f"   ✅ Data prep happens once, queries are fast")
    print(f"   ✅ Can rebuild indices without affecting runtime")
    print(f"   ✅ Scalable architecture with minimal shared components")


if __name__ == "__main__":
    """Execute workflow analysis"""
    analyze_core_files()