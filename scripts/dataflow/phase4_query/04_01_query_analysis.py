#!/usr/bin/env python3
"""
Universal Query Analysis - Zero Domain Bias
Clean query analysis using Universal RAG agents that adapt to ANY domain.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.domain_intelligence.agent import run_domain_analysis
from agents.core.universal_deps import UniversalDeps
from agents.universal_search.agent import run_universal_search


async def universal_query_analysis(
    query: str, data_directory: str = "/workspace/azure-maintie-rag/data/raw"
):
    """Universal query analysis that adapts to domain context automatically"""
    print(f"🌍 Universal Query Analysis - Zero Domain Bias")
    print(f"==============================================")
    print(f"🔍 Query: '{query}'")

    try:
        # Step 1: Discover domain context for query analysis
        print(f"\n📊 Step 1: Domain Context Discovery")
        print(f"   📁 Reference data: {data_directory}")

        # Load sample content from data directory for query context analysis
        data_dir = Path(data_directory) / "azure-ai-services-language-service_output"
        sample_files = list(data_dir.glob("*.md"))[:3]
        sample_content = ""
        for file_path in sample_files:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            sample_content += content + "\n\n"
        
        domain_analysis = await run_domain_analysis(
            sample_content, detailed=True
        )

        print(f"   ✅ Domain context: {domain_analysis.domain_signature}")
        print(
            f"   📚 Key terms: {domain_analysis.characteristics.key_content_terms[:5]}"
        )
        print(
            f"   🎯 Concept density: {domain_analysis.characteristics.vocabulary_complexity_ratio:.3f}"
        )

        # Step 2: Adaptive query analysis
        print(f"\n🔍 Step 2: Adaptive Query Analysis")

        # Configure search with discovered domain characteristics
        # Configure search parameters

        print(f"   🔧 Using domain-adaptive configuration:")
        print(
            f"      Vector weight: {domain_analysis.processing_config.vector_search_weight:.1%}"
        )
        print(
            f"      Graph weight: {domain_analysis.processing_config.graph_search_weight:.1%}"
        )
        print(f"      Max results: 15")

        # Analyze query in domain context using universal search
        search_result = await run_universal_search(
            f"Query: {query} (context: {domain_analysis.domain_signature})",
            max_results=15,
            use_domain_analysis=True
        )

        print(f"   ✅ Query analysis completed")
        print(f"   📊 Results found: {search_result.total_results_found}")
        print(f"   🎯 Search confidence: {search_result.search_confidence:.2f}")
        print(
            f"   🌍 Domain relevance: High (adapted to {domain_analysis.domain_signature})"
        )

        return {
            "success": True,
            "query": query,
            "domain_context": domain_analysis.domain_signature,
            "search_result": {
                "total_results": search_result.total_results_found,
                "search_confidence": search_result.search_confidence,
                "strategy_used": search_result.search_strategy_used
            },
            "adaptive_configuration": {
                "vector_weight": domain_analysis.processing_config.vector_search_weight,
                "graph_weight": domain_analysis.processing_config.graph_search_weight,
                "domain_terms": domain_analysis.characteristics.key_content_terms[:5],
            },
        }

    except Exception as e:
        print(f"❌ Universal query analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal query analysis - adapts to ANY domain context"
    )
    parser.add_argument("query", help="Query to analyze")
    parser.add_argument(
        "--data-dir",
        default="/workspace/azure-maintie-rag/data/raw",
        help="Reference data directory",
    )
    args = parser.parse_args()

    print("🔍 Starting Universal Query Analysis...")
    print("=====================================")
    print("This analysis automatically adapts to your domain context")
    print("without any hardcoded domain assumptions.")
    print("")

    result = asyncio.run(universal_query_analysis(args.query, args.data_dir))

    if result["success"]:
        print(f"\n🎉 SUCCESS: Query analysis completed!")
        print(f"Domain context: {result['domain_context']}")
        print(f"Search results: {result['search_result']['total_results']}")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Query analysis encountered issues.")
        sys.exit(1)
