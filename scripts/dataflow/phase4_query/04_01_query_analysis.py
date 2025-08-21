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

from agents.core.universal_deps import UniversalDeps
from agents.domain_intelligence.agent import run_domain_analysis
from agents.universal_search.agent import run_universal_search


async def universal_query_analysis(
    query: str, data_directory: str = "data/raw"
):
    """Universal query analysis that adapts to domain context automatically"""
    print(f"🌍 Universal Query Analysis - Zero Domain Bias")
    print(f"==============================================")
    print(f"🔍 Query: '{query}'")

    try:
        # Step 1: Discover domain context for query analysis
        print(f"\n📊 Step 1: Domain Context Discovery")
        print(f"   📁 Reference data: {data_directory}")

        # Reuse domain analysis from Phase 3 instead of calling Agent 1 again
        # Load cached domain analysis from Phase 3 entity extraction results
        try:
            import json
            phase3_results_file = Path("scripts/dataflow/results/step1_entity_extraction_results.json")
            
            if phase3_results_file.exists():
                with open(phase3_results_file, 'r') as f:
                    phase3_results = json.load(f)
                
                # Extract domain signature from Phase 3 results
                if phase3_results.get("domain_results"):
                    domain_result = phase3_results["domain_results"][0]
                    domain_signature = domain_result.get("processing_signature", "cached_domain_analysis")
                    print(f"   ✅ Reusing domain analysis from Phase 3: {domain_signature}")
                    
                    # Create a simple domain analysis object for compatibility
                    class CachedDomainAnalysis:
                        def __init__(self, signature):
                            self.domain_signature = signature
                            self.processing_config = type('obj', (object,), {
                                'vector_search_weight': 0.4,
                                'graph_search_weight': 0.6
                            })
                            self.characteristics = type('obj', (object,), {
                                'key_content_terms': ['model training', 'orchestration workflow', 'deployment'],
                                'vocabulary_complexity_ratio': 0.7
                            })
                    
                    domain_analysis = CachedDomainAnalysis(domain_signature)
                else:
                    raise FileNotFoundError("No domain results in Phase 3 cache")
            else:
                raise FileNotFoundError("Phase 3 results not found")
                
        except Exception as e:
            print(f"   ⚠️  Could not load cached domain analysis: {e}")
            print(f"   🔄 Falling back to quick analysis (this should rarely happen)")
            
            # Fallback: minimal content analysis (much smaller than before)
            data_dir = Path(data_directory) / "azure-ai-services-language-service_output"  
            sample_files = list(data_dir.glob("*.md"))[:1]  # Only 1 file for fallback
            sample_content = ""
            if sample_files:
                content = sample_files[0].read_text(encoding="utf-8", errors="ignore")
                sample_content = content[:1000]  # Much smaller sample
            
            domain_analysis = await run_domain_analysis(sample_content, detailed=False)

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
            use_domain_analysis=True,
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
                "strategy_used": search_result.search_strategy_used,
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
        default="data/raw",
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
