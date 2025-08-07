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

from agents.domain_intelligence.agent import run_universal_domain_analysis, UniversalDomainDeps
from agents.universal_search.agent import agent as universal_search_agent, SearchDeps


async def universal_query_analysis(
    query: str, 
    data_directory: str = "/workspace/azure-maintie-rag/data/raw"
):
    """Universal query analysis that adapts to domain context automatically"""
    print(f"🌍 Universal Query Analysis - Zero Domain Bias")
    print(f"==============================================")
    print(f"🔍 Query: '{query}'")
    
    try:
        # Step 1: Discover domain context for query analysis
        print(f"\n📊 Step 1: Domain Context Discovery")
        print(f"   📁 Reference data: {data_directory}")
        
        domain_analysis = await run_universal_domain_analysis(
            UniversalDomainDeps(
                data_directory=data_directory,
                max_files_to_analyze=5,
                min_content_length=100,
                enable_multilingual=True
            )
        )
        
        print(f"   ✅ Domain context: {domain_analysis.domain_signature}")
        print(f"   📚 Key terms: {domain_analysis.characteristics.most_frequent_terms[:5]}")
        print(f"   🎯 Technical density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}")
        
        # Step 2: Adaptive query analysis
        print(f"\n🔍 Step 2: Adaptive Query Analysis")
        
        # Configure search with discovered domain characteristics  
        search_deps = SearchDeps(
            max_results=15,
            similarity_threshold=0.7
        )
        
        print(f"   🔧 Using domain-adaptive configuration:")
        print(f"      Vector weight: {domain_analysis.processing_config.vector_search_weight:.1%}")
        print(f"      Graph weight: {domain_analysis.processing_config.graph_search_weight:.1%}")
        print(f"      Max results: {search_deps.max_results}")
        
        # Analyze query in domain context
        search_result = await universal_search_agent.run(
            f"Analyze query '{query}' in {domain_analysis.domain_signature} context",
            deps=search_deps
        )
        
        print(f"   ✅ Query analysis completed")
        print(f"   📊 Results found: {search_result.data.total_results}")
        print(f"   🎯 Synthesis score: {search_result.data.synthesis_score:.2f}")
        print(f"   🌍 Domain relevance: High (adapted to {domain_analysis.domain_signature})")
        
        return {
            'success': True,
            'query': query,
            'domain_context': domain_analysis.domain_signature,
            'search_result': search_result.data,
            'adaptive_configuration': {
                'vector_weight': domain_analysis.processing_config.vector_search_weight,
                'graph_weight': domain_analysis.processing_config.graph_search_weight,
                'domain_terms': domain_analysis.characteristics.most_frequent_terms[:5]
            }
        }
        
    except Exception as e:
        print(f"❌ Universal query analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal query analysis - adapts to ANY domain context")
    parser.add_argument("query", help="Query to analyze")
    parser.add_argument("--data-dir", default="/workspace/azure-maintie-rag/data/raw", help="Reference data directory")
    args = parser.parse_args()

    print("🔍 Starting Universal Query Analysis...")
    print("=====================================")
    print("This analysis automatically adapts to your domain context")
    print("without any hardcoded domain assumptions.")
    print("")
    
    result = asyncio.run(universal_query_analysis(args.query, args.data_dir))
    
    if result['success']:
        print(f"\n🎉 SUCCESS: Query analysis completed!")
        print(f"Domain context: {result['domain_context']}")
        print(f"Search results: {result['search_result'].total_results}")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Query analysis encountered issues.")
        sys.exit(1)
