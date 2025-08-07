#!/usr/bin/env python3
"""
Universal Search Demo - PydanticAI Query Generation Pattern
==========================================================

Demonstrates the enhanced universal search with AI-generated queries:
- Domain intelligence analysis with AI-generated prompts
- AI-optimized Azure Cognitive Search queries
- AI-generated Gremlin graph traversal queries
- Query orchestration and optimization strategies

Agent-centric workflow using PydanticAI query generation pattern.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.domain_intelligence.agent import (
    UniversalDomainDeps,
    run_universal_domain_analysis,
)

# Import query generation orchestrator for enhanced search workflows
from agents.query_generation.universal_query_orchestrator import (
    generate_analysis_query_orchestrated,
    generate_gremlin_query_orchestrated,
    generate_search_query_orchestrated,
    query_orchestrator,
)

# Import PydanticAI agents and query orchestration
from agents.universal_search.agent import (
    SearchDeps,
)
from agents.universal_search.agent import agent as search_agent
from agents.universal_search.agent import (
    run_universal_search,
)


async def universal_search_demo_pipeline(
    search_query: str = "Azure RAG system knowledge extraction",
    data_directory: str = "/workspace/azure-maintie-rag/data/raw",
    session_id: str = None,
) -> Dict[str, Any]:
    """
    Enhanced universal search demo with AI-generated query optimization
    Uses PydanticAI agents with query generation orchestration
    """
    session_id = session_id or f"search_demo_{int(time.time())}"
    print("🔍 Universal Search Demo - AI-Generated Query Optimization")
    print(f"Session: {session_id}")
    print("=" * 70)

    results = {
        "session_id": session_id,
        "search_query": search_query,
        "data_directory": data_directory,
        "stages": [],
        "overall_status": "in_progress",
    }

    start_time = time.time()

    try:
        # Stage 1: Domain Intelligence for Search Context
        print("🧠 Stage 1: Domain Intelligence Analysis for Search Optimization")
        stage_start = time.time()

        print(f"   📁 Analyzing domain context from: {Path(data_directory).name}")

        domain_analysis = await run_universal_domain_analysis(
            UniversalDomainDeps(
                data_directory=data_directory,
                max_files_to_analyze=5,
                min_content_length=100,
                enable_multilingual=True,
            )
        )

        stage_duration = time.time() - stage_start
        results["stages"].append(
            {
                "stage": "domain_intelligence",
                "agent": "Domain Intelligence Agent",
                "duration": stage_duration,
                "status": "completed",
                "domain_discovered": domain_analysis.domain_signature,
                "technical_density": domain_analysis.characteristics.technical_vocabulary_ratio,
                "search_optimization": "domain_aware",
            }
        )

        print(f"   ✅ Domain context: {domain_analysis.domain_signature}")
        print(
            f"   📊 Technical density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}"
        )
        print(f"   ⏱️  Duration: {stage_duration:.2f}s")

        # Stage 2: AI-Generated Search Query Optimization
        print(f"\n🎯 Stage 2: AI-Generated Search Query Optimization")
        query_stage_start = time.time()

        print(f'   🔍 Generating optimized search strategies for: "{search_query}"')

        # Generate vector search optimization
        vector_search_strategy = await generate_search_query_orchestrated(
            operation_type="vector",
            search_text=search_query,
            vector_fields=["content_vector"],
            searchable_fields=["content", "title", "metadata"],
            filter_conditions={"domain_type": domain_analysis.domain_signature},
            domain_context={
                "domain_signature": domain_analysis.domain_signature,
                "technical_density": domain_analysis.characteristics.technical_vocabulary_ratio,
                "search_mode": "semantic_similarity",
            },
            similarity_threshold=0.7,
            top_k=10,
        )

        # Generate graph search optimization
        graph_search_strategy = await generate_gremlin_query_orchestrated(
            operation_type="traverse_graph",
            graph_context={
                "start_entity": search_query.split()[0],
                "traversal_type": "both",
                "domain_context": domain_analysis.domain_signature,
            },
            confidence_threshold=0.7,
            max_results=15,
            traversal_depth=3,
        )

        query_stage_duration = time.time() - query_stage_start

        vector_success = vector_search_strategy.success
        graph_success = graph_search_strategy.success

        print(
            f"   {'✅' if vector_success else '⚠️'} Vector search strategy: {'Generated' if vector_success else 'Failed'}"
        )
        if vector_success:
            print(f"      🎯 Strategy: {vector_search_strategy.query_data.explanation}")

        print(
            f"   {'✅' if graph_success else '⚠️'} Graph search strategy: {'Generated' if graph_success else 'Failed'}"
        )
        if graph_success:
            print(f"      🎯 Strategy: {graph_search_strategy.query_data.explanation}")

        print(f"   ⏱️  Query optimization: {query_stage_duration:.2f}s")

        results["stages"].append(
            {
                "stage": "query_optimization",
                "strategy": "AI-generated multi-modal search optimization",
                "duration": query_stage_duration,
                "status": "completed",
                "vector_strategy_success": vector_success,
                "graph_strategy_success": graph_success,
                "optimization_applied": vector_success or graph_success,
            }
        )

        # Stage 3: Enhanced Universal Search Execution
        print(f"\n🚀 Stage 3: Enhanced Universal Search with AI-Optimized Queries")
        search_stage_start = time.time()

        print(f'   🎯 Domain-aware search for: "{search_query}"')
        print(f"   🌍 Domain context: {domain_analysis.domain_signature}")
        print(
            f"   📊 Using {len([s for s in [vector_success, graph_success] if s])} AI-optimized strategies"
        )

        try:
            # Run enhanced universal search with AI-generated optimizations
            search_result = await run_universal_search(
                query=search_query,
                max_results=12,
                similarity_threshold=0.7,
                enable_vector=True,
                enable_graph=True,
                enable_storage=True,
                enable_gnn=True,  # Phase 2: GNN search
                enable_monitoring=True,  # Phase 2: Performance tracking
            )

            search_stage_duration = time.time() - search_stage_start

            results["stages"].append(
                {
                    "stage": "universal_search",
                    "agent": "Universal Search Agent",
                    "duration": search_stage_duration,
                    "status": "completed",
                    "results_found": len(search_result.results),
                    "modalities_used": search_result.modalities_used,
                    "synthesis_score": search_result.synthesis_score,
                    "ai_optimization": "enhanced_query_generation",
                }
            )

            print(f"   ✅ Universal search completed")
            print(f"   📊 Results found: {len(search_result.results)}")
            print(f"   🔗 Modalities used: {', '.join(search_result.modalities_used)}")
            print(f"   🎯 Synthesis score: {search_result.synthesis_score:.3f}")
            print(f"   ⏱️  Search duration: {search_stage_duration:.2f}s")

            # Store search results for summary
            search_results = search_result.results

        except Exception as search_error:
            search_stage_duration = time.time() - search_stage_start
            print(f"   ⚠️  Search execution issue: {str(search_error)[:100]}...")
            print(f"   📝 Simulating search results for demo")

            # Fallback simulated results
            search_results = [
                {
                    "content": f"Azure RAG system with {domain_analysis.domain_signature} domain knowledge",
                    "relevance_score": 0.85,
                    "source": "simulated_vector_search",
                    "result_type": "vector",
                },
                {
                    "content": f"Knowledge extraction patterns for {domain_analysis.domain_signature}",
                    "relevance_score": 0.78,
                    "source": "simulated_graph_search",
                    "result_type": "graph",
                },
            ]

            results["stages"].append(
                {
                    "stage": "universal_search",
                    "agent": "Universal Search Agent",
                    "duration": search_stage_duration,
                    "status": "simulated",
                    "results_found": len(search_results),
                    "modalities_used": ["vector", "graph"],
                    "note": "Search unavailable - simulated results",
                }
            )

            print(f"   📝 Simulated results: {len(search_results)}")
            print(f"   ⏱️  Duration: {search_stage_duration:.2f}s")

        # Stage 4: Results Analysis and Performance Metrics
        print(f"\n📊 Stage 4: Results Analysis with Query Performance Metrics")
        analysis_stage_start = time.time()

        # Generate performance analysis query
        try:
            performance_analysis = await generate_analysis_query_orchestrated(
                analysis_type="performance_optimization",
                domain_context={
                    "search_query": search_query,
                    "domain_signature": domain_analysis.domain_signature,
                    "results_count": len(search_results),
                    "modalities_used": [
                        r.get("result_type", "unknown") for r in search_results
                    ],
                },
                target_metrics=[
                    "search_efficiency",
                    "result_relevance",
                    "modality_coverage",
                ],
                optimization_goals=[
                    "improve_precision",
                    "reduce_latency",
                    "enhance_coverage",
                ],
            )

            if performance_analysis.success:
                print(f"   ✅ Generated performance analysis strategy")
                print(
                    f"   🎯 Analysis approach: {performance_analysis.query_data.execution_strategy}"
                )
                performance_analyzed = True
            else:
                performance_analyzed = False
                print(f"   ⚠️  Performance analysis generation failed")

        except Exception as e:
            performance_analyzed = False
            print(f"   ⚠️  Performance analysis unavailable: {str(e)[:50]}...")

        analysis_stage_duration = time.time() - analysis_stage_start
        print(f"   ⏱️  Analysis duration: {analysis_stage_duration:.2f}s")

        results["stages"].append(
            {
                "stage": "performance_analysis",
                "strategy": "AI-generated performance optimization",
                "duration": analysis_stage_duration,
                "status": "completed" if performance_analyzed else "skipped",
                "analysis_generated": performance_analyzed,
            }
        )

        # Pipeline Complete
        total_time = time.time() - start_time
        results.update(
            {
                "overall_status": "completed",
                "total_duration": total_time,
                "domain_analysis": {
                    "domain_signature": domain_analysis.domain_signature,
                    "confidence": domain_analysis.content_type_confidence,
                    "technical_density": domain_analysis.characteristics.technical_vocabulary_ratio,
                },
                "search_summary": {
                    "query": search_query,
                    "results": search_results[:3],  # First 3 for summary
                    "total_results": len(search_results),
                    "ai_optimizations_applied": sum(
                        [vector_success, graph_success, performance_analyzed]
                    ),
                },
                "query_generation_stats": {
                    "vector_optimization": vector_success,
                    "graph_optimization": graph_success,
                    "performance_analysis": performance_analyzed,
                    "total_ai_queries_generated": sum(
                        [vector_success, graph_success, performance_analyzed]
                    ),
                },
            }
        )

        print(f"\n🎉 Universal Search Demo Complete!")
        print(f"   📄 Session: {session_id}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🌍 Domain: {domain_analysis.domain_signature}")
        print(
            f"   🤖 Agents used: Domain Intelligence + Universal Search + Query Generation"
        )
        print(f"   📊 Results: {len(search_results)} items across multiple modalities")
        print(
            f"   🎯 AI optimizations: {sum([vector_success, graph_success, performance_analyzed])}/3 successful"
        )

        return results

    except Exception as e:
        total_time = time.time() - start_time
        results.update(
            {
                "overall_status": "failed",
                "total_duration": total_time,
                "error": str(e),
                "stages_completed": len(results.get("stages", [])),
            }
        )

        print(f"❌ Universal Search Demo Failed!")
        print(f"   📄 Session: {session_id}")
        print(f"   ⏱️  Time elapsed: {total_time:.2f}s")
        print(f"   📊 Stages completed: {len(results.get('stages', []))}")
        print(f"   ❌ Error: {e}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Search Demo - AI Query Generation"
    )
    parser.add_argument(
        "--query",
        default="Azure RAG system knowledge extraction",
        help="Search query to execute",
    )
    parser.add_argument(
        "--data-dir",
        default="/workspace/azure-maintie-rag/data/raw",
        help="Data directory for domain analysis",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    print("🔍 Azure Universal RAG - Enhanced Search Demo")
    print("=" * 70)
    print("AI-enhanced workflow using:")
    print("• Domain Intelligence Agent → discovers domain characteristics")
    print("• Query Generation Orchestrator → optimizes search strategies")
    print("• Universal Search Agent → executes multi-modal search")
    print("• Performance Analysis → evaluates and optimizes results")
    print("")

    # Run the enhanced demo pipeline
    result = asyncio.run(
        universal_search_demo_pipeline(
            search_query=args.query, data_directory=args.data_dir
        )
    )

    # Handle JSON output
    if args.json or args.output:
        json_output = json.dumps(result, indent=2, default=str)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_output)
            print(f"\n📄 Results saved to: {output_path}")

        if args.json:
            print(f"\n" + "=" * 70)
            print("Pipeline Results JSON:")
            print(json_output)

    # Final summary
    if result["overall_status"] == "completed":
        print(f"\n🎉 SUCCESS: Enhanced search demo completed!")
        print(f"   📄 Session: {result['session_id']}")
        print(f"   ⏱️  Duration: {result['total_duration']:.2f}s")
        print(f"   🤖 {len(result['stages'])} agent stages completed")
        print(
            f"   🎯 AI optimizations applied: {result['query_generation_stats']['total_ai_queries_generated']}"
        )
        print("Enhanced universal search with AI-generated query optimization!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Pipeline encountered issues.")
        print(f"   📄 Session: {result['session_id']}")
        print(f"   ⏱️  Duration: {result.get('total_duration', 0):.2f}s")
        print(f"   🤖 Stages completed: {len(result.get('stages', []))}")
        print("Check the error messages above for details.")
        sys.exit(1)
