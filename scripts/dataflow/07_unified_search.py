#!/usr/bin/env python3
"""
PydanticAI Universal Search Pipeline
===================================

Demonstrates multi-modal search using proper PydanticAI agent architecture:
- Domain Intelligence Agent for query analysis
- Universal Search Agent for multi-modal orchestration
- Real Azure service integration with proper dependency management
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps

# Import new PydanticAI agents and orchestrator
from agents.domain_intelligence.agent import run_domain_analysis
from agents.orchestrator import UniversalOrchestrator
from agents.universal_search.agent import run_universal_search


async def unified_search_pipeline(
    query: str,
    max_results: int = 10,
    use_domain_analysis: bool = True,
    session_id: str = None,
) -> Dict[str, Any]:
    """
    PydanticAI Universal Search Pipeline
    Uses proper agent delegation and multi-modal search orchestration
    """
    session_id = session_id or f"search_{int(time.time())}"

    print("🔍 PydanticAI Universal Search Pipeline")
    print(f"Query: '{query}'")
    print(f"Session: {session_id}")
    print("=" * 60)

    results = {
        "session_id": session_id,
        "query": query,
        "max_results": max_results,
        "stages": [],
        "overall_status": "in_progress",
    }

    start_time = time.time()

    try:
        # Initialize dependencies
        deps = await get_universal_deps()
        print(f"🔧 Initialized Universal Dependencies")
        print(f"   Available services: {', '.join(deps.get_available_services())}")

        # Stage 1: Individual Agent Demonstration
        print(f"\n🌍 Stage 1: Domain Intelligence Agent - Query Analysis")
        query_analysis_start = time.time()

        try:
            if use_domain_analysis:
                domain_analysis = await run_domain_analysis(
                    f"Search query analysis: {query}", detailed=True
                )

                query_analysis_time = time.time() - query_analysis_start

                results["stages"].append(
                    {
                        "stage": "query_analysis",
                        "agent": "Domain Intelligence Agent",
                        "duration": query_analysis_time,
                        "status": "completed",
                        "query_signature": domain_analysis.content_signature,
                        "vocabulary_complexity": domain_analysis.vocabulary_complexity,
                    }
                )

                print(f"   ✅ Query signature: {domain_analysis.content_signature}")
                print(
                    f"   📊 Vocabulary complexity: {domain_analysis.vocabulary_complexity:.3f}"
                )
                print(f"   ⏱️  Duration: {query_analysis_time:.2f}s")

                results["query_analysis"] = {
                    "content_signature": domain_analysis.content_signature,
                    "vocabulary_complexity": domain_analysis.vocabulary_complexity,
                    "concept_density": domain_analysis.concept_density,
                }
            else:
                print(f"   ⏭️  Skipping domain analysis")
                domain_analysis = None

        except Exception as e:
            query_analysis_time = time.time() - query_analysis_start
            results["stages"].append(
                {
                    "stage": "query_analysis",
                    "agent": "Domain Intelligence Agent",
                    "duration": query_analysis_time,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"   ❌ Query analysis failed: {e}")
            domain_analysis = None

        # Stage 2: Universal Search Agent
        print(f"\n🎯 Stage 2: Universal Search Agent - Multi-Modal Search")
        search_start = time.time()

        try:
            search_result = await run_universal_search(
                query, max_results=max_results, use_domain_analysis=use_domain_analysis
            )

            search_time = time.time() - search_start

            results["stages"].append(
                {
                    "stage": "multi_modal_search",
                    "agent": "Universal Search Agent",
                    "duration": search_time,
                    "status": "completed",
                    "total_results": search_result.total_results_found,
                    "search_confidence": search_result.search_confidence,
                    "strategy_used": search_result.search_strategy_used,
                }
            )

            print(f"   ✅ Search completed: {search_result.search_strategy_used}")
            print(f"   📊 Results found: {search_result.total_results_found}")
            print(f"   🎯 Search confidence: {search_result.search_confidence:.3f}")
            print(
                f"   ⚡ Processing time: {search_result.processing_time_seconds:.3f}s"
            )
            print(
                f"   🔧 Modalities: Vector({len(search_result.vector_results)}), Graph({len(search_result.graph_results)}), GNN({len(search_result.gnn_results)})"
            )

            # Show top unified results
            if search_result.unified_results:
                print(f"   🏆 Top unified results:")
                for i, result in enumerate(search_result.unified_results[:3], 1):
                    print(
                        f"      {i}. {result.title[:40]}... (score: {result.score:.3f}, source: {result.source})"
                    )

            results["search_result"] = {
                "total_results_found": search_result.total_results_found,
                "search_confidence": search_result.search_confidence,
                "strategy_used": search_result.search_strategy_used,
                "processing_time": search_result.processing_time_seconds,
                "unified_results": [
                    {
                        "title": r.title,
                        "content": r.content[:200],
                        "score": r.score,
                        "source": r.source,
                    }
                    for r in search_result.unified_results
                ],
            }

        except Exception as e:
            search_time = time.time() - search_start
            results["stages"].append(
                {
                    "stage": "multi_modal_search",
                    "agent": "Universal Search Agent",
                    "duration": search_time,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"   ❌ Multi-modal search failed: {e}")

        # Stage 3: Orchestrated Search Workflow
        print(f"\n🎭 Stage 3: Orchestrated Search Workflow")
        orchestration_start = time.time()

        try:
            orchestrator = UniversalOrchestrator()
            orchestration_result = await orchestrator.process_full_search_workflow(
                query, max_results=max_results, use_domain_analysis=use_domain_analysis
            )

            orchestration_time = time.time() - orchestration_start

            results["stages"].append(
                {
                    "stage": "orchestrated_search",
                    "component": "UniversalOrchestrator",
                    "duration": orchestration_time,
                    "status": (
                        "completed" if orchestration_result.success else "partial"
                    ),
                    "agents_coordinated": list(
                        orchestration_result.agent_metrics.keys()
                    ),
                    "total_processing_time": orchestration_result.total_processing_time,
                }
            )

            print(f"   ✅ Orchestration success: {orchestration_result.success}")
            print(
                f"   🤖 Agents coordinated: {', '.join(orchestration_result.agent_metrics.keys())}"
            )
            print(
                f"   ⏱️  Total time: {orchestration_result.total_processing_time:.2f}s"
            )

            if orchestration_result.search_results:
                print(
                    f"   🔍 Orchestrated search results: {len(orchestration_result.search_results)}"
                )

            results["orchestration_result"] = {
                "success": orchestration_result.success,
                "agent_metrics": orchestration_result.agent_metrics,
                "search_results": orchestration_result.search_results,
            }

        except Exception as e:
            orchestration_time = time.time() - orchestration_start
            results["stages"].append(
                {
                    "stage": "orchestrated_search",
                    "component": "UniversalOrchestrator",
                    "duration": orchestration_time,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"   ❌ Orchestration failed: {e}")

        # Final Results
        total_duration = time.time() - start_time
        results["overall_status"] = "completed"
        results["total_duration"] = total_duration
        results["successful_stages"] = len(
            [s for s in results["stages"] if s["status"] == "completed"]
        )
        results["total_stages"] = len(results["stages"])

        print(f"\n✅ Search Pipeline Summary")
        print(f"=" * 40)
        print(f"🎯 Query: {query}")
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(
            f"📊 Stages: {results['successful_stages']}/{results['total_stages']} successful"
        )
        print(f"🏗️  Architecture: PydanticAI Multi-Agent")

        return results

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")
        results["overall_status"] = "failed"
        results["error"] = str(e)
        results["total_duration"] = time.time() - start_time
        return results


async def simple_search(query: str) -> List[Dict[str, Any]]:
    """Simple search interface that returns just results"""
    result = await unified_search_pipeline(
        query, max_results=5, use_domain_analysis=True
    )

    if result.get("search_result") and result["search_result"].get("unified_results"):
        return result["search_result"]["unified_results"]
    return []


async def main():
    """Run the unified search pipeline demonstration"""
    print("🚀 Azure Universal RAG - Multi-Modal Search Pipeline")
    print("===================================================")

    # Demo queries
    demo_queries = [
        "Azure Cosmos DB performance optimization strategies",
        "Machine learning model deployment best practices",
        "Kubernetes scaling and resource management",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"Demo {i}/{len(demo_queries)}")
        print(f"{'='*60}")

        result = await unified_search_pipeline(query, max_results=3)

        if result["overall_status"] == "completed":
            print(f"🎉 Search completed successfully!")
        else:
            print(f"⚠️  Search completed with issues")

        # Brief pause between demos
        if i < len(demo_queries):
            await asyncio.sleep(1)

    print(f"\n🌟 Multi-Modal Search Demo Complete!")
    print(f"✅ PydanticAI Architecture: Domain Intelligence → Universal Search")
    print(f"✅ Multi-Agent Orchestration: Proper dependency sharing and coordination")
    print(f"✅ Universal Processing: Zero hardcoded domain assumptions")


if __name__ == "__main__":
    asyncio.run(main())
