#!/usr/bin/env python3
"""
Query Pipeline Orchestrator - Complete Query Phase (Stages 06-09)
User Query → Query Analysis → Unified Search → Context Retrieval → Response Generation

This script orchestrates the complete query phase of the README architecture:
- Stage 06: Query Analysis (User Query → Azure OpenAI Analysis)
- Stage 07: Unified Search (Query Analysis → Vector + Graph + GNN Search)
- Stage 08: Context Retrieval (Search Results → Context Preparation)
- Stage 09: Response Generation (Context → Final Answer with Citations)
"""

import argparse
import asyncio
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService

# Import individual stage classes using importlib
query_analysis_module = importlib.import_module("06_query_analysis")
unified_search_module = importlib.import_module("07_unified_search")
context_retrieval_module = importlib.import_module("08_context_retrieval")
response_generation_module = importlib.import_module("09_response_generation")

GNNQueryAnalysisStage = query_analysis_module.GNNQueryAnalysisStage
UnifiedSearchStage = unified_search_module.UnifiedSearchStage
ContextRetrievalStage = context_retrieval_module.ContextRetrievalStage
ResponseGenerationStage = response_generation_module.ResponseGenerationStage

logger = logging.getLogger(__name__)


class QueryPipelineOrchestrator:
    """Complete Query Phase Orchestrator - Stages 06-09"""

    def __init__(self):
        self.infrastructure = InfrastructureService()

        # Initialize stage classes
        self.query_analysis = GNNQueryAnalysisStage()
        self.unified_search = UnifiedSearchStage()
        self.context_retrieval = ContextRetrievalStage()
        self.response_generation = ResponseGenerationStage()

    async def execute_query_pipeline(
        self,
        user_query: str,
        domain: str = "general",
        response_style: str = "comprehensive",
        max_context_length: int = 8000,
        streaming_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete query pipeline

        Args:
            user_query: User's natural language query
            domain: Target domain
            response_style: Response style (comprehensive, concise, detailed)
            max_context_length: Maximum context length
            streaming_callback: Optional callback for streaming updates

        Returns:
            Dict with complete query pipeline results
        """
        print("🎯 Query Pipeline Orchestrator - Complete Query Phase")
        print("=" * 65)

        start_time = asyncio.get_event_loop().time()

        pipeline_results = {
            "orchestrator": "10_query_pipeline",
            "user_query": user_query,
            "domain": domain,
            "response_style": response_style,
            "pipeline_start": datetime.now().isoformat(),
            "stages_executed": [],
            "stage_results": {},
            "pipeline_metrics": {
                "total_duration": 0,
                "query_analysis_time": 0,
                "search_time": 0,
                "context_preparation_time": 0,
                "response_generation_time": 0,
                "total_search_results": 0,
                "context_length": 0,
                "response_length": 0,
                "citations_count": 0,
            },
            "final_answer": {},
            "success": False,
        }

        try:
            # Stage 06: Query Analysis
            print(f"\n🔄 Stage 06: Query Analysis")
            if streaming_callback:
                await streaming_callback(
                    "query_analysis", "started", {"query": user_query}
                )

            stage06_start = asyncio.get_event_loop().time()
            stage06_result = await self.query_analysis.execute(
                query=user_query, domain=domain
            )
            stage06_duration = asyncio.get_event_loop().time() - stage06_start

            pipeline_results["stage_results"]["06_query_analysis"] = stage06_result
            pipeline_results["stages_executed"].append("06")
            pipeline_results["pipeline_metrics"]["query_analysis_time"] = round(
                stage06_duration, 2
            )

            if not stage06_result.get("success"):
                raise Exception(f"Stage 06 failed: {stage06_result.get('error')}")

            analysis_data = stage06_result.get("analysis", {})
            print(
                f"✅ Stage 06 Complete: {analysis_data.get('query_type', 'unknown')} query analyzed"
            )

            if streaming_callback:
                await streaming_callback(
                    "query_analysis",
                    "completed",
                    {
                        "query_type": analysis_data.get("query_type"),
                        "entities": analysis_data.get("entities", []),
                    },
                )

            # Stage 07: Unified Search
            print(f"\n🔄 Stage 07: Unified Search")
            if streaming_callback:
                await streaming_callback(
                    "unified_search",
                    "started",
                    {"search_strategy": stage06_result.get("search_strategy", {})},
                )

            stage07_start = asyncio.get_event_loop().time()
            stage07_result = await self.unified_search.execute(
                query=user_query, domain=domain, max_results=20
            )
            stage07_duration = asyncio.get_event_loop().time() - stage07_start

            pipeline_results["stage_results"]["07_unified_search"] = stage07_result
            pipeline_results["stages_executed"].append("07")
            pipeline_results["pipeline_metrics"]["search_time"] = round(
                stage07_duration, 2
            )
            pipeline_results["pipeline_metrics"][
                "total_search_results"
            ] = stage07_result.get("total_results", 0)

            if not stage07_result.get("success"):
                raise Exception(f"Stage 07 failed: {stage07_result.get('error')}")

            search_components = stage07_result.get("search_components", {})
            print(
                f"✅ Stage 07 Complete: {stage07_result.get('total_results', 0)} unified results"
            )

            if streaming_callback:
                await streaming_callback(
                    "unified_search",
                    "completed",
                    {
                        "total_results": stage07_result.get("total_results", 0),
                        "search_components": search_components,
                    },
                )

            # Stage 08: Context Retrieval
            print(f"\n🔄 Stage 08: Context Retrieval")
            if streaming_callback:
                await streaming_callback(
                    "context_retrieval",
                    "started",
                    {"max_context_length": max_context_length},
                )

            stage08_start = asyncio.get_event_loop().time()
            stage08_result = await self.context_retrieval.execute(
                query=user_query, domain=domain, max_context_items=15
            )
            stage08_duration = asyncio.get_event_loop().time() - stage08_start

            pipeline_results["stage_results"]["08_context_retrieval"] = stage08_result
            pipeline_results["stages_executed"].append("08")
            pipeline_results["pipeline_metrics"]["context_preparation_time"] = round(
                stage08_duration, 2
            )

            # Extract context statistics from stage 08 result
            total_context_items = stage08_result.get("total_context_items", 0)
            total_citations = stage08_result.get("total_citations", 0)
            pipeline_results["pipeline_metrics"]["context_length"] = total_context_items
            pipeline_results["pipeline_metrics"]["citations_count"] = total_citations

            if not stage08_result.get("success"):
                raise Exception(f"Stage 08 failed: {stage08_result.get('error')}")

            print(
                f"✅ Stage 08 Complete: {total_context_items} context items, {total_citations} citations"
            )

            if streaming_callback:
                await streaming_callback(
                    "context_retrieval",
                    "completed",
                    {
                        "context_length": total_context_items,
                        "citations_count": total_citations,
                    },
                )

            # Stage 09: Response Generation
            print(f"\n🔄 Stage 09: Response Generation")
            if streaming_callback:
                await streaming_callback(
                    "response_generation", "started", {"response_style": response_style}
                )

            stage09_start = asyncio.get_event_loop().time()
            stage09_result = await self.response_generation.execute(
                query=user_query, domain=domain, max_results=15
            )
            stage09_duration = asyncio.get_event_loop().time() - stage09_start

            pipeline_results["stage_results"]["09_response_generation"] = stage09_result
            pipeline_results["stages_executed"].append("09")
            pipeline_results["pipeline_metrics"]["response_generation_time"] = round(
                stage09_duration, 2
            )

            if not stage09_result.get("success"):
                raise Exception(f"Stage 09 failed: {stage09_result.get('error')}")

            # Extract final answer and citations from stage 09 result
            final_answer_text = stage09_result.get("final_answer", "")
            citations = stage09_result.get("citations", [])

            final_response = {"answer": final_answer_text, "citations": citations}
            pipeline_results["final_answer"] = final_response
            pipeline_results["pipeline_metrics"]["response_length"] = len(
                final_answer_text
            )

            response_length = stage09_result.get("response_length", 0)
            print(f"✅ Stage 09 Complete: {response_length} char response")

            if streaming_callback:
                await streaming_callback(
                    "response_generation",
                    "completed",
                    {
                        "response_length": response_length,
                        "quality_score": 0,  # Not available in current implementation
                    },
                )

            # Pipeline Success
            total_duration = asyncio.get_event_loop().time() - start_time
            pipeline_results["pipeline_metrics"]["total_duration"] = round(
                total_duration, 2
            )
            pipeline_results["pipeline_end"] = datetime.now().isoformat()
            pipeline_results["success"] = True

            print(f"\n🎉 Query Pipeline Complete!")
            print(f"   🔍 Query analyzed: {analysis_data.get('query_type', 'unknown')}")
            print(
                f"   🎯 Search results: {pipeline_results['pipeline_metrics']['total_search_results']}"
            )
            print(
                f"   📚 Context length: {pipeline_results['pipeline_metrics']['context_length']:,} chars"
            )
            print(
                f"   💬 Response length: {pipeline_results['pipeline_metrics']['response_length']:,} chars"
            )
            print(
                f"   📎 Citations: {pipeline_results['pipeline_metrics']['citations_count']}"
            )
            print(
                f"   ⏱️  Total duration: {pipeline_results['pipeline_metrics']['total_duration']}s"
            )

            if streaming_callback:
                await streaming_callback(
                    "pipeline", "completed", pipeline_results["pipeline_metrics"]
                )

            return pipeline_results

        except Exception as e:
            pipeline_results["error"] = str(e)
            pipeline_results["pipeline_end"] = datetime.now().isoformat()
            total_duration = asyncio.get_event_loop().time() - start_time
            pipeline_results["pipeline_metrics"]["total_duration"] = round(
                total_duration, 2
            )

            print(f"\n❌ Query Pipeline Failed: {e}")
            print(f"   📊 Stages completed: {len(pipeline_results['stages_executed'])}")
            print(
                f"   ⏱️  Duration: {pipeline_results['pipeline_metrics']['total_duration']}s"
            )

            if streaming_callback:
                await streaming_callback("pipeline", "failed", {"error": str(e)})

            logger.error(f"Query pipeline failed: {e}", exc_info=True)
            return pipeline_results

    async def execute_streaming_query(
        self,
        user_query: str,
        domain: str = "general",
        response_style: str = "comprehensive",
        max_context_length: int = 8000,
    ) -> Dict[str, Any]:
        """
        Execute query pipeline with streaming progress updates

        This method provides real-time progress updates as the query is processed
        """
        streaming_events = []

        async def streaming_callback(stage: str, status: str, data: Dict[str, Any]):
            """Capture streaming events"""
            event = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "status": status,
                "data": data,
            }
            streaming_events.append(event)
            print(f"📡 Streaming Update: {stage} - {status}")

        # Execute pipeline with streaming
        result = await self.execute_query_pipeline(
            user_query=user_query,
            domain=domain,
            response_style=response_style,
            max_context_length=max_context_length,
            streaming_callback=streaming_callback,
        )

        # Add streaming events to result
        result["streaming_events"] = streaming_events

        return result


async def main():
    """Main entry point for query pipeline orchestrator"""
    parser = argparse.ArgumentParser(
        description="Query Pipeline Orchestrator - Complete Query Phase (Stages 06-09)"
    )
    parser.add_argument("query", help="User's natural language query")
    parser.add_argument("--domain", default="general", help="Target domain")
    parser.add_argument(
        "--response-style",
        choices=["comprehensive", "concise", "detailed"],
        default="comprehensive",
        help="Response style",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=8000,
        help="Maximum context length in characters",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Enable streaming progress updates"
    )
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = QueryPipelineOrchestrator()

    if args.streaming:
        # Execute with streaming
        results = await orchestrator.execute_streaming_query(
            user_query=args.query,
            domain=args.domain,
            response_style=args.response_style,
            max_context_length=args.max_context_length,
        )
    else:
        # Execute normal pipeline
        results = await orchestrator.execute_query_pipeline(
            user_query=args.query,
            domain=args.domain,
            response_style=args.response_style,
            max_context_length=args.max_context_length,
        )

    # Display final answer to user
    if results.get("success"):
        final_answer = results.get("final_answer", {})
        print(f"\n{'='*75}")
        print(f"🎯 FINAL ANSWER:")
        print(f"{'='*75}")
        print(final_answer.get("answer", "No answer generated"))
        print(f"\n📚 CITATIONS:")
        for citation in final_answer.get("citations", []):
            citation_id = citation.get("citation_id", citation.get("id", "Unknown"))
            print(
                f"{citation_id} {citation.get('source_type', 'Unknown')}: {citation.get('content_preview', '')}"
            )
        print(f"{'='*75}")

    # Save results if requested
    if args.output and results.get("success"):
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")

    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
