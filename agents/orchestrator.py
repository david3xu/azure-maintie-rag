"""
Universal RAG Orchestrator - Proper PydanticAI Multi-Agent Coordination
========================================================================

Orchestrates the Azure Universal RAG multi-agent system using proper PydanticAI patterns.
Demonstrates agent delegation, dependency sharing, and workflow coordination.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.core.universal_models import UniversalDomainAnalysis, SearchResult
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.universal_search.agent import run_universal_search


class UniversalWorkflowResult(BaseModel):
    """Results from universal multi-agent workflow."""

    success: bool
    domain_analysis: Optional[UniversalDomainAnalysis] = None
    extraction_summary: Optional[Dict[str, Any]] = None
    search_results: Optional[List[SearchResult]] = None
    total_processing_time: float = Field(ge=0.0)
    agent_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class UniversalOrchestrator:
    """
    Universal RAG orchestrator using proper PydanticAI patterns.

    Demonstrates:
    - Proper agent delegation with shared dependencies
    - Multi-agent workflow coordination
    - Universal content processing without domain assumptions
    - Real Azure service integration
    """

    def __init__(self):
        """Initialize universal orchestrator with shared dependencies."""
        self.version = "3.0.0-pydantic-ai"
        self._deps: Optional[UniversalDeps] = None

    async def _ensure_deps(self) -> UniversalDeps:
        """Ensure dependencies are initialized."""
        if self._deps is None:
            self._deps = await get_universal_deps()
        return self._deps

    async def process_content_with_domain_analysis(
        self, content: str
    ) -> UniversalWorkflowResult:
        """
        Process content through domain analysis with proper PydanticAI patterns.

        Demonstrates proper agent delegation and shared dependency usage.
        """
        start_time = time.time()
        deps = await self._ensure_deps()
        errors = []
        warnings = []
        agent_metrics = {}

        try:
            print("🌍 Step 1: Universal domain analysis...")

            # Step 1: Domain Intelligence Agent
            domain_start = time.time()
            try:
                domain_analysis = await run_domain_analysis(content, detailed=True)
                domain_time = time.time() - domain_start
                agent_metrics["domain_intelligence"] = {
                    "processing_time": domain_time,
                    "success": True,
                    "content_signature": domain_analysis.content_signature,
                    "vocabulary_complexity": domain_analysis.vocabulary_complexity,
                    "concept_density": domain_analysis.concept_density,
                }
                print(f"   ✅ Domain analysis completed in {domain_time:.2f}s")
                print(f"   📊 Content signature: {domain_analysis.content_signature}")
                print(
                    f"   🎯 Vocabulary complexity: {domain_analysis.vocabulary_complexity:.2f}"
                )

            except Exception as e:
                domain_time = time.time() - domain_start
                errors.append(f"Domain analysis failed: {e}")
                agent_metrics["domain_intelligence"] = {
                    "processing_time": domain_time,
                    "success": False,
                    "error": str(e),
                }
                domain_analysis = None
                warnings.append("Continuing without domain analysis")

            return UniversalWorkflowResult(
                success=len(errors) == 0,
                domain_analysis=domain_analysis,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Workflow orchestration failed: {e}")
            return UniversalWorkflowResult(
                success=False,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                errors=errors,
                warnings=warnings,
            )

    async def process_knowledge_extraction_workflow(
        self, content: str, use_domain_analysis: bool = True
    ) -> UniversalWorkflowResult:
        """
        Demonstrate multi-agent coordination: Domain Intelligence → Knowledge Extraction.
        """
        start_time = time.time()
        deps = await self._ensure_deps()
        errors = []
        warnings = []
        agent_metrics = {}

        try:
            print("📚 Multi-Agent Knowledge Extraction Workflow")

            # Step 1: Domain analysis first (proper agent delegation)
            domain_analysis = None
            if use_domain_analysis:
                print("🌍 Step 1: Domain Intelligence Agent...")
                domain_start = time.time()
                try:
                    domain_analysis = await run_domain_analysis(content, detailed=True)
                    domain_time = time.time() - domain_start
                    agent_metrics["domain_intelligence"] = {
                        "processing_time": domain_time,
                        "success": True,
                        "patterns_discovered": len(domain_analysis.discovered_patterns),
                    }
                    print(
                        f"   ✅ Analysis completed: {domain_analysis.content_signature}"
                    )
                except Exception as e:
                    domain_time = time.time() - domain_start
                    errors.append(f"Domain analysis failed: {e}")
                    agent_metrics["domain_intelligence"] = {
                        "processing_time": domain_time,
                        "success": False,
                        "error": str(e),
                    }
                    warnings.append("Proceeding without domain analysis")

            # Step 2: Knowledge extraction (with domain analysis delegation)
            print("🔬 Step 2: Knowledge Extraction Agent...")
            extraction_start = time.time()
            try:
                extraction_result = await run_knowledge_extraction(
                    content, use_domain_analysis
                )
                extraction_time = time.time() - extraction_start
                agent_metrics["knowledge_extraction"] = {
                    "processing_time": extraction_time,
                    "success": True,
                    "entities_found": len(extraction_result.entities),
                    "relationships_found": len(extraction_result.relationships),
                    "extraction_confidence": extraction_result.extraction_confidence,
                }

                extraction_summary = {
                    "entities_count": len(extraction_result.entities),
                    "relationships_count": len(extraction_result.relationships),
                    "confidence": extraction_result.extraction_confidence,
                    "processing_signature": extraction_result.processing_signature,
                    "top_entities": [e.text for e in extraction_result.entities[:5]],
                }

                print(f"   ✅ Extraction completed in {extraction_time:.2f}s")
                print(
                    f"   📊 Found {len(extraction_result.entities)} entities, {len(extraction_result.relationships)} relationships"
                )

            except Exception as e:
                extraction_time = time.time() - extraction_start
                errors.append(f"Knowledge extraction failed: {e}")
                agent_metrics["knowledge_extraction"] = {
                    "processing_time": extraction_time,
                    "success": False,
                    "error": str(e),
                }
                extraction_summary = None

            return UniversalWorkflowResult(
                success=len(errors) == 0,
                domain_analysis=domain_analysis,
                extraction_summary=extraction_summary,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Knowledge extraction workflow failed: {e}")
            return UniversalWorkflowResult(
                success=False,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                errors=errors,
                warnings=warnings,
            )

    async def process_full_search_workflow(
        self, query: str, max_results: int = 10, use_domain_analysis: bool = True
    ) -> UniversalWorkflowResult:
        """
        Demonstrate complete multi-agent workflow: Domain Intelligence → Universal Search.
        """
        start_time = time.time()
        deps = await self._ensure_deps()
        errors = []
        warnings = []
        agent_metrics = {}

        try:
            print("🔍 Full Multi-Modal Search Workflow")

            # Step 1: Domain analysis of query
            domain_analysis = None
            if use_domain_analysis:
                print("🌍 Step 1: Query analysis with Domain Intelligence Agent...")
                domain_start = time.time()
                try:
                    domain_analysis = await run_domain_analysis(
                        f"Search query analysis: {query}", detailed=True
                    )
                    domain_time = time.time() - domain_start
                    agent_metrics["domain_intelligence"] = {
                        "processing_time": domain_time,
                        "success": True,
                        "query_signature": domain_analysis.content_signature,
                    }
                    print(
                        f"   ✅ Query analysis completed: {domain_analysis.content_signature}"
                    )
                except Exception as e:
                    domain_time = time.time() - domain_start
                    errors.append(f"Query analysis failed: {e}")
                    agent_metrics["domain_intelligence"] = {
                        "processing_time": domain_time,
                        "success": False,
                        "error": str(e),
                    }
                    warnings.append("Proceeding without query analysis")

            # Step 2: Multi-modal search
            print("🎯 Step 2: Universal Search Agent...")
            search_start = time.time()
            try:
                search_result = await run_universal_search(
                    query, max_results, use_domain_analysis
                )
                search_time = time.time() - search_start
                agent_metrics["universal_search"] = {
                    "processing_time": search_time,
                    "success": True,
                    "total_results": search_result.total_results_found,
                    "search_confidence": search_result.search_confidence,
                    "strategy_used": search_result.search_strategy_used,
                }

                print(f"   ✅ Search completed in {search_time:.2f}s")
                print(
                    f"   📊 Found {search_result.total_results_found} results with {search_result.search_confidence:.2f} confidence"
                )
                print(f"   🔧 Strategy: {search_result.search_strategy_used}")

            except Exception as e:
                search_time = time.time() - search_start
                errors.append(f"Universal search failed: {e}")
                agent_metrics["universal_search"] = {
                    "processing_time": search_time,
                    "success": False,
                    "error": str(e),
                }
                search_result = None

            return UniversalWorkflowResult(
                success=len(errors) == 0,
                domain_analysis=domain_analysis,
                search_results=search_result.unified_results if search_result else None,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(f"Search workflow failed: {e}")
            return UniversalWorkflowResult(
                success=False,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                errors=errors,
                warnings=warnings,
            )


# Example usage demonstrating proper PydanticAI multi-agent coordination
async def main():
    """Demonstrate universal multi-agent orchestration with proper PydanticAI patterns."""
    orchestrator = UniversalOrchestrator()

    # Sample content for demonstration
    sample_content = """
    Azure Cosmos DB is a globally distributed database service that supports multiple data models
    including document, key-value, graph, and column-family. It provides comprehensive SLAs
    for throughput, availability, latency, and consistency. The service integrates with Azure Functions
    and Azure App Service for building scalable applications.
    """

    sample_query = (
        "Azure Cosmos DB performance optimization and distributed database architecture"
    )

    print("🚀 Universal RAG Multi-Agent Orchestration Demo")
    print("=" * 60)

    # Demo 1: Domain analysis workflow
    print("\n📊 Demo 1: Domain Analysis Workflow")
    print("-" * 40)
    result1 = await orchestrator.process_content_with_domain_analysis(sample_content)
    print(f"Success: {result1.success}")
    print(f"Processing time: {result1.total_processing_time:.2f}s")
    if result1.domain_analysis:
        print(f"Content signature: {result1.domain_analysis.content_signature}")

    # Demo 2: Knowledge extraction workflow
    print("\n📚 Demo 2: Multi-Agent Knowledge Extraction")
    print("-" * 50)
    result2 = await orchestrator.process_knowledge_extraction_workflow(sample_content)
    print(f"Success: {result2.success}")
    print(f"Processing time: {result2.total_processing_time:.2f}s")
    if result2.extraction_summary:
        print(f"Entities: {result2.extraction_summary['entities_count']}")
        print(f"Relationships: {result2.extraction_summary['relationships_count']}")

    # Demo 3: Full search workflow
    print("\n🔍 Demo 3: Multi-Modal Search Workflow")
    print("-" * 45)
    result3 = await orchestrator.process_full_search_workflow(
        sample_query, max_results=5
    )
    print(f"Success: {result3.success}")
    print(f"Processing time: {result3.total_processing_time:.2f}s")
    if result3.search_results:
        print(f"Search results found: {len(result3.search_results)}")

    # Summary
    print("\n✅ Multi-Agent Orchestration Summary")
    print("=" * 40)
    all_results = [result1, result2, result3]
    successful_workflows = sum(1 for r in all_results if r.success)
    total_time = sum(r.total_processing_time for r in all_results)

    print(f"Successful workflows: {successful_workflows}/3")
    print(f"Total processing time: {total_time:.2f}s")

    # Show agent metrics
    for i, result in enumerate(all_results, 1):
        if result.agent_metrics:
            print(f"\nDemo {i} Agent Metrics:")
            for agent_name, metrics in result.agent_metrics.items():
                status = "✅" if metrics.get("success", False) else "❌"
                time_str = f"{metrics.get('processing_time', 0):.2f}s"
                print(f"  {status} {agent_name}: {time_str}")


# Export classes for use in other modules
__all__ = ["UniversalOrchestrator", "UniversalWorkflowResult"]


if __name__ == "__main__":
    asyncio.run(main())
