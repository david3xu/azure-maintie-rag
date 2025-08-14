"""
Universal RAG Orchestrator - Proper PydanticAI Multi-Agent Coordination
========================================================================

Orchestrates the Azure Universal RAG multi-agent system using proper PydanticAI patterns.
Demonstrates agent delegation, dependency sharing, and workflow coordination.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.core.universal_models import SearchResult, UniversalDomainAnalysis
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.universal_search.agent import run_universal_search
from infrastructure.utilities.azure_cost_tracker import AzureServiceCostTracker
from infrastructure.utilities.workflow_evidence_collector import (
    AzureDataWorkflowEvidenceCollector,
)


class UniversalWorkflowResult(BaseModel):
    """Results from universal multi-agent workflow."""

    success: bool
    domain_analysis: Optional[UniversalDomainAnalysis] = None
    extraction_summary: Optional[Dict[str, Any]] = None
    search_results: Optional[List[SearchResult]] = None
    total_processing_time: float = Field(ge=0.0)
    agent_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cost_summary: Dict[str, Any] = Field(default_factory=dict)  # Added cost tracking
    evidence_report: Optional[Dict[str, Any]] = Field(
        default=None, description="Workflow evidence audit trail"
    )
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
        self.cost_tracker = AzureServiceCostTracker()  # Added cost tracking
        self.evidence_collector: Optional[AzureDataWorkflowEvidenceCollector] = None

        # Centralized domain analysis cache to avoid redundant Agent 1 calls
        self._domain_analysis_cache: Optional[UniversalDomainAnalysis] = None
        self._cache_content_hash: Optional[str] = None

    async def _ensure_deps(self) -> UniversalDeps:
        """Ensure dependencies are initialized."""
        if self._deps is None:
            self._deps = await get_universal_deps()
        return self._deps

    async def _get_or_create_domain_analysis(
        self,
        content: str,
        use_domain_analysis: bool = True,
        content_type: str = "content",
    ) -> Optional[UniversalDomainAnalysis]:
        """
        Get domain analysis from cache or create new one.
        This ensures Agent 1 is called only ONCE per content in the orchestrator.
        """
        if not use_domain_analysis:
            return None

        # Simple content hash for caching
        import hashlib

        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Return cached result if same content
        if (
            self._domain_analysis_cache is not None
            and self._cache_content_hash == content_hash
        ):
            print(f"🔄 Using cached domain analysis for {content_type}")
            return self._domain_analysis_cache

        # Call Agent 1 only once and cache result
        print(f"🌍 Calling Agent 1 for {content_type} domain analysis...")
        domain_start = time.time()
        try:
            domain_analysis = await run_domain_analysis(content, detailed=True)
            domain_time = time.time() - domain_start

            # Cache the result
            self._domain_analysis_cache = domain_analysis
            self._cache_content_hash = content_hash

            print(f"   ✅ Domain analysis completed in {domain_time:.2f}s")
            print(f"   📊 Content signature: {domain_analysis.content_signature}")
            print(f"   🔄 Result cached for reuse")

            return domain_analysis

        except Exception as e:
            print(f"   ❌ Domain analysis failed: {e}")
            return None

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

            # Step 1: Domain Intelligence Agent (use centralized method)
            domain_start = time.time()
            domain_analysis = await self._get_or_create_domain_analysis(
                content, use_domain_analysis=True, content_type="content"
            )
            domain_time = time.time() - domain_start

            if domain_analysis:
                agent_metrics["domain_intelligence"] = {
                    "processing_time": domain_time,
                    "success": True,
                    "content_signature": domain_analysis.content_signature,
                    "vocabulary_complexity": domain_analysis.characteristics.vocabulary_complexity,
                    "concept_density": domain_analysis.characteristics.concept_density,
                }
                print(
                    f"   🎯 Vocabulary complexity: {domain_analysis.characteristics.vocabulary_complexity:.2f}"
                )
            else:
                agent_metrics["domain_intelligence"] = {
                    "processing_time": domain_time,
                    "success": False,
                    "error": "Domain analysis failed",
                }
                errors.append("Domain analysis failed")
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

        # Initialize evidence collector for audit trail
        workflow_id = f"knowledge_extraction_{int(start_time)}"
        self.evidence_collector = AzureDataWorkflowEvidenceCollector(workflow_id)

        try:
            print("📚 Multi-Agent Knowledge Extraction Workflow")

            # Step 1: Domain analysis first (use centralized method)
            print("🌍 Step 1: Domain Intelligence Agent...")
            domain_start = time.time()
            domain_analysis = await self._get_or_create_domain_analysis(
                content, use_domain_analysis=use_domain_analysis, content_type="content"
            )
            domain_time = time.time() - domain_start

            if domain_analysis:
                agent_metrics["domain_intelligence"] = {
                    "processing_time": domain_time,
                    "success": True,
                    "patterns_discovered": len(
                        getattr(domain_analysis, "discovered_patterns", [])
                    ),
                }
            else:
                agent_metrics["domain_intelligence"] = {
                    "processing_time": domain_time,
                    "success": False,
                    "error": "Domain analysis failed or disabled",
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

            # Calculate cost estimates for the workflow
            cost_summary = self._calculate_workflow_costs(
                agent_metrics, extraction_summary, domain_analysis
            )

            # Generate evidence report for audit trail
            evidence_report = None
            if self.evidence_collector:
                try:
                    # Record workflow completion evidence
                    await self.evidence_collector.record_azure_service_evidence(
                        step_number=99,  # Final step
                        azure_service="workflow_orchestrator",
                        operation_type="knowledge_extraction_workflow",
                        input_data={
                            "content_length": len(content),
                            "use_domain_analysis": use_domain_analysis,
                        },
                        output_data={
                            "entities_count": (
                                extraction_summary.get("entities_count", 0)
                                if extraction_summary
                                else 0
                            ),
                            "relationships_count": (
                                extraction_summary.get("relationships_count", 0)
                                if extraction_summary
                                else 0
                            ),
                            "success": len(errors) == 0,
                        },
                        processing_time_ms=(time.time() - start_time) * 1000,
                        quality_metrics={
                            "workflow_success_rate": 1.0 if len(errors) == 0 else 0.0,
                            "agent_success_count": sum(
                                1
                                for metrics in agent_metrics.values()
                                if metrics.get("success", False)
                            ),
                        },
                    )

                    evidence_report = (
                        await self.evidence_collector.generate_workflow_evidence_report()
                    )
                    logger.info(f"Evidence report generated for workflow {workflow_id}")
                except Exception as e:
                    logger.warning(f"Evidence report generation failed: {e}")

            return UniversalWorkflowResult(
                success=len(errors) == 0,
                domain_analysis=domain_analysis,
                extraction_summary=extraction_summary,
                total_processing_time=time.time() - start_time,
                agent_metrics=agent_metrics,
                cost_summary=cost_summary,
                evidence_report=evidence_report,
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

            # PRODUCTION OPTIMIZATION: Skip Agent 1 for user queries
            # Agent 1 should only run during data ingestion to build the database
            # User queries search the pre-analyzed data directly for fast responses
            print("⚡ Step 1: Using pre-analyzed data (Agent 1 skipped for production speed)...")
            domain_start = time.time()
            domain_analysis = None  # Skip query analysis for production performance
            domain_time = time.time() - domain_start

            agent_metrics["domain_intelligence"] = {
                "processing_time": domain_time,
                "success": True,
                "optimization": "skipped_for_production_speed",
                "note": "Agent 1 runs during data ingestion, not user queries",
            }

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

    def _calculate_workflow_costs(
        self,
        agent_metrics: Dict[str, Dict[str, Any]],
        extraction_summary: Optional[Dict[str, Any]],
        domain_analysis: Optional[UniversalDomainAnalysis],
    ) -> Dict[str, Any]:
        """
        Calculate estimated costs for the workflow based on agent usage.

        Provides enterprise-level cost monitoring and budgeting capabilities.
        """
        total_cost = 0.0
        cost_breakdown = {}

        try:
            # Calculate domain analysis costs (Azure OpenAI usage)
            if "domain_intelligence" in agent_metrics:
                domain_cost = self.cost_tracker.estimate_operation_cost(
                    "azure_openai", "request", 1  # One analysis request
                )
                cost_breakdown["domain_analysis"] = domain_cost
                total_cost += domain_cost

            # Calculate knowledge extraction costs
            if "knowledge_extraction" in agent_metrics and extraction_summary:
                entities_count = extraction_summary.get("entities_count", 0)
                relationships_count = extraction_summary.get("relationships_count", 0)

                # Estimate based on entities and relationships processed
                extraction_requests = max(
                    1, (entities_count + relationships_count) // 10
                )
                extraction_cost = self.cost_tracker.estimate_operation_cost(
                    "azure_openai", "request", extraction_requests
                )

                # Add Cosmos DB costs for graph storage
                cosmos_cost = self.cost_tracker.estimate_operation_cost(
                    "cosmos_db", "operation", entities_count + relationships_count
                )

                cost_breakdown["knowledge_extraction"] = extraction_cost
                cost_breakdown["graph_storage"] = cosmos_cost
                total_cost += extraction_cost + cosmos_cost

            # Calculate search costs if applicable
            if "universal_search" in agent_metrics:
                search_cost = self.cost_tracker.estimate_operation_cost(
                    "cognitive_search", "query", 1
                )
                cost_breakdown["universal_search"] = search_cost
                total_cost += search_cost

            return {
                "total_estimated_cost_usd": round(total_cost, 6),
                "cost_breakdown": cost_breakdown,
                "cost_tracking_enabled": True,
                "cost_estimation_method": "azure_service_cost_tracker",
                "currency": "USD",
                "estimated_at": time.time(),
            }

        except Exception as e:
            return {
                "total_estimated_cost_usd": 0.0,
                "cost_breakdown": {},
                "cost_tracking_enabled": False,
                "error": f"Cost calculation failed: {e}",
                "estimated_at": time.time(),
            }


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
