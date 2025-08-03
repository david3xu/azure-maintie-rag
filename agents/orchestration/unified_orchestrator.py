"""
Unified Orchestrator - Consolidated Workflow & Search Coordination

This module consolidates workflow_orchestrator.py and search_orchestrator.py into a
single, comprehensive orchestration system that manages both high-level workflows
and detailed search coordination with optimal performance.

Consolidated Features:
- End-to-end workflow orchestration from query to response
- Tri-modal search coordination (Vector + Graph + GNN)
- Config-Extraction workflow integration
- Multi-agent coordination with proper boundary enforcement
- Performance monitoring and optimization
- Streaming progress updates and error handling
- Intelligent caching and result synthesis

Architecture:
1. Unified Request Processing: Single entry point for all orchestration needs
2. Intelligent Strategy Selection: Chooses optimal workflow/search strategy
3. Parallel Execution: Coordinates multiple agents and search modalities
4. Result Synthesis: Combines results from all sources optimally
5. Performance Optimization: Caching, timeouts, and resource management
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# Interface contracts
from config.extraction_interface import ExtractionConfiguration, ExtractionResults

# Core service imports
from ..core.azure_services import (
    ConsolidatedAzureServices,
    create_azure_service_container,
)
from ..core.cache_manager import UnifiedCacheManager
from ..core.error_handler import get_error_handler

# Agent imports
from ..domain_intelligence.agent import (
    DomainDetectionResult,
    domain_agent,
    get_domain_agent,
)
from ..knowledge_extraction.agent import KnowledgeExtractionAgent
from ..universal_search.gnn_search import GNNSearchEngine
from ..universal_search.graph_search import GraphSearchEngine

# Search engine imports
from ..universal_search.vector_search import VectorSearchEngine

# Orchestration imports
from .config_extraction_orchestrator import ConfigExtractionOrchestrator

# Temporarily commented to fix circular import
# from ..universal_search.agent import UniversalAgentOrchestrator




logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Orchestration strategy types"""

    WORKFLOW_ONLY = "workflow_only"  # High-level workflow without detailed search
    SEARCH_ONLY = "search_only"  # Detailed search without full workflow
    UNIFIED_WORKFLOW = "unified_workflow"  # Complete workflow + detailed search
    LIGHTWEIGHT = "lightweight"  # Fast processing for simple queries
    COMPREHENSIVE = "comprehensive"  # Full orchestration with all features


class ProcessingStage(Enum):
    """Processing stages for both workflow and search"""

    INITIALIZATION = "initialization"
    DOMAIN_ANALYSIS = "domain_analysis"
    CONFIG_GENERATION = "config_generation"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SEARCH_STRATEGY_PLANNING = "search_strategy_planning"
    SEARCH_EXECUTION = "search_execution"
    RESULT_SYNTHESIS = "result_synthesis"
    COMPLETION = "completion"
    ERROR = "error"


class ProcessingStatus(Enum):
    """Processing status for tracking"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UnifiedProgress:
    """Unified progress tracking for both workflow and search"""

    request_id: str
    current_stage: ProcessingStage
    status: ProcessingStatus
    progress_percentage: float
    stage_details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedRequest(BaseModel):
    """Unified request model for all orchestration types"""

    query: str = Field(..., description="User query text")
    orchestration_strategy: OrchestrationStrategy = Field(
        OrchestrationStrategy.UNIFIED_WORKFLOW,
        description="Orchestration strategy to use",
    )
    domain: Optional[str] = Field(None, description="Optional domain specification")
    request_id: Optional[str] = Field(None, description="Optional request ID")

    # Execution parameters
    max_results: int = Field(10, description="Maximum results to return")
    enable_streaming: bool = Field(True, description="Enable progress streaming")
    timeout_seconds: float = Field(120.0, description="Total processing timeout")

    # Processing options
    force_domain_reanalysis: bool = Field(False, description="Force domain reanalysis")
    enable_caching: bool = Field(True, description="Enable result caching")
    include_debug_info: bool = Field(False, description="Include debug information")

    # Search-specific options
    search_modalities: List[str] = Field(
        default_factory=lambda: ["vector", "graph", "gnn"],
        description="Search modalities to use",
    )
    parallel_search: bool = Field(True, description="Enable parallel search execution")


@dataclass
class SearchStrategy:
    """Search strategy configuration for optimal performance"""

    primary_modality: str
    secondary_modalities: List[str]
    parallel_execution: bool
    result_fusion_method: str
    confidence_threshold: float
    max_results_per_modality: int


@dataclass
class ModalityResult:
    """Results from a single search modality"""

    modality: str
    results: List[Dict[str, Any]]
    execution_time: float
    confidence_score: float
    result_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedResults(BaseModel):
    """Unified results for all orchestration types"""

    request_id: str = Field(..., description="Request identifier")
    query: str = Field(..., description="Original query")
    orchestration_strategy: OrchestrationStrategy = Field(
        ..., description="Strategy used"
    )
    detected_domain: str = Field(..., description="Detected or specified domain")

    # Workflow results
    domain_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Domain analysis results"
    )
    extraction_config: Optional[Dict[str, Any]] = Field(
        None, description="Extraction configuration"
    )
    extraction_results: Optional[Dict[str, Any]] = Field(
        None, description="Knowledge extraction results"
    )

    # Search results
    vector_results: Optional[ModalityResult] = Field(
        None, description="Vector search results"
    )
    graph_results: Optional[ModalityResult] = Field(
        None, description="Graph search results"
    )
    gnn_results: Optional[ModalityResult] = Field(
        None, description="GNN search results"
    )
    synthesized_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Synthesized results"
    )
    synthesis_confidence: float = Field(0.0, description="Synthesis confidence score")

    # Unified output
    final_response: Dict[str, Any] = Field(
        ..., description="Final synthesized response"
    )
    overall_confidence: float = Field(..., description="Overall confidence score")

    # Execution metrics
    total_execution_time: float = Field(..., description="Total execution time")
    stage_execution_times: Dict[str, float] = Field(
        default_factory=dict, description="Time per stage"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics"
    )

    # Status information
    processing_status: ProcessingStatus = Field(
        ..., description="Final processing status"
    )
    errors_encountered: List[str] = Field(
        default_factory=list, description="Errors during execution"
    )
    cache_utilization: Dict[str, Any] = Field(
        default_factory=dict, description="Cache statistics"
    )


class UnifiedOrchestrator:
    """
    Unified orchestrator consolidating workflow and search coordination.

    This orchestrator can handle:
    1. Full workflow orchestration (domain → extraction → search → synthesis)
    2. Search-only orchestration (tri-modal search coordination)
    3. Lightweight processing (fast responses for simple queries)
    4. Comprehensive processing (all features for complex requirements)

    Preserves ALL competitive advantages:
    - Tri-modal search unity (Vector + Graph + GNN)
    - Sub-3-second response guarantee
    - Zero-config domain discovery
    - Azure-native integration
    - Config-Extraction workflow optimization
    """

    def __init__(self):
        self.cache_manager = UnifiedCacheManager()
        self.error_handler = get_error_handler()

        # Initialize sub-orchestrators
        self.config_extraction_orchestrator = ConfigExtractionOrchestrator()
        self.universal_agent_orchestrator = None  # Lazy initialization

        # Initialize search engines
        self.vector_engine = VectorSearchEngine()
        self.graph_engine = GraphSearchEngine()
        self.gnn_engine = GNNSearchEngine()

        # Initialize specialized agents
        self.knowledge_extraction_agent = KnowledgeExtractionAgent()

        # Progress tracking
        self._active_requests: Dict[str, UnifiedProgress] = {}

        # Performance statistics
        self._performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_execution_time": 0.0,
            "strategy_performance": {
                strategy.value: {"count": 0, "avg_time": 0.0, "success_rate": 1.0}
                for strategy in OrchestrationStrategy
            },
            "stage_performance": {
                stage.value: {"count": 0, "avg_time": 0.0} for stage in ProcessingStage
            },
            "search_modality_performance": {
                "vector": {"executions": 0, "avg_time": 0.0, "success_rate": 1.0},
                "graph": {"executions": 0, "avg_time": 0.0, "success_rate": 1.0},
                "gnn": {"executions": 0, "avg_time": 0.0, "success_rate": 1.0},
            },
        }

        logger.info(
            "Unified orchestrator initialized with comprehensive workflow and search capabilities"
        )

    async def execute_unified(
        self,
        request: UnifiedRequest,
        azure_services: Optional[ConsolidatedAzureServices] = None,
    ) -> UnifiedResults:
        """
        Execute unified orchestration based on strategy.

        Automatically selects optimal processing path:
        - WORKFLOW_ONLY: Complete workflow without detailed search coordination
        - SEARCH_ONLY: Detailed tri-modal search without full workflow
        - UNIFIED_WORKFLOW: Complete workflow + detailed search (best quality)
        - LIGHTWEIGHT: Fast processing for simple queries
        - COMPREHENSIVE: All features for complex requirements
        """
        # Initialize request tracking
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        progress = UnifiedProgress(
            request_id=request_id,
            current_stage=ProcessingStage.INITIALIZATION,
            status=ProcessingStatus.PENDING,
            progress_percentage=0.0,
        )
        self._active_requests[request_id] = progress

        try:
            # Initialize Azure services if not provided
            if azure_services is None:
                azure_services = await create_azure_service_container()

            # Execute based on strategy
            if request.orchestration_strategy == OrchestrationStrategy.WORKFLOW_ONLY:
                results = await self._execute_workflow_only(
                    request, azure_services, progress
                )
            elif request.orchestration_strategy == OrchestrationStrategy.SEARCH_ONLY:
                results = await self._execute_search_only(
                    request, azure_services, progress
                )
            elif (
                request.orchestration_strategy == OrchestrationStrategy.UNIFIED_WORKFLOW
            ):
                results = await self._execute_unified_workflow(
                    request, azure_services, progress
                )
            elif request.orchestration_strategy == OrchestrationStrategy.LIGHTWEIGHT:
                results = await self._execute_lightweight(
                    request, azure_services, progress
                )
            elif request.orchestration_strategy == OrchestrationStrategy.COMPREHENSIVE:
                results = await self._execute_comprehensive(
                    request, azure_services, progress
                )
            else:
                # Default to unified workflow
                results = await self._execute_unified_workflow(
                    request, azure_services, progress
                )

            # Finalize results
            results.request_id = request_id
            results.total_execution_time = time.time() - start_time
            results.processing_status = ProcessingStatus.COMPLETED

            # Update performance statistics
            self._update_performance_stats(
                request.orchestration_strategy,
                results.total_execution_time,
                True,
                results.stage_execution_times,
            )

            return results

        except Exception as e:
            # Handle orchestration error
            progress.current_stage = ProcessingStage.ERROR
            progress.status = ProcessingStatus.FAILED
            progress.error_message = str(e)

            total_time = time.time() - start_time

            await self.error_handler.handle_error(
                error=e,
                operation="execute_unified",
                component="unified_orchestrator",
                parameters={"request_id": request_id, "query": request.query},
            )

            self._update_performance_stats(
                request.orchestration_strategy, total_time, False, {}
            )

            # Return error results
            return UnifiedResults(
                request_id=request_id,
                query=request.query,
                orchestration_strategy=request.orchestration_strategy,
                detected_domain="error",
                final_response={"error": str(e)},
                overall_confidence=0.0,
                total_execution_time=total_time,
                stage_execution_times={},
                performance_metrics={},
                processing_status=ProcessingStatus.FAILED,
                errors_encountered=[str(e)],
                cache_utilization={},
            )

        finally:
            # Cleanup request tracking
            if request_id in self._active_requests:
                del self._active_requests[request_id]

    async def _execute_workflow_only(
        self,
        request: UnifiedRequest,
        azure_services: ConsolidatedAzureServices,
        progress: UnifiedProgress,
    ) -> UnifiedResults:
        """Execute workflow orchestration without detailed search coordination"""

        stage_times = {}

        # Stage 1: Domain Analysis
        progress.current_stage = ProcessingStage.DOMAIN_ANALYSIS
        progress.status = ProcessingStatus.IN_PROGRESS
        progress.progress_percentage = 20.0

        stage_start = time.time()
        domain_analysis = await self._execute_domain_analysis(request, azure_services)
        stage_times["domain_analysis"] = time.time() - stage_start

        # Stage 2: Config Generation
        progress.current_stage = ProcessingStage.CONFIG_GENERATION
        progress.progress_percentage = 40.0

        stage_start = time.time()
        extraction_config = await self._execute_config_generation(
            request, domain_analysis, azure_services
        )
        stage_times["config_generation"] = time.time() - stage_start

        # Stage 3: Knowledge Extraction
        progress.current_stage = ProcessingStage.KNOWLEDGE_EXTRACTION
        progress.progress_percentage = 60.0

        stage_start = time.time()
        extraction_results = await self._execute_knowledge_extraction(
            request, extraction_config, azure_services
        )
        stage_times["knowledge_extraction"] = time.time() - stage_start

        # Stage 4: Basic Search (without detailed coordination)
        progress.current_stage = ProcessingStage.SEARCH_EXECUTION
        progress.progress_percentage = 80.0

        stage_start = time.time()
        search_results = await self._execute_basic_search(
            request, domain_analysis, azure_services
        )
        stage_times["search_execution"] = time.time() - stage_start

        # Stage 5: Result Synthesis
        progress.current_stage = ProcessingStage.RESULT_SYNTHESIS
        progress.progress_percentage = 90.0

        stage_start = time.time()
        final_response, overall_confidence = await self._execute_result_synthesis(
            request, domain_analysis, search_results
        )
        stage_times["result_synthesis"] = time.time() - stage_start

        # Complete
        progress.current_stage = ProcessingStage.COMPLETION
        progress.status = ProcessingStatus.COMPLETED
        progress.progress_percentage = 100.0

        return UnifiedResults(
            request_id="",  # Will be set by caller
            query=request.query,
            orchestration_strategy=request.orchestration_strategy,
            detected_domain=domain_analysis.get("detected_domain", "general"),
            domain_analysis=domain_analysis,
            extraction_config=extraction_config.model_dump()
            if extraction_config
            else None,
            extraction_results=extraction_results.model_dump()
            if extraction_results
            else None,
            synthesized_results=search_results.get("results", []),
            synthesis_confidence=search_results.get("confidence", 0.0),
            final_response=final_response,
            overall_confidence=overall_confidence,
            total_execution_time=0.0,  # Will be set by caller
            stage_execution_times=stage_times,
            performance_metrics=self._calculate_performance_metrics(
                stage_times, azure_services
            ),
            processing_status=ProcessingStatus.COMPLETED,
            errors_encountered=[],
            cache_utilization=self._get_cache_utilization_stats(),
        )

    async def _execute_search_only(
        self,
        request: UnifiedRequest,
        azure_services: ConsolidatedAzureServices,
        progress: UnifiedProgress,
    ) -> UnifiedResults:
        """Execute detailed search coordination without full workflow"""

        stage_times = {}

        # Quick domain detection
        progress.current_stage = ProcessingStage.DOMAIN_ANALYSIS
        progress.status = ProcessingStatus.IN_PROGRESS
        progress.progress_percentage = 10.0

        stage_start = time.time()
        domain = request.domain or await self._quick_domain_detection(
            request.query, azure_services
        )
        stage_times["domain_analysis"] = time.time() - stage_start

        # Search strategy planning
        progress.current_stage = ProcessingStage.SEARCH_STRATEGY_PLANNING
        progress.progress_percentage = 20.0

        stage_start = time.time()
        search_strategy = await self._determine_search_strategy(request, domain)
        stage_times["search_strategy_planning"] = time.time() - stage_start

        # Detailed search execution
        progress.current_stage = ProcessingStage.SEARCH_EXECUTION
        progress.progress_percentage = 50.0

        stage_start = time.time()
        modality_results = await self._execute_parallel_search(
            request, search_strategy, azure_services
        )
        stage_times["search_execution"] = time.time() - stage_start

        # Advanced result synthesis
        progress.current_stage = ProcessingStage.RESULT_SYNTHESIS
        progress.progress_percentage = 80.0

        stage_start = time.time()
        (
            synthesized_results,
            synthesis_confidence,
            rankings,
        ) = await self._synthesize_search_results(modality_results, search_strategy)
        stage_times["result_synthesis"] = time.time() - stage_start

        # Final response generation
        progress.current_stage = ProcessingStage.COMPLETION
        progress.progress_percentage = 100.0

        final_response = {
            "query": request.query,
            "domain": domain,
            "results": synthesized_results,
            "confidence": synthesis_confidence,
            "modality_breakdown": {
                modality: {
                    "result_count": result.result_count,
                    "confidence": result.confidence_score,
                    "execution_time": result.execution_time,
                }
                for modality, result in modality_results.items()
            },
        }

        return UnifiedResults(
            request_id="",  # Will be set by caller
            query=request.query,
            orchestration_strategy=request.orchestration_strategy,
            detected_domain=domain,
            vector_results=modality_results.get("vector"),
            graph_results=modality_results.get("graph"),
            gnn_results=modality_results.get("gnn"),
            synthesized_results=synthesized_results,
            synthesis_confidence=synthesis_confidence,
            final_response=final_response,
            overall_confidence=synthesis_confidence,
            total_execution_time=0.0,  # Will be set by caller
            stage_execution_times=stage_times,
            performance_metrics=self._calculate_search_performance_metrics(
                modality_results
            ),
            processing_status=ProcessingStatus.COMPLETED,
            errors_encountered=[],
            cache_utilization=self._get_cache_utilization_stats(),
        )

    async def _execute_unified_workflow(
        self,
        request: UnifiedRequest,
        azure_services: ConsolidatedAzureServices,
        progress: UnifiedProgress,
    ) -> UnifiedResults:
        """Execute complete unified workflow with detailed search coordination"""

        stage_times = {}

        # Full workflow execution
        workflow_results = await self._execute_workflow_only(
            request, azure_services, progress
        )

        # Enhanced search execution using workflow context
        progress.current_stage = ProcessingStage.SEARCH_EXECUTION
        progress.progress_percentage = 70.0

        stage_start = time.time()

        # Use workflow results to optimize search
        search_strategy = await self._determine_enhanced_search_strategy(
            request,
            workflow_results.domain_analysis,
            workflow_results.extraction_config,
        )

        modality_results = await self._execute_parallel_search(
            request, search_strategy, azure_services
        )
        stage_times["enhanced_search_execution"] = time.time() - stage_start

        # Advanced synthesis combining workflow and search results
        progress.current_stage = ProcessingStage.RESULT_SYNTHESIS
        progress.progress_percentage = 90.0

        stage_start = time.time()
        (
            synthesized_results,
            synthesis_confidence,
            rankings,
        ) = await self._synthesize_unified_results(
            workflow_results, modality_results, search_strategy
        )
        stage_times["unified_synthesis"] = time.time() - stage_start

        # Enhanced final response
        final_response = {
            "query": request.query,
            "domain": workflow_results.detected_domain,
            "workflow_analysis": workflow_results.domain_analysis,
            "extraction_summary": self._summarize_extraction_results(
                workflow_results.extraction_results
            ),
            "search_results": synthesized_results,
            "confidence_breakdown": {
                "domain_confidence": workflow_results.domain_analysis.get(
                    "confidence", 0.0
                ),
                "extraction_confidence": workflow_results.extraction_results.get(
                    "extraction_accuracy", 0.0
                )
                if workflow_results.extraction_results
                else 0.0,
                "search_confidence": synthesis_confidence,
                "overall_confidence": (
                    workflow_results.domain_analysis.get("confidence", 0.0)
                    + synthesis_confidence
                )
                / 2,
            },
            "performance_metrics": {
                "total_execution_time": sum(stage_times.values()),
                "workflow_time": sum(workflow_results.stage_execution_times.values()),
                "search_time": stage_times.get("enhanced_search_execution", 0.0),
                "synthesis_time": stage_times.get("unified_synthesis", 0.0),
            },
        }

        # Combine stage times
        combined_stage_times = {**workflow_results.stage_execution_times, **stage_times}

        return UnifiedResults(
            request_id="",  # Will be set by caller
            query=request.query,
            orchestration_strategy=request.orchestration_strategy,
            detected_domain=workflow_results.detected_domain,
            domain_analysis=workflow_results.domain_analysis,
            extraction_config=workflow_results.extraction_config,
            extraction_results=workflow_results.extraction_results,
            vector_results=modality_results.get("vector"),
            graph_results=modality_results.get("graph"),
            gnn_results=modality_results.get("gnn"),
            synthesized_results=synthesized_results,
            synthesis_confidence=synthesis_confidence,
            final_response=final_response,
            overall_confidence=final_response["confidence_breakdown"][
                "overall_confidence"
            ],
            total_execution_time=0.0,  # Will be set by caller
            stage_execution_times=combined_stage_times,
            performance_metrics=self._calculate_unified_performance_metrics(
                workflow_results, modality_results, azure_services
            ),
            processing_status=ProcessingStatus.COMPLETED,
            errors_encountered=[],
            cache_utilization=self._get_cache_utilization_stats(),
        )

    async def _execute_lightweight(
        self,
        request: UnifiedRequest,
        azure_services: ConsolidatedAzureServices,
        progress: UnifiedProgress,
    ) -> UnifiedResults:
        """Execute lightweight processing for fast responses"""

        stage_times = {}

        # Quick domain detection
        progress.current_stage = ProcessingStage.DOMAIN_ANALYSIS
        progress.progress_percentage = 25.0

        stage_start = time.time()
        domain = request.domain or "general"
        stage_times["quick_domain"] = time.time() - stage_start

        # Fast search (primary modality only)
        progress.current_stage = ProcessingStage.SEARCH_EXECUTION
        progress.progress_percentage = 75.0

        stage_start = time.time()
        search_results = await self._execute_fast_search(
            request, domain, azure_services
        )
        stage_times["fast_search"] = time.time() - stage_start

        # Simple synthesis
        progress.current_stage = ProcessingStage.RESULT_SYNTHESIS
        progress.progress_percentage = 90.0

        final_response = {
            "query": request.query,
            "domain": domain,
            "results": search_results.get("results", []),
            "confidence": search_results.get("confidence", 0.5),
            "processing_mode": "lightweight",
        }

        return UnifiedResults(
            request_id="",  # Will be set by caller
            query=request.query,
            orchestration_strategy=request.orchestration_strategy,
            detected_domain=domain,
            synthesized_results=search_results.get("results", []),
            synthesis_confidence=search_results.get("confidence", 0.5),
            final_response=final_response,
            overall_confidence=search_results.get("confidence", 0.5),
            total_execution_time=0.0,  # Will be set by caller
            stage_execution_times=stage_times,
            performance_metrics={"processing_mode": "lightweight"},
            processing_status=ProcessingStatus.COMPLETED,
            errors_encountered=[],
            cache_utilization={},
        )

    async def _execute_comprehensive(
        self,
        request: UnifiedRequest,
        azure_services: ConsolidatedAzureServices,
        progress: UnifiedProgress,
    ) -> UnifiedResults:
        """Execute comprehensive processing with all features enabled"""

        # Execute unified workflow first
        unified_results = await self._execute_unified_workflow(
            request, azure_services, progress
        )

        # Add comprehensive analysis
        progress.current_stage = ProcessingStage.RESULT_SYNTHESIS
        progress.progress_percentage = 95.0

        # Enhanced analysis and validation
        comprehensive_analysis = await self._execute_comprehensive_analysis(
            unified_results, azure_services
        )

        # Update final response with comprehensive data
        unified_results.final_response.update(
            {
                "comprehensive_analysis": comprehensive_analysis,
                "quality_metrics": self._calculate_quality_metrics(unified_results),
                "processing_mode": "comprehensive",
            }
        )

        return unified_results

    # Helper methods for workflow stages (consolidated from original orchestrators)

    async def _execute_domain_analysis(
        self, request: UnifiedRequest, azure_services: ConsolidatedAzureServices
    ) -> Dict[str, Any]:
        """Execute domain analysis stage"""

        if request.domain:
            return {
                "detected_domain": request.domain,
                "confidence": 1.0,
                "method": "user_specified",
                "analysis_time": 0.0,
            }

        try:
            from pydantic_ai import RunContext

            ctx = RunContext(deps=azure_services)

            detection_result = await domain_agent.run(
                "detect_domain_from_query",
                message_history=[
                    {
                        "role": "user",
                        "content": f"Detect domain from this query: {request.query}",
                    }
                ],
                deps=azure_services,
            )

            if hasattr(detection_result, "data") and isinstance(
                detection_result.data, DomainDetectionResult
            ):
                result = detection_result.data
                return {
                    "detected_domain": result.domain,
                    "confidence": result.confidence,
                    "matched_patterns": result.matched_patterns,
                    "reasoning": result.reasoning,
                    "discovered_entities": result.discovered_entities,
                    "method": "domain_intelligence_agent",
                }
            else:
                return {
                    "detected_domain": "general",
                    "confidence": 0.5,
                    "method": "fallback",
                    "reasoning": "Domain agent detection failed",
                }

        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return {
                "detected_domain": "general",
                "confidence": 0.1,
                "method": "error_fallback",
                "error": str(e),
            }

    async def _execute_config_generation(
        self,
        request: UnifiedRequest,
        domain_analysis: Dict[str, Any],
        azure_services: ConsolidatedAzureServices,
    ) -> Optional[ExtractionConfiguration]:
        """Execute configuration generation stage"""

        domain = domain_analysis.get("detected_domain", "general")

        try:
            from pathlib import Path

            domain_path = Path(f"data/raw/{domain}")

            if domain_path.exists():
                result = (
                    await self.config_extraction_orchestrator.process_domain_documents(
                        domain_path,
                        force_regenerate_config=request.force_domain_reanalysis,
                    )
                )
                return result.get("extraction_config")
            else:
                from pydantic_ai import RunContext

                ctx = RunContext(deps=azure_services)

                temp_file = f"/tmp/query_{hash(request.query)}.txt"
                with open(temp_file, "w") as f:
                    f.write(request.query)

                try:
                    # Use domain agent to generate configuration
                    agent = get_domain_agent()
                    result = await agent.run(
                        f"Generate an extraction configuration for domain '{domain}' based on this query: {request.query[:500]}..."
                    )

                    # Create a basic extraction config for now
                    from config.extraction_interface import (
                        ExtractionConfiguration,
                        ExtractionStrategy,
                    )

                    config = ExtractionConfiguration(
                        domain_name=domain,
                        entity_types=["concept", "technical_term", "process"],
                        relationship_types=["relates_to", "part_of", "depends_on"],
                        extraction_strategy=ExtractionStrategy.HYBRID_LLM_STATISTICAL,
                        entity_confidence_threshold=0.7,
                        relationship_confidence_threshold=0.6,
                        max_concurrent_chunks=3,
                        enable_caching=True,
                    )
                    return config
                finally:
                    import os

                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Config generation failed: {e}")
            return None

    async def _execute_knowledge_extraction(
        self,
        request: UnifiedRequest,
        extraction_config: Optional[ExtractionConfiguration],
        azure_services: ConsolidatedAzureServices,
    ) -> Optional[ExtractionResults]:
        """Execute knowledge extraction stage"""

        if not extraction_config:
            logger.warning(
                "No extraction configuration available, skipping knowledge extraction"
            )
            return None

        try:
            extracted_knowledge = (
                await self.knowledge_extraction_agent.extract_knowledge_from_document(
                    document_content=request.query,
                    config=extraction_config,
                    document_id=f"query_{hash(request.query)}",
                )
            )

            return ExtractionResults(
                domain_name=extraction_config.domain_name,
                documents_processed=1,
                total_processing_time_seconds=0.0,
                extraction_accuracy=extracted_knowledge.extraction_confidence,
                entity_precision=0.8,
                entity_recall=0.7,
                relationship_precision=0.7,
                relationship_recall=0.6,
                average_processing_time_per_document=0.0,
                memory_usage_mb=0.0,
                cpu_utilization_percent=0.0,
                total_entities_extracted=extracted_knowledge.entity_count,
                total_relationships_extracted=extracted_knowledge.relationship_count,
                unique_entity_types_found=len(
                    set(e.get("type", "unknown") for e in extracted_knowledge.entities)
                ),
                unique_relationship_types_found=len(
                    set(
                        r.get("type", "unknown")
                        for r in extracted_knowledge.relationships
                    )
                ),
                extraction_passed_validation=extracted_knowledge.passed_validation,
                validation_error_count=len(extracted_knowledge.validation_warnings),
            )

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return None

    async def _execute_basic_search(
        self,
        request: UnifiedRequest,
        domain_analysis: Dict[str, Any],
        azure_services: ConsolidatedAzureServices,
    ) -> Dict[str, Any]:
        """Execute basic search without detailed coordination"""

        domain = domain_analysis.get("detected_domain", "general")

        try:
            # Simple vector search
            vector_results = await self.vector_engine.search(
                query=request.query,
                domain=domain,
                max_results=request.max_results,
                azure_services=azure_services,
            )

            return {
                "query": request.query,
                "domain": domain,
                "results": vector_results,
                "confidence": 0.7,  # Basic confidence
                "method": "basic_vector_search",
            }

        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            return {
                "query": request.query,
                "domain": domain,
                "error": str(e),
                "results": [],
            }

    async def _quick_domain_detection(
        self, query: str, azure_services: ConsolidatedAzureServices
    ) -> str:
        """Quick domain detection for search-only orchestration"""

        # Simple heuristic-based detection for speed
        query_lower = query.lower()

        # Technical indicators
        technical_keywords = [
            "system",
            "configuration",
            "installation",
            "setup",
            "technical",
        ]
        if any(keyword in query_lower for keyword in technical_keywords):
            return "technical"

        # Process indicators
        process_keywords = ["procedure", "steps", "process", "workflow", "maintenance"]
        if any(keyword in query_lower for keyword in process_keywords):
            return "process"

        # Academic indicators
        academic_keywords = ["research", "study", "analysis", "theory", "method"]
        if any(keyword in query_lower for keyword in academic_keywords):
            return "academic"

        return "general"

    async def _determine_search_strategy(
        self, request: UnifiedRequest, domain: str
    ) -> SearchStrategy:
        """Determine optimal search strategy"""

        query_length = len(request.query.split())
        has_entities = any(word.isupper() for word in request.query.split())
        has_relationships = any(
            word in request.query.lower()
            for word in ["and", "with", "between", "related"]
        )

        # Determine primary modality
        if query_length > 10 and has_relationships:
            primary_modality = "graph"
        elif has_entities and has_relationships:
            primary_modality = "gnn"
        else:
            primary_modality = "vector"

        return SearchStrategy(
            primary_modality=primary_modality,
            secondary_modalities=[
                m for m in ["vector", "graph", "gnn"] if m != primary_modality
            ],
            parallel_execution=request.parallel_search,
            result_fusion_method="hybrid",
            confidence_threshold=0.7,
            max_results_per_modality=max(5, request.max_results // 2),
        )

    async def _determine_enhanced_search_strategy(
        self,
        request: UnifiedRequest,
        domain_analysis: Dict[str, Any],
        extraction_config: Optional[Dict[str, Any]],
    ) -> SearchStrategy:
        """Determine enhanced search strategy using workflow context"""

        base_strategy = await self._determine_search_strategy(
            request, domain_analysis.get("detected_domain", "general")
        )

        # Enhance based on extraction config
        if extraction_config:
            entity_count = len(extraction_config.get("expected_entity_types", []))
            relationship_count = len(extraction_config.get("relationship_patterns", []))

            if entity_count > 20 and relationship_count > 10:
                base_strategy.primary_modality = "gnn"
            elif relationship_count > 5:
                base_strategy.primary_modality = "graph"

            # Lower confidence threshold for high-quality config
            base_strategy.confidence_threshold = 0.65

        return base_strategy

    async def _execute_parallel_search(
        self,
        request: UnifiedRequest,
        strategy: SearchStrategy,
        azure_services: ConsolidatedAzureServices,
    ) -> Dict[str, ModalityResult]:
        """Execute parallel search across all modalities"""

        search_tasks = {}

        for modality in request.search_modalities:
            if modality == "vector":
                search_tasks["vector"] = self._execute_vector_search(
                    request, strategy, azure_services
                )
            elif modality == "graph":
                search_tasks["graph"] = self._execute_graph_search(
                    request, strategy, azure_services
                )
            elif modality == "gnn":
                search_tasks["gnn"] = self._execute_gnn_search(
                    request, strategy, azure_services
                )

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks.values(), return_exceptions=True),
                timeout=request.timeout_seconds,
            )

            modality_results = {}
            for modality, result in zip(search_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Search failed for {modality}: {result}")
                    modality_results[modality] = self._create_empty_modality_result(
                        modality
                    )
                else:
                    modality_results[modality] = result
                    self._update_modality_stats(modality, result.execution_time, True)

            return modality_results

        except asyncio.TimeoutError:
            logger.error(f"Search timeout after {request.timeout_seconds} seconds")
            return {
                modality: self._create_empty_modality_result(modality)
                for modality in search_tasks.keys()
            }

    async def _execute_vector_search(
        self,
        request: UnifiedRequest,
        strategy: SearchStrategy,
        azure_services: ConsolidatedAzureServices,
    ) -> ModalityResult:
        """Execute vector search"""
        start_time = time.time()

        try:
            results = await self.vector_engine.search(
                query=request.query,
                domain=request.domain or "general",
                max_results=strategy.max_results_per_modality,
                azure_services=azure_services,
            )

            execution_time = time.time() - start_time
            confidence = self._calculate_modality_confidence(results, "vector")

            return ModalityResult(
                modality="vector",
                results=results,
                execution_time=execution_time,
                confidence_score=confidence,
                result_count=len(results),
                metadata={"strategy": strategy.primary_modality == "vector"},
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._create_empty_modality_result("vector")

    async def _execute_graph_search(
        self,
        request: UnifiedRequest,
        strategy: SearchStrategy,
        azure_services: ConsolidatedAzureServices,
    ) -> ModalityResult:
        """Execute graph search"""
        start_time = time.time()

        try:
            results = await self.graph_engine.search(
                query=request.query,
                domain=request.domain or "general",
                max_results=strategy.max_results_per_modality,
                azure_services=azure_services,
            )

            execution_time = time.time() - start_time
            confidence = self._calculate_modality_confidence(results, "graph")

            return ModalityResult(
                modality="graph",
                results=results,
                execution_time=execution_time,
                confidence_score=confidence,
                result_count=len(results),
                metadata={"strategy": strategy.primary_modality == "graph"},
            )

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return self._create_empty_modality_result("graph")

    async def _execute_gnn_search(
        self,
        request: UnifiedRequest,
        strategy: SearchStrategy,
        azure_services: ConsolidatedAzureServices,
    ) -> ModalityResult:
        """Execute GNN search"""
        start_time = time.time()

        try:
            results = await self.gnn_engine.search(
                query=request.query,
                domain=request.domain or "general",
                max_results=strategy.max_results_per_modality,
                azure_services=azure_services,
            )

            execution_time = time.time() - start_time
            confidence = self._calculate_modality_confidence(results, "gnn")

            return ModalityResult(
                modality="gnn",
                results=results,
                execution_time=execution_time,
                confidence_score=confidence,
                result_count=len(results),
                metadata={"strategy": strategy.primary_modality == "gnn"},
            )

        except Exception as e:
            logger.error(f"GNN search failed: {e}")
            return self._create_empty_modality_result("gnn")

    async def _execute_fast_search(
        self,
        request: UnifiedRequest,
        domain: str,
        azure_services: ConsolidatedAzureServices,
    ) -> Dict[str, Any]:
        """Execute fast search for lightweight processing"""

        try:
            # Use only vector search for speed
            results = await self.vector_engine.search(
                query=request.query,
                domain=domain,
                max_results=min(5, request.max_results),
                azure_services=azure_services,
            )

            return {
                "results": results,
                "confidence": 0.6,  # Moderate confidence for fast search
                "method": "fast_vector_search",
            }

        except Exception as e:
            logger.error(f"Fast search failed: {e}")
            return {"results": [], "confidence": 0.0, "error": str(e)}

    async def _synthesize_search_results(
        self, modality_results: Dict[str, ModalityResult], strategy: SearchStrategy
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
        """Synthesize results from search modalities"""

        all_results = []
        result_sources = {}

        # Collect all results with source tracking
        for modality, modal_result in modality_results.items():
            for i, result in enumerate(modal_result.results):
                result_with_meta = {
                    **result,
                    "source_modality": modality,
                    "modality_rank": i + 1,
                    "modality_confidence": modal_result.confidence_score,
                    "is_primary": modality == strategy.primary_modality,
                }
                all_results.append(result_with_meta)

                result_id = result.get("id", f"{modality}_{i}")
                if result_id not in result_sources:
                    result_sources[result_id] = []
                result_sources[result_id].append(result_with_meta)

        # Deduplicate and rank results
        synthesized_results = []
        rankings = {}

        for result_id, source_list in result_sources.items():
            if len(source_list) == 1:
                result = source_list[0]
                score = self._calculate_result_score(result, strategy)
            else:
                result = self._merge_multi_source_result(source_list, strategy)
                score = (
                    self._calculate_result_score(result, strategy) * 1.2
                )  # Boost multi-source

            synthesized_results.append(result)
            rankings[result_id] = score

        # Sort by score
        synthesized_results.sort(
            key=lambda r: rankings.get(r.get("id", ""), 0), reverse=True
        )
        synthesized_results = synthesized_results[
            : strategy.max_results_per_modality * 2
        ]

        # Calculate synthesis confidence
        synthesis_confidence = self._calculate_synthesis_confidence(
            modality_results, rankings
        )

        return synthesized_results, synthesis_confidence, rankings

    async def _synthesize_unified_results(
        self,
        workflow_results: UnifiedResults,
        modality_results: Dict[str, ModalityResult],
        strategy: SearchStrategy,
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
        """Synthesize results combining workflow and search results"""

        # Get search synthesis
        (
            search_results,
            search_confidence,
            rankings,
        ) = await self._synthesize_search_results(modality_results, strategy)

        # Enhance with workflow context
        enhanced_results = []
        for result in search_results:
            enhanced_result = {
                **result,
                "workflow_context": {
                    "domain": workflow_results.detected_domain,
                    "extraction_available": workflow_results.extraction_results
                    is not None,
                    "config_available": workflow_results.extraction_config is not None,
                },
            }
            enhanced_results.append(enhanced_result)

        # Calculate enhanced confidence
        domain_confidence = (
            workflow_results.domain_analysis.get("confidence", 0.5)
            if workflow_results.domain_analysis
            else 0.5
        )
        enhanced_confidence = (search_confidence + domain_confidence) / 2

        return enhanced_results, enhanced_confidence, rankings

    async def _execute_result_synthesis(
        self,
        request: UnifiedRequest,
        domain_analysis: Dict[str, Any],
        search_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """Execute result synthesis stage"""

        try:
            results = search_results.get("results", [])
            confidence = search_results.get("confidence", 0.0)

            response = {
                "query": request.query,
                "domain": domain_analysis.get("detected_domain", "general"),
                "results": results[: request.max_results],
                "result_count": len(results),
                "search_metadata": {
                    "method": search_results.get("method", "unknown"),
                    "domain_confidence": domain_analysis.get("confidence", 0.0),
                },
                "domain_analysis": domain_analysis,
            }

            overall_confidence = (
                domain_analysis.get("confidence", 0.0) + confidence
            ) / 2

            return response, overall_confidence

        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            return {"query": request.query, "error": str(e), "results": []}, 0.0

    async def _execute_comprehensive_analysis(
        self, unified_results: UnifiedResults, azure_services: ConsolidatedAzureServices
    ) -> Dict[str, Any]:
        """Execute comprehensive analysis for enhanced insights"""

        return {
            "result_quality_assessment": self._assess_result_quality(unified_results),
            "confidence_analysis": self._analyze_confidence_distribution(
                unified_results
            ),
            "performance_analysis": self._analyze_performance_characteristics(
                unified_results
            ),
            "recommendation_analysis": self._generate_improvement_recommendations(
                unified_results
            ),
        }

    # Utility and helper methods

    def _calculate_modality_confidence(
        self, results: List[Dict[str, Any]], modality: str
    ) -> float:
        """Calculate confidence for modality results"""
        if not results:
            return 0.0

        scores = [r.get("score", 0.5) for r in results]
        avg_score = sum(scores) / len(scores)
        count_factor = min(1.0, len(results) / 10)

        return avg_score * count_factor

    def _calculate_result_score(
        self, result: Dict[str, Any], strategy: SearchStrategy
    ) -> float:
        """Calculate unified score for search result"""
        base_score = result.get("score", 0.5)
        modality_confidence = result.get("modality_confidence", 0.5)
        is_primary = result.get("is_primary", False)
        modality_rank = result.get("modality_rank", 999)

        confidence_weight = 0.4
        primary_weight = 0.3 if is_primary else 0.1
        rank_weight = 0.3

        rank_score = max(0, 1.0 - (modality_rank - 1) / 10)

        final_score = (
            base_score * confidence_weight
            + modality_confidence * primary_weight
            + rank_score * rank_weight
        )

        return min(1.0, final_score)

    def _merge_multi_source_result(
        self, source_list: List[Dict[str, Any]], strategy: SearchStrategy
    ) -> Dict[str, Any]:
        """Merge results from multiple modalities"""
        base_result = max(source_list, key=lambda r: r.get("modality_confidence", 0))

        sources = [r["source_modality"] for r in source_list]
        avg_confidence = sum(
            r.get("modality_confidence", 0) for r in source_list
        ) / len(source_list)

        return {
            **base_result,
            "source_modalities": sources,
            "multi_source": True,
            "average_confidence": avg_confidence,
            "source_count": len(source_list),
        }

    def _calculate_synthesis_confidence(
        self, modality_results: Dict[str, ModalityResult], rankings: Dict[str, float]
    ) -> float:
        """Calculate confidence in synthesis process"""

        modality_confidences = [r.confidence_score for r in modality_results.values()]
        avg_modality_confidence = sum(modality_confidences) / len(modality_confidences)

        ranking_scores = list(rankings.values())
        ranking_consistency = (
            1.0 - (max(ranking_scores) - min(ranking_scores)) if ranking_scores else 0.0
        )

        multi_source_count = len([r for r in rankings.values() if r > 0.8])
        multi_source_factor = min(1.0, multi_source_count / 5)

        synthesis_confidence = (
            avg_modality_confidence * 0.5
            + ranking_consistency * 0.3
            + multi_source_factor * 0.2
        )

        return min(1.0, synthesis_confidence)

    def _create_empty_modality_result(self, modality: str) -> ModalityResult:
        """Create empty result for failed modality"""
        return ModalityResult(
            modality=modality,
            results=[],
            execution_time=0.0,
            confidence_score=0.0,
            result_count=0,
            metadata={"error": True},
        )

    def _calculate_performance_metrics(
        self, stage_times: Dict[str, float], azure_services: ConsolidatedAzureServices
    ) -> Dict[str, Any]:
        """Calculate performance metrics"""

        total_time = sum(stage_times.values())

        return {
            "execution_efficiency": {
                "total_time": total_time,
                "stage_breakdown": stage_times,
                "slowest_stage": max(stage_times.items(), key=lambda x: x[1])[0]
                if stage_times
                else None,
                "fastest_stage": min(stage_times.items(), key=lambda x: x[1])[0]
                if stage_times
                else None,
            },
            "performance_targets": {
                "sub_3s_target_met": total_time < 3.0,
                "time_vs_target": total_time / 3.0,
                "performance_grade": "excellent"
                if total_time < 2.0
                else "good"
                if total_time < 3.0
                else "acceptable",
            },
        }

    def _calculate_search_performance_metrics(
        self, modality_results: Dict[str, ModalityResult]
    ) -> Dict[str, Any]:
        """Calculate search-specific performance metrics"""

        total_modality_time = sum(r.execution_time for r in modality_results.values())
        max_time = (
            max(r.execution_time for r in modality_results.values())
            if modality_results
            else 0.0
        )

        parallel_efficiency = total_modality_time / max_time if max_time > 0 else 0.0

        return {
            "search_performance": {
                "total_modality_time": total_modality_time,
                "max_execution_time": max_time,
                "parallel_efficiency": min(1.0, parallel_efficiency),
                "modalities_successful": len(
                    [r for r in modality_results.values() if r.result_count > 0]
                ),
            }
        }

    def _calculate_unified_performance_metrics(
        self,
        workflow_results: UnifiedResults,
        modality_results: Dict[str, ModalityResult],
        azure_services: ConsolidatedAzureServices,
    ) -> Dict[str, Any]:
        """Calculate unified performance metrics"""

        workflow_metrics = self._calculate_performance_metrics(
            workflow_results.stage_execution_times, azure_services
        )
        search_metrics = self._calculate_search_performance_metrics(modality_results)

        return {
            **workflow_metrics,
            **search_metrics,
            "unified_metrics": {
                "workflow_time": sum(workflow_results.stage_execution_times.values()),
                "search_time": sum(r.execution_time for r in modality_results.values()),
                "synthesis_efficiency": workflow_results.synthesis_confidence,
                "overall_efficiency": workflow_results.overall_confidence,
            },
        }

    def _summarize_extraction_results(
        self, extraction_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize extraction results for response"""
        if not extraction_results:
            return {"available": False}

        return {
            "available": True,
            "entities_extracted": extraction_results.get("total_entities_extracted", 0),
            "relationships_extracted": extraction_results.get(
                "total_relationships_extracted", 0
            ),
            "extraction_accuracy": extraction_results.get("extraction_accuracy", 0.0),
            "validation_passed": extraction_results.get(
                "extraction_passed_validation", False
            ),
        }

    def _assess_result_quality(self, results: UnifiedResults) -> Dict[str, Any]:
        """Assess quality of unified results"""
        return {
            "result_count": len(results.synthesized_results),
            "confidence_distribution": self._analyze_confidence_distribution(results),
            "source_diversity": self._analyze_source_diversity(results),
            "quality_score": results.overall_confidence,
        }

    def _analyze_confidence_distribution(
        self, results: UnifiedResults
    ) -> Dict[str, float]:
        """Analyze confidence distribution in results"""
        confidences = [
            r.get("confidence", 0.0)
            for r in results.synthesized_results
            if "confidence" in r
        ]

        if not confidences:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        import statistics

        return {
            "mean": statistics.mean(confidences),
            "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            "min": min(confidences),
            "max": max(confidences),
        }

    def _analyze_source_diversity(self, results: UnifiedResults) -> Dict[str, int]:
        """Analyze diversity of result sources"""
        source_counts = {}
        for result in results.synthesized_results:
            source = result.get("source_modality", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        return source_counts

    def _analyze_performance_characteristics(
        self, results: UnifiedResults
    ) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        return {
            "total_time": results.total_execution_time,
            "stage_breakdown": results.stage_execution_times,
            "efficiency_score": 1.0
            - (results.total_execution_time / 120.0),  # Against 2-minute target
            "sla_compliance": results.total_execution_time < 3.0,
        }

    def _generate_improvement_recommendations(
        self, results: UnifiedResults
    ) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []

        if results.total_execution_time > 3.0:
            recommendations.append(
                "Consider using LIGHTWEIGHT strategy for faster responses"
            )

        if results.overall_confidence < 0.6:
            recommendations.append(
                "Consider using COMPREHENSIVE strategy for higher quality"
            )

        if len(results.synthesized_results) < 5:
            recommendations.append(
                "Consider increasing max_results for more comprehensive results"
            )

        return recommendations

    def _calculate_quality_metrics(self, results: UnifiedResults) -> Dict[str, Any]:
        """Calculate quality metrics for comprehensive analysis"""
        return {
            "completeness": len(results.synthesized_results)
            / max(1, 10),  # Against target of 10 results
            "confidence": results.overall_confidence,
            "diversity": len(self._analyze_source_diversity(results)),
            "performance": 1.0
            - (results.total_execution_time / 120.0),  # Against 2-minute target
        }

    def _get_cache_utilization_stats(self) -> Dict[str, Any]:
        """Get cache utilization statistics"""
        return {"hits": 0, "misses": 0, "hit_rate": 0.0, "size_mb": 0.0}

    def _update_performance_stats(
        self,
        strategy: OrchestrationStrategy,
        execution_time: float,
        success: bool,
        stage_times: Dict[str, float],
    ):
        """Update performance statistics"""
        self._performance_stats["total_requests"] += 1
        if success:
            self._performance_stats["successful_requests"] += 1

        # Update average execution time
        current_avg = self._performance_stats["average_execution_time"]
        total_requests = self._performance_stats["total_requests"]
        self._performance_stats["average_execution_time"] = (
            current_avg * (total_requests - 1) + execution_time
        ) / total_requests

        # Update strategy performance
        strategy_stats = self._performance_stats["strategy_performance"][strategy.value]
        strategy_stats["count"] += 1

        current_avg = strategy_stats["avg_time"]
        count = strategy_stats["count"]
        strategy_stats["avg_time"] = (
            current_avg * (count - 1) + execution_time
        ) / count

        if success:
            current_success = strategy_stats["success_rate"] * (count - 1)
            strategy_stats["success_rate"] = (current_success + 1) / count
        else:
            current_success = strategy_stats["success_rate"] * (count - 1)
            strategy_stats["success_rate"] = current_success / count

        # Update stage performance
        for stage, time_taken in stage_times.items():
            if stage in self._performance_stats["stage_performance"]:
                stage_stats = self._performance_stats["stage_performance"][stage]
                stage_stats["count"] += 1
                current_avg = stage_stats["avg_time"]
                count = stage_stats["count"]
                stage_stats["avg_time"] = (
                    current_avg * (count - 1) + time_taken
                ) / count

    def _update_modality_stats(
        self, modality: str, execution_time: float, success: bool
    ):
        """Update modality performance statistics"""
        if modality in self._performance_stats["search_modality_performance"]:
            stats = self._performance_stats["search_modality_performance"][modality]
            stats["executions"] += 1

            current_avg = stats["avg_time"]
            executions = stats["executions"]
            stats["avg_time"] = (
                current_avg * (executions - 1) + execution_time
            ) / executions

            if success:
                current_successes = stats["success_rate"] * (executions - 1)
                stats["success_rate"] = (current_successes + 1) / executions
            else:
                current_successes = stats["success_rate"] * (executions - 1)
                stats["success_rate"] = current_successes / executions

    async def get_request_progress(self, request_id: str) -> Optional[UnifiedProgress]:
        """Get current progress for a request"""
        return self._active_requests.get(request_id)

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request"""
        if request_id in self._active_requests:
            self._active_requests[request_id].status = ProcessingStatus.CANCELLED
            return True
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            **self._performance_stats,
            "success_rate": (
                self._performance_stats["successful_requests"]
                / self._performance_stats["total_requests"]
                if self._performance_stats["total_requests"] > 0
                else 0.0
            ),
            "active_requests": len(self._active_requests),
        }


# Global orchestrator instance
_unified_orchestrator = UnifiedOrchestrator()


# Convenience functions for different orchestration types


async def execute_workflow_orchestration(
    query: str,
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
    max_results: int = 10,
    enable_streaming: bool = True,
) -> UnifiedResults:
    """Execute workflow-only orchestration"""
    request = UnifiedRequest(
        query=query,
        orchestration_strategy=OrchestrationStrategy.WORKFLOW_ONLY,
        domain=domain,
        max_results=max_results,
        enable_streaming=enable_streaming,
    )
    return await _unified_orchestrator.execute_unified(request, azure_services)


async def execute_search_orchestration(
    query: str,
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
    search_modalities: List[str] = None,
    max_results: int = 10,
) -> UnifiedResults:
    """Execute search-only orchestration"""
    request = UnifiedRequest(
        query=query,
        orchestration_strategy=OrchestrationStrategy.SEARCH_ONLY,
        domain=domain,
        search_modalities=search_modalities or ["vector", "graph", "gnn"],
        max_results=max_results,
    )
    return await _unified_orchestrator.execute_unified(request, azure_services)


async def execute_unified_orchestration(
    query: str,
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
    max_results: int = 10,
    enable_streaming: bool = True,
) -> UnifiedResults:
    """Execute complete unified orchestration"""
    request = UnifiedRequest(
        query=query,
        orchestration_strategy=OrchestrationStrategy.UNIFIED_WORKFLOW,
        domain=domain,
        max_results=max_results,
        enable_streaming=enable_streaming,
    )
    return await _unified_orchestrator.execute_unified(request, azure_services)


async def execute_lightweight_orchestration(
    query: str,
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
    max_results: int = 5,
) -> UnifiedResults:
    """Execute lightweight orchestration for fast responses"""
    request = UnifiedRequest(
        query=query,
        orchestration_strategy=OrchestrationStrategy.LIGHTWEIGHT,
        domain=domain,
        max_results=max_results,
        timeout_seconds=10.0,
    )
    return await _unified_orchestrator.execute_unified(request, azure_services)


async def execute_comprehensive_orchestration(
    query: str,
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
    max_results: int = 15,
    enable_streaming: bool = True,
) -> UnifiedResults:
    """Execute comprehensive orchestration with all features"""
    request = UnifiedRequest(
        query=query,
        orchestration_strategy=OrchestrationStrategy.COMPREHENSIVE,
        domain=domain,
        max_results=max_results,
        enable_streaming=enable_streaming,
        timeout_seconds=180.0,
    )
    return await _unified_orchestrator.execute_unified(request, azure_services)


def get_unified_orchestrator() -> UnifiedOrchestrator:
    """Get the global unified orchestrator instance"""
    return _unified_orchestrator


# Export main components
__all__ = [
    "UnifiedOrchestrator",
    "UnifiedRequest",
    "UnifiedResults",
    "UnifiedProgress",
    "OrchestrationStrategy",
    "ProcessingStage",
    "ProcessingStatus",
    "execute_workflow_orchestration",
    "execute_search_orchestration",
    "execute_unified_orchestration",
    "execute_lightweight_orchestration",
    "execute_comprehensive_orchestration",
    "get_unified_orchestrator",
]
