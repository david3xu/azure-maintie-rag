"""
Search Orchestrator - Unified Search Coordination

This module provides centralized search orchestration that coordinates all search
modalities (Vector + Graph + GNN) with optimal performance and result synthesis.

Key Features:
- Tri-modal search orchestration with simultaneous execution
- Dynamic search strategy optimization based on query analysis
- Result synthesis and ranking across all modalities
- Performance monitoring and caching coordination
- Domain-aware search parameter optimization

Architecture Integration:
- Uses Config-Extraction workflow for optimal search parameters
- Integrates with Domain Intelligence Agent for query analysis
- Coordinates with Azure services for scalable search execution
- Provides unified interface for all search operations
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Interface contracts
from config.extraction_interface import ExtractionConfiguration

# Azure service integration
from ..core.azure_services import ConsolidatedAzureServices
from ..core.cache_manager import UnifiedCacheManager
from ..core.error_handler import get_error_handler
from ..core.performance_monitor import get_performance_monitor
from ..universal_search.gnn_search import GNNSearchEngine
from ..universal_search.graph_search import GraphSearchEngine

# Search engine imports
from ..universal_search.vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy:
    """Search strategy configuration for optimal performance"""

    primary_modality: str  # "vector", "graph", or "gnn"
    secondary_modalities: List[str]
    parallel_execution: bool
    result_fusion_method: str  # "weighted", "ranked", "hybrid"
    confidence_threshold: float
    max_results_per_modality: int


class SearchRequest(BaseModel):
    """Unified search request model"""

    query: str = Field(..., description="Search query text")
    domain: str = Field(..., description="Domain context for search optimization")
    strategy: Optional[SearchStrategy] = Field(
        None, description="Optional search strategy override"
    )
    max_results: int = Field(10, description="Maximum total results")
    include_metadata: bool = Field(True, description="Include result metadata")
    timeout_seconds: float = Field(30.0, description="Search timeout")


class ModalityResult(BaseModel):
    """Results from a single search modality"""

    modality: str = Field(..., description="Search modality name")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    execution_time: float = Field(..., description="Execution time in seconds")
    confidence_score: float = Field(..., description="Overall confidence score")
    result_count: int = Field(..., description="Number of results returned")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResults(BaseModel):
    """Unified search results across all modalities"""

    query: str = Field(..., description="Original search query")
    domain: str = Field(..., description="Domain context")
    total_execution_time: float = Field(..., description="Total execution time")

    # Individual modality results
    vector_results: ModalityResult = Field(..., description="Vector search results")
    graph_results: ModalityResult = Field(..., description="Graph search results")
    gnn_results: ModalityResult = Field(..., description="GNN search results")

    # Unified results
    synthesized_results: List[Dict[str, Any]] = Field(
        ..., description="Synthesized unified results"
    )
    synthesis_confidence: float = Field(..., description="Synthesis confidence score")
    result_rankings: Dict[str, float] = Field(
        ..., description="Result relevance rankings"
    )

    # Performance metrics
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    parallel_efficiency: float = Field(..., description="Parallel execution efficiency")


class SearchOrchestrator:
    """
    Unified search orchestrator that coordinates all search modalities with
    optimal performance and intelligent result synthesis.
    """

    def __init__(self):
        self.cache_manager = UnifiedCacheManager()
        self.error_handler = get_error_handler()
        self.performance_monitor = get_performance_monitor()

        # Initialize search engines
        self.vector_engine = VectorSearchEngine()
        self.graph_engine = GraphSearchEngine()
        self.gnn_engine = GNNSearchEngine()

        # Performance tracking
        self._search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "modality_performance": {
                "vector": {"searches": 0, "avg_time": 0.0, "success_rate": 1.0},
                "graph": {"searches": 0, "avg_time": 0.0, "success_rate": 1.0},
                "gnn": {"searches": 0, "avg_time": 0.0, "success_rate": 1.0},
            },
        }

    async def execute_search(
        self,
        request: SearchRequest,
        azure_services: Optional[ConsolidatedAzureServices] = None,
        extraction_config: Optional[ExtractionConfiguration] = None,
    ) -> SearchResults:
        """
        Execute unified search across all modalities with optimal orchestration.

        Args:
            request: Search request parameters
            azure_services: Azure service container for search execution
            extraction_config: Optional extraction configuration for optimization

        Returns:
            SearchResults: Unified search results with synthesis and rankings
        """
        start_time = time.time()

        try:
            # Step 1: Determine optimal search strategy
            strategy = await self._determine_search_strategy(request, extraction_config)

            # Step 2: Execute search across all modalities
            modality_results = await self._execute_parallel_search(
                request, strategy, azure_services
            )

            # Step 3: Synthesize and rank results
            (
                synthesized_results,
                synthesis_confidence,
                rankings,
            ) = await self._synthesize_results(modality_results, strategy)

            # Step 4: Calculate performance metrics
            execution_time = time.time() - start_time
            performance_metrics = self._calculate_performance_metrics(
                modality_results, execution_time
            )

            # Step 5: Update statistics
            self._update_search_statistics(execution_time, True)

            # Step 6: Monitor tri-modal search performance
            modalities_used = [
                modality
                for modality, result in modality_results.items()
                if result.result_count > 0
            ]

            await self.performance_monitor.track_tri_modal_search_performance(
                search_time=execution_time,
                confidence=synthesis_confidence,
                modalities_used=modalities_used,
                correlation_id=f"search_{hash(request.query)}",
            )

            return SearchResults(
                query=request.query,
                domain=request.domain,
                total_execution_time=execution_time,
                vector_results=modality_results["vector"],
                graph_results=modality_results["graph"],
                gnn_results=modality_results["gnn"],
                synthesized_results=synthesized_results,
                synthesis_confidence=synthesis_confidence,
                result_rankings=rankings,
                cache_hit_rate=performance_metrics["cache_hit_rate"],
                parallel_efficiency=performance_metrics["parallel_efficiency"],
            )

        except Exception as e:
            execution_time = time.time() - start_time
            await self.error_handler.handle_error(
                error=e,
                operation="execute_search",
                component="search_orchestrator",
                parameters={"query": request.query, "domain": request.domain},
            )
            self._update_search_statistics(execution_time, False)
            raise

    async def _determine_search_strategy(
        self,
        request: SearchRequest,
        extraction_config: Optional[ExtractionConfiguration],
    ) -> SearchStrategy:
        """Determine optimal search strategy based on query and domain analysis"""

        # Use provided strategy if available
        if request.strategy:
            return request.strategy

        # Analyze query characteristics
        query_length = len(request.query.split())
        has_entities = any(word.isupper() for word in request.query.split())
        has_relationships = any(
            word in request.query.lower()
            for word in ["and", "with", "between", "related"]
        )

        # Determine primary modality based on query characteristics
        if extraction_config:
            # Use extraction configuration for optimization
            entity_count = len(extraction_config.expected_entity_types)
            relationship_count = len(extraction_config.relationship_patterns)

            if entity_count > 20 and relationship_count > 10:
                primary_modality = "gnn"  # Complex pattern search
            elif relationship_count > 5:
                primary_modality = "graph"  # Relationship-focused
            else:
                primary_modality = "vector"  # Semantic similarity
        else:
            # Fallback heuristics
            if query_length > 10 and has_relationships:
                primary_modality = "graph"
            elif has_entities and has_relationships:
                primary_modality = "gnn"
            else:
                primary_modality = "vector"

        # Configure strategy
        return SearchStrategy(
            primary_modality=primary_modality,
            secondary_modalities=[
                m for m in ["vector", "graph", "gnn"] if m != primary_modality
            ],
            parallel_execution=True,  # Always use parallel execution for performance
            result_fusion_method="hybrid",  # Use hybrid fusion for best results
            confidence_threshold=0.7,
            max_results_per_modality=max(5, request.max_results // 2),
        )

    async def _execute_parallel_search(
        self,
        request: SearchRequest,
        strategy: SearchStrategy,
        azure_services: Optional[ConsolidatedAzureServices],
    ) -> Dict[str, ModalityResult]:
        """Execute search across all modalities in parallel"""

        # Create search tasks
        search_tasks = {
            "vector": self._execute_vector_search(request, strategy, azure_services),
            "graph": self._execute_graph_search(request, strategy, azure_services),
            "gnn": self._execute_gnn_search(request, strategy, azure_services),
        }

        # Execute searches with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks.values(), return_exceptions=True),
                timeout=request.timeout_seconds,
            )

            # Process results
            modality_results = {}
            for modality, result in zip(search_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Search failed for {modality}: {result}")
                    modality_results[modality] = self._create_empty_result(modality)
                else:
                    modality_results[modality] = result

            return modality_results

        except asyncio.TimeoutError:
            logger.error(f"Search timeout after {request.timeout_seconds} seconds")
            return {
                modality: self._create_empty_result(modality)
                for modality in search_tasks.keys()
            }

    async def _execute_vector_search(
        self,
        request: SearchRequest,
        strategy: SearchStrategy,
        azure_services: Optional[ConsolidatedAzureServices],
    ) -> ModalityResult:
        """Execute vector search with optimization"""
        start_time = time.time()

        try:
            # Execute vector search using the engine
            results = await self.vector_engine.search(
                query=request.query,
                domain=request.domain,
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
            return self._create_empty_result("vector")

    async def _execute_graph_search(
        self,
        request: SearchRequest,
        strategy: SearchStrategy,
        azure_services: Optional[ConsolidatedAzureServices],
    ) -> ModalityResult:
        """Execute graph search with optimization"""
        start_time = time.time()

        try:
            # Execute graph search using the engine
            results = await self.graph_engine.search(
                query=request.query,
                domain=request.domain,
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
            return self._create_empty_result("graph")

    async def _execute_gnn_search(
        self,
        request: SearchRequest,
        strategy: SearchStrategy,
        azure_services: Optional[ConsolidatedAzureServices],
    ) -> ModalityResult:
        """Execute GNN search with optimization"""
        start_time = time.time()

        try:
            # Execute GNN search using the engine
            results = await self.gnn_engine.search(
                query=request.query,
                domain=request.domain,
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
            return self._create_empty_result("gnn")

    async def _synthesize_results(
        self, modality_results: Dict[str, ModalityResult], strategy: SearchStrategy
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
        """Synthesize results from all modalities into unified rankings"""

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

                # Track result sources for deduplication
                result_id = result.get("id", f"{modality}_{i}")
                if result_id not in result_sources:
                    result_sources[result_id] = []
                result_sources[result_id].append(result_with_meta)

        # Deduplicate and rank results
        synthesized_results = []
        rankings = {}

        for result_id, source_list in result_sources.items():
            if len(source_list) == 1:
                # Single source result
                result = source_list[0]
                score = self._calculate_result_score(result, strategy)
            else:
                # Multi-source result - synthesize
                result = self._merge_multi_source_result(source_list, strategy)
                score = (
                    self._calculate_result_score(result, strategy) * 1.2
                )  # Boost multi-source

            synthesized_results.append(result)
            rankings[result_id] = score

        # Sort by score and limit results
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

    def _calculate_result_score(
        self, result: Dict[str, Any], strategy: SearchStrategy
    ) -> float:
        """Calculate unified score for a search result"""
        base_score = result.get("score", 0.5)
        modality_confidence = result.get("modality_confidence", 0.5)
        is_primary = result.get("is_primary", False)
        modality_rank = result.get("modality_rank", 999)

        # Weight factors
        confidence_weight = 0.4
        primary_weight = 0.3 if is_primary else 0.1
        rank_weight = 0.3

        # Calculate weighted score
        rank_score = max(0, 1.0 - (modality_rank - 1) / 10)  # Rank decay

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
        # Use the highest confidence result as base
        base_result = max(source_list, key=lambda r: r.get("modality_confidence", 0))

        # Aggregate metadata
        sources = [r["source_modality"] for r in source_list]
        avg_confidence = sum(
            r.get("modality_confidence", 0) for r in source_list
        ) / len(source_list)

        merged_result = {
            **base_result,
            "source_modalities": sources,
            "multi_source": True,
            "average_confidence": avg_confidence,
            "source_count": len(source_list),
        }

        return merged_result

    def _calculate_modality_confidence(
        self, results: List[Dict[str, Any]], modality: str
    ) -> float:
        """Calculate overall confidence for a modality's results"""
        if not results:
            return 0.0

        # Calculate average score from results
        scores = [r.get("score", 0.5) for r in results]
        avg_score = sum(scores) / len(scores)

        # Adjust based on result count (more results = higher confidence up to a point)
        count_factor = min(1.0, len(results) / 10)

        return avg_score * count_factor

    def _calculate_synthesis_confidence(
        self, modality_results: Dict[str, ModalityResult], rankings: Dict[str, float]
    ) -> float:
        """Calculate confidence in the synthesis process"""

        # Base confidence from modality results
        modality_confidences = [r.confidence_score for r in modality_results.values()]
        avg_modality_confidence = sum(modality_confidences) / len(modality_confidences)

        # Boost for consistent rankings across modalities
        ranking_scores = list(rankings.values())
        ranking_consistency = (
            1.0 - (max(ranking_scores) - min(ranking_scores)) if ranking_scores else 0.0
        )

        # Boost for multi-source results
        multi_source_count = len([r for r in rankings.values() if r > 0.8])
        multi_source_factor = min(1.0, multi_source_count / 5)

        synthesis_confidence = (
            avg_modality_confidence * 0.5
            + ranking_consistency * 0.3
            + multi_source_factor * 0.2
        )

        return min(1.0, synthesis_confidence)

    def _create_empty_result(self, modality: str) -> ModalityResult:
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
        self, modality_results: Dict[str, ModalityResult], total_time: float
    ) -> Dict[str, float]:
        """Calculate performance metrics for the search execution"""

        # Calculate parallel efficiency
        total_modality_time = sum(r.execution_time for r in modality_results.values())
        parallel_efficiency = (
            total_modality_time / total_time if total_time > 0 else 0.0
        )

        # Estimate cache hit rate (simplified)
        fast_results = sum(
            1 for r in modality_results.values() if r.execution_time < 0.1
        )
        cache_hit_rate = (
            fast_results / len(modality_results) if modality_results else 0.0
        )

        return {
            "parallel_efficiency": min(1.0, parallel_efficiency),
            "cache_hit_rate": cache_hit_rate,
        }

    def _update_search_statistics(self, execution_time: float, success: bool):
        """Update internal search statistics"""
        self._search_stats["total_searches"] += 1
        if success:
            self._search_stats["successful_searches"] += 1

        # Update average response time
        current_avg = self._search_stats["average_response_time"]
        total_searches = self._search_stats["total_searches"]
        self._search_stats["average_response_time"] = (
            current_avg * (total_searches - 1) + execution_time
        ) / total_searches

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self._search_stats,
            "success_rate": (
                self._search_stats["successful_searches"]
                / self._search_stats["total_searches"]
                if self._search_stats["total_searches"] > 0
                else 0.0
            ),
        }


# Global orchestrator instance for module access
_search_orchestrator = SearchOrchestrator()


async def execute_unified_search(
    query: str,
    domain: str,
    azure_services: Optional[ConsolidatedAzureServices] = None,
    extraction_config: Optional[ExtractionConfiguration] = None,
    max_results: int = 10,
) -> SearchResults:
    """
    Convenience function for executing unified search.

    Args:
        query: Search query text
        domain: Domain context for optimization
        azure_services: Azure service container
        extraction_config: Optional extraction configuration
        max_results: Maximum results to return

    Returns:
        SearchResults: Unified search results across all modalities
    """
    request = SearchRequest(query=query, domain=domain, max_results=max_results)

    return await _search_orchestrator.execute_search(
        request, azure_services, extraction_config
    )


def get_search_orchestrator() -> SearchOrchestrator:
    """Get the global search orchestrator instance"""
    return _search_orchestrator


# Export main components
__all__ = [
    "SearchOrchestrator",
    "SearchRequest",
    "SearchResults",
    "ModalityResult",
    "SearchStrategy",
    "execute_unified_search",
    "get_search_orchestrator",
]
