"""
Tri-Modal Search Orchestrator - Main Coordination Logic

This module orchestrates Vector + Graph + GNN searches simultaneously
and synthesizes unified results according to the tri-modal unity principle.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..universal_search.gnn_search import GNNSearchEngine
from ..universal_search.graph_search import GraphSearchEngine

# Import search modalities
from ..universal_search.vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from tri-modal orchestration"""

    query: str
    results: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    modality_breakdown: Dict[str, Any]
    correlation_id: str


@dataclass
class ModalityResult:
    """Result from individual search modality"""

    content: str
    confidence: float
    metadata: Dict[str, Any]
    execution_time: float
    source: str


class TriModalOrchestrator:
    """
    Unified orchestrator that executes all three modalities simultaneously
    and synthesizes results according to tri-modal unity principle.

    This implements the core competitive advantage of simultaneous
    Vector + Graph + GNN execution without heuristic selection.
    """

    def __init__(self, timeout: float = None):
        # Default timeout for tri-modal search
        self.timeout = timeout or 2.5  # Default 2.5 seconds
        self.vector_engine = VectorSearchEngine()
        self.graph_engine = GraphSearchEngine()
        self.gnn_engine = GNNSearchEngine()

        # Performance tracking
        self.execution_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "average_response_time": 0.0,
            "modality_performance": {
                "vector": {"executions": 0, "avg_time": 0.0, "success_rate": 1.0},
                "graph": {"executions": 0, "avg_time": 0.0, "success_rate": 1.0},
                "gnn": {"executions": 0, "avg_time": 0.0, "success_rate": 1.0},
            },
        }

        logger.info("Tri-modal orchestrator initialized with 2.5s timeout")

    def _get_default_search_types(self) -> List[str]:
        """Get default search types based on available capabilities (data-driven)"""
        # This preserves our competitive advantage of tri-modal unity
        available_types = []

        # Vector search is always available (fundamental capability)
        available_types.append("vector")

        # Check graph search availability
        if hasattr(self, "graph_engine") and self.graph_engine:
            available_types.append("graph")

        # Check GNN search availability
        if hasattr(self, "gnn_engine") and self.gnn_engine:
            available_types.append("gnn")

        return available_types

    async def search(
        self,
        query: str,
        search_types: List[str] = None,
        domain: str = None,
        max_results: int = 10,
        correlation_id: Optional[str] = None,
    ) -> SearchResult:
        """
        Execute unified tri-modal search with result synthesis.

        This is the main entry point that implements the tri-modal unity principle
        by executing all search modalities simultaneously.
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid.uuid4())
        # Get available search modalities dynamically (tri-modal unity principle)
        search_types = search_types or self._get_default_search_types()

        logger.info(
            f"Executing tri-modal search for query: {query[:100]}",
            extra={
                "correlation_id": correlation_id,
                "search_types": search_types,
                "domain": domain,
                "max_results": max_results,
            },
        )

        try:
            # Execute all modalities in parallel
            search_tasks = []
            context = {
                "domain": domain,
                "max_results": max_results,
                "correlation_id": correlation_id,
            }

            # Create search tasks based on requested types
            if "vector" in search_types:
                search_tasks.append(
                    self._execute_with_timeout(
                        self.vector_engine.execute_search(query, context), "vector"
                    )
                )

            if "graph" in search_types:
                search_tasks.append(
                    self._execute_with_timeout(
                        self.graph_engine.execute_search(query, context), "graph"
                    )
                )

            if "gnn" in search_types:
                search_tasks.append(
                    self._execute_with_timeout(
                        self.gnn_engine.execute_search(query, context), "gnn"
                    )
                )

            # Execute all searches simultaneously
            modality_results = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )

            # Process results and handle any exceptions
            processed_results = {}
            execution_errors = []

            modality_names = []
            if "vector" in search_types:
                modality_names.append("vector")
            if "graph" in search_types:
                modality_names.append("graph")
            if "gnn" in search_types:
                modality_names.append("gnn")

            for i, result in enumerate(modality_results):
                modality_name = (
                    modality_names[i] if i < len(modality_names) else f"unknown_{i}"
                )

                if isinstance(result, Exception):
                    execution_errors.append(
                        f"{modality_name} search failed: {str(result)}"
                    )
                    logger.warning(f"{modality_name} search failed: {result}")
                else:
                    processed_results[modality_name] = result
                    self._update_modality_stats(
                        modality_name, result.execution_time, True
                    )

            # Synthesize unified results
            unified_result = self._synthesize_results(
                query, processed_results, execution_errors, correlation_id
            )

            execution_time = time.time() - start_time
            unified_result.execution_time = execution_time

            # Update overall statistics
            self._update_execution_stats(execution_time, len(execution_errors) == 0)

            logger.info(
                f"Tri-modal search completed in {execution_time:.3f}s",
                extra={
                    "correlation_id": correlation_id,
                    "confidence": unified_result.confidence,
                    "total_results": len(unified_result.results),
                    "modalities_successful": len(processed_results),
                },
            )

            return unified_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Tri-modal search failed: {e}",
                extra={"correlation_id": correlation_id},
            )

            self._update_execution_stats(execution_time, False)

            # Return minimal result on failure
            return SearchResult(
                query=query,
                results=[],
                confidence=0.0,
                execution_time=execution_time,
                modality_breakdown={"error": str(e)},
                correlation_id=correlation_id,
            )

    async def _execute_with_timeout(self, search_task, modality_name: str):
        """Execute search task with timeout protection"""
        try:
            return await asyncio.wait_for(search_task, timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"{modality_name} search timed out after {self.timeout}s")
            raise TimeoutError(
                f"{modality_name} search exceeded {self.timeout}s timeout"
            )

    def _synthesize_results(
        self,
        query: str,
        modality_results: Dict[str, ModalityResult],
        execution_errors: List[str],
        correlation_id: str,
    ) -> SearchResult:
        """
        Synthesize results from multiple modalities into unified result.

        This implements the tri-modal unity principle by combining results
        from all available modalities with confidence-weighted synthesis.
        """
        unified_results = []
        total_confidence = 0.0
        modality_breakdown = {
            "vector": {},
            "graph": {},
            "gnn": {},
            "synthesis": {},
            "errors": execution_errors,
        }

        # Process results from each modality
        for modality_name, result in modality_results.items():
            modality_breakdown[modality_name] = {
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "metadata": result.metadata,
            }

            # Weight results by confidence
            weighted_confidence = result.confidence
            total_confidence += weighted_confidence

            # Add modality-specific results to unified results
            unified_results.append(
                {
                    "content": result.content,
                    "confidence": result.confidence,
                    "source": result.source,
                    "modality": modality_name,
                    "metadata": result.metadata,
                }
            )

        # Calculate overall confidence
        num_successful_modalities = len(modality_results)
        if num_successful_modalities > 0:
            # Tri-modal bonus: Higher confidence when all modalities succeed
            base_confidence = total_confidence / num_successful_modalities
            tri_modal_bonus = 0.1 * (num_successful_modalities / 3.0)  # Up to 10% bonus
            overall_confidence = min(1.0, base_confidence + tri_modal_bonus)
        else:
            overall_confidence = 0.0

        # Add synthesis metadata
        modality_breakdown["synthesis"] = {
            "modalities_used": num_successful_modalities,
            "tri_modal_bonus": tri_modal_bonus
            if num_successful_modalities > 0
            else 0.0,
            "base_confidence": base_confidence
            if num_successful_modalities > 0
            else 0.0,
            "overall_confidence": overall_confidence,
        }

        return SearchResult(
            query=query,
            results=unified_results,
            confidence=overall_confidence,
            execution_time=0.0,  # Will be set by caller
            modality_breakdown=modality_breakdown,
            correlation_id=correlation_id,
        )

    def _update_modality_stats(
        self, modality: str, execution_time: float, success: bool
    ):
        """Update performance statistics for individual modality"""
        if modality in self.execution_stats["modality_performance"]:
            stats = self.execution_stats["modality_performance"][modality]
            stats["executions"] += 1

            # Update average execution time
            current_avg = stats["avg_time"]
            executions = stats["executions"]
            stats["avg_time"] = (
                (current_avg * (executions - 1)) + execution_time
            ) / executions

            # Update success rate
            if success:
                current_successes = stats["success_rate"] * (executions - 1)
                stats["success_rate"] = (current_successes + 1) / executions
            else:
                current_successes = stats["success_rate"] * (executions - 1)
                stats["success_rate"] = current_successes / executions

    def _update_execution_stats(self, execution_time: float, success: bool):
        """Update overall execution statistics"""
        self.execution_stats["total_searches"] += 1

        if success:
            self.execution_stats["successful_searches"] += 1

        # Update average response time
        total = self.execution_stats["total_searches"]
        avg = self.execution_stats["average_response_time"]
        self.execution_stats["average_response_time"] = (
            (avg * (total - 1)) + execution_time
        ) / total

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.execution_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all search modalities"""
        health_status = {
            "overall_status": "healthy",
            "modalities": {},
            "timestamp": time.time(),
        }

        # Test each modality with a simple query
        test_query = "health check test"
        test_context = {"domain": "test", "max_results": 1}

        for name, engine in [
            ("vector", self.vector_engine),
            ("graph", self.graph_engine),
            ("gnn", self.gnn_engine),
        ]:
            try:
                start_time = time.time()
                await asyncio.wait_for(
                    engine.execute_search(test_query, test_context), timeout=1.0
                )
                health_status["modalities"][name] = {
                    "status": "healthy",
                    "response_time": time.time() - start_time,
                }
            except Exception as e:
                health_status["modalities"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["overall_status"] = "degraded"

        return health_status
