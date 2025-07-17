"""
Enhanced Universal RAG System
Provides advanced query processing with Universal RAG orchestrator integration
Supports any domain through pure text processing and dynamic type discovery
Enhanced with Universal Workflow Manager for detailed real-time progress tracking
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from core.orchestration.universal_rag_orchestrator_complete import (
    UniversalRAGOrchestrator, create_universal_rag_from_texts, create_universal_rag_from_directory
)
from config.settings import settings

logger = logging.getLogger(__name__)

# Global instances for enhanced RAG systems by domain
enhanced_rag_instances: Dict[str, 'EnhancedUniversalRAG'] = {}


class EnhancedUniversalRAG:
    """
    Enhanced Universal RAG System

    Provides sophisticated query processing with:
    - Universal RAG orchestrator integration
    - Dynamic domain adaptation
    - Real-time workflow progress tracking (via Universal Workflow Manager)
    - Multi-strategy search capabilities
    - Performance monitoring and optimization
    """

    def __init__(self, domain_name: str = "general"):
        """
        Initialize Enhanced Universal RAG for a specific domain

        Args:
            domain_name: Name of the domain for this instance
        """
        self.domain_name = domain_name
        self.universal_orchestrator: Optional[UniversalRAGOrchestrator] = None
        self.components_initialized = False

        # Search strategies
        self.available_strategies = ["universal", "vector_focused", "graph_enhanced", "hybrid"]
        self.active_strategy = "universal"

        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0

        # System status
        self.initialization_results: Dict[str, Any] = {}

        logger.info(f"Enhanced Universal RAG initialized for domain: {domain_name}")

    async def initialize_components(
        self,
        text_files: Optional[List[Path]] = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize Universal RAG components for the domain

        Args:
            text_files: Optional list of text files to process
            force_rebuild: Whether to force rebuild even if components exist

        Returns:
            Initialization results dictionary
        """
        if self.components_initialized and not force_rebuild:
            logger.info(f"Enhanced Universal RAG already initialized for domain: {self.domain_name}")
            return self.initialization_results

        logger.info(f"Initializing Enhanced Universal RAG components for domain: {self.domain_name}")
        start_time = datetime.now()

        try:
            # Initialize Universal RAG Orchestrator
            if text_files:
                self.universal_orchestrator = create_universal_rag_from_texts(
                    texts=[f.read_text() for f in text_files],
                    text_sources=[str(f) for f in text_files],
                    domain_name=self.domain_name
                )
            else:
                # Use default directory for the domain
                default_data_dir = Path(settings.raw_data_dir)
                self.universal_orchestrator = await create_universal_rag_from_directory(
                    data_directory=default_data_dir,
                    domain_name=self.domain_name
                )

            # Verify initialization
            if not self.universal_orchestrator or not self.universal_orchestrator.initialized:
                raise Exception("Universal RAG Orchestrator initialization failed")

            self.components_initialized = True
            processing_time = (datetime.now() - start_time).total_seconds()

            # Store initialization results
            self.initialization_results = {
                "success": True,
                "domain": self.domain_name,
                "processing_time": processing_time,
                "system_stats": self.universal_orchestrator.get_system_status()["system_stats"],
                "discovered_types": self.universal_orchestrator.get_system_status()["discovered_types"],
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Enhanced Universal RAG initialization completed in {processing_time:.2f}s")
            return self.initialization_results

        except Exception as e:
            logger.error(f"Enhanced Universal RAG initialization failed: {e}", exc_info=True)
            self.initialization_results = {
                "success": False,
                "error": str(e),
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            }
            return self.initialization_results

    async def process_query(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True,
        stream_progress: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a query using Enhanced Universal RAG with optional workflow progress tracking

        Args:
            query: User query to process
            max_results: Maximum number of results to return
            include_explanations: Whether to include explanations
            enable_safety_warnings: Whether to enable safety warnings
            stream_progress: Whether to stream progress updates
            progress_callback: Optional callback for progress updates (workflow manager)

        Returns:
            Comprehensive query processing results
        """
        if not self.components_initialized:
            return {
                "success": False,
                "error": "Enhanced Universal RAG not initialized",
                "query": query,
                "domain": self.domain_name
            }

        logger.info(f"Processing Enhanced Universal RAG query: {query}")
        start_time = datetime.now()

        try:
            # Progress tracking with workflow manager integration
            if progress_callback:
                await progress_callback("🔍 Analyzing your question...", 10)

            # Process query using universal orchestrator with detailed progress
            query_results = await self.universal_orchestrator.process_query(
                query=query,
                max_results=max_results,
                include_explanations=include_explanations,
                stream_progress=stream_progress,
                progress_callback=progress_callback
            )

            if not query_results.get("success", False):
                return {
                    "success": False,
                    "error": query_results.get("error", "Query processing failed"),
                    "query": query,
                    "domain": self.domain_name
                }

            # Enhanced results processing with progress updates
            if progress_callback:
                await progress_callback("🧠 Enhancing results with AI optimizations...", 85)

            enhanced_results = await self._enhance_query_results(
                query_results,
                enable_safety_warnings,
                progress_callback
            )

            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time)

            if progress_callback:
                await progress_callback("✅ Query processing complete!", 100)

            # Compile comprehensive final response
            final_response = {
                "success": True,
                "query": query,
                "domain": self.domain_name,
                "strategy": self.active_strategy,
                "query_analysis": enhanced_results.get("analysis", {}),
                "enhanced_query": enhanced_results.get("enhanced_query", {}),
                "search_results": enhanced_results.get("search_results", []),
                "generated_response": enhanced_results.get("response", {}),
                "processing_time": processing_time,
                "query_count": self.query_count,
                "average_processing_time": self.average_processing_time,
                "system_stats": self.universal_orchestrator.get_system_status()["system_stats"],
                "performance_metrics": {
                    "total_queries": self.query_count,
                    "average_time": self.average_processing_time,
                    "current_time": processing_time,
                    "strategy_used": self.active_strategy
                },
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Enhanced Universal RAG query completed in {processing_time:.2f}s")
            return final_response

        except Exception as e:
            logger.error(f"Enhanced Universal RAG query processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "domain": self.domain_name,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }

    async def _enhance_query_results(
        self,
        query_results: Dict[str, Any],
        enable_safety_warnings: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Apply additional enhancements to query results

        Args:
            query_results: Raw query results from orchestrator
            enable_safety_warnings: Whether to include safety warnings
            progress_callback: Optional progress callback

        Returns:
            Enhanced query results
        """
        try:
            if progress_callback:
                await progress_callback("🔧 Applying advanced optimizations...", 90)

            enhanced = {
                "analysis": query_results.get("analysis", {}),
                "enhanced_query": query_results.get("enhanced_query", {}),
                "search_results": query_results.get("search_results", []),
                "response": query_results.get("response", {})
            }

            # Add safety warnings if enabled
            if enable_safety_warnings:
                enhanced["safety_warnings"] = self._generate_safety_warnings(query_results)

            # Add quality indicators
            enhanced["quality_indicators"] = self._calculate_quality_indicators(query_results)

            # Add enhanced metadata
            enhanced["metadata"] = {
                "domain": self.domain_name,
                "strategy": self.active_strategy,
                "enhancement_applied": True,
                "safety_warnings_enabled": enable_safety_warnings
            }

            if progress_callback:
                await progress_callback("✨ Enhancements applied successfully", 95)

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing query results: {e}", exc_info=True)
            # Return original results if enhancement fails
            return query_results

    def _generate_safety_warnings(self, query_results: Dict[str, Any]) -> List[str]:
        """Generate relevant safety warnings based on query results"""
        warnings = []

        # Check response content for safety considerations
        response_text = str(query_results.get("response", {}).get("content", "")).lower()

        if any(term in response_text for term in ["danger", "risk", "hazard", "safety"]):
            warnings.append("This response contains safety-related information. Please follow all safety protocols.")

        if any(term in response_text for term in ["electrical", "voltage", "current"]):
            warnings.append("Electrical work should only be performed by qualified personnel.")

        if any(term in response_text for term in ["chemical", "toxic", "corrosive"]):
            warnings.append("Handle chemical substances according to safety data sheets and regulations.")

        return warnings

    def _calculate_quality_indicators(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality indicators for the response"""
        search_results = query_results.get("search_results", [])
        response = query_results.get("response", {})

        return {
            "source_diversity": len(set(r.get("source", "") for r in search_results)),
            "result_count": len(search_results),
            "response_length": len(str(response.get("content", ""))),
            "confidence_score": response.get("confidence", 0.0),
            "processing_strategy": self.active_strategy
        }

    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance tracking metrics"""
        self.query_count += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.query_count

    def set_search_strategy(self, strategy: str) -> bool:
        """
        Set the active search strategy

        Args:
            strategy: Strategy name to use

        Returns:
            True if strategy was set successfully
        """
        if strategy in self.available_strategies:
            self.active_strategy = strategy
            logger.info(f"Search strategy set to: {strategy}")
            return True
        else:
            logger.warning(f"Invalid search strategy: {strategy}. Available: {self.available_strategies}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        base_status = {
            "domain": self.domain_name,
            "components_initialized": self.components_initialized,
            "active_strategy": self.active_strategy,
            "available_strategies": self.available_strategies,
            "performance_metrics": {
                "total_queries": self.query_count,
                "average_processing_time": self.average_processing_time,
                "total_processing_time": self.total_processing_time
            },
            "initialization_results": self.initialization_results
        }

        # Add orchestrator status if available
        if self.universal_orchestrator:
            orchestrator_status = self.universal_orchestrator.get_system_status()
            base_status.update({
                "orchestrator_status": orchestrator_status,
                "system_stats": orchestrator_status.get("system_stats", {}),
                "discovered_types": orchestrator_status.get("discovered_types", {})
            })

        return base_status

    async def process_batch_queries(
        self,
        queries: List[str],
        max_results: int = 10,
        include_explanations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch

        Args:
            queries: List of queries to process
            max_results: Maximum results per query
            include_explanations: Whether to include explanations

        Returns:
            List of query results
        """
        logger.info(f"Processing batch of {len(queries)} queries for domain: {self.domain_name}")

        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i+1}/{len(queries)}: {query}")

            result = await self.process_query(
                query=query,
                max_results=max_results,
                include_explanations=include_explanations
            )

            result["batch_index"] = i
            results.append(result)

        logger.info(f"Completed batch processing of {len(queries)} queries")
        return results


def get_enhanced_rag_instance(domain_name: str = "general") -> EnhancedUniversalRAG:
    """
    Get or create Enhanced Universal RAG instance for a domain

    Args:
        domain_name: Domain name

    Returns:
        Enhanced Universal RAG instance
    """
    if domain_name not in enhanced_rag_instances:
        enhanced_rag_instances[domain_name] = EnhancedUniversalRAG(domain_name)

    return enhanced_rag_instances[domain_name]


async def initialize_enhanced_rag_system(
    domain_name: str = "general",
    text_files: Optional[List[Path]] = None,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Initialize Enhanced Universal RAG system for a domain

    Args:
        domain_name: Domain name to initialize
        text_files: Optional list of text files to process
        force_rebuild: Whether to force rebuild

    Returns:
        Initialization results
    """
    enhanced_rag = get_enhanced_rag_instance(domain_name)
    return await enhanced_rag.initialize_components(text_files, force_rebuild)


def list_enhanced_rag_instances() -> List[str]:
    """
    List all active Enhanced Universal RAG instances

    Returns:
        List of domain names with active instances
    """
    return list(enhanced_rag_instances.keys())


def cleanup_enhanced_rag_instances() -> None:
    """Clean up all Enhanced Universal RAG instances"""
    global enhanced_rag_instances
    enhanced_rag_instances.clear()
    logger.info("All Enhanced Universal RAG instances cleaned up")