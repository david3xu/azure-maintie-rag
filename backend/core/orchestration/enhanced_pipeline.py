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

from .rag_orchestration_service import (
    AzureRAGOrchestrationService, create_universal_rag_from_texts, create_universal_rag_from_directory
)
from ...config.settings import settings

logger = logging.getLogger(__name__)

# Global instances for enhanced RAG systems by domain
enhanced_rag_instances: Dict[str, 'AzureRAGEnhancedPipeline'] = {}


class AzureRAGEnhancedPipeline:
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
        self.universal_orchestrator: Optional[AzureRAGOrchestrationService] = None
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
        progress_callback: Optional[Callable] = None,
        workflow_manager = None  # ✅ ADD: Accept workflow manager parameter
    ) -> Dict[str, Any]:
        """
        Process a query using Enhanced Universal RAG with detailed workflow progress tracking

        Args:
            query: User query to process
            max_results: Maximum number of results to return
            include_explanations: Whether to include explanations
            enable_safety_warnings: Whether to enable safety warnings
            stream_progress: Whether to stream progress updates
            progress_callback: Optional callback for progress updates (legacy support)
            workflow_manager: Optional workflow manager for detailed step tracking

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
            # ✅ CHANGE: Pass workflow_manager to universal orchestrator for detailed 7-step workflow
            query_results = await self.universal_orchestrator.process_query(
                query=query,
                max_results=max_results,
                include_explanations=include_explanations,
                stream_progress=stream_progress,
                progress_callback=progress_callback,  # Legacy support
                workflow_manager=workflow_manager  # ✅ Pass workflow manager for detailed steps
            )

            if not query_results.get("success", False):
                return {
                    "success": False,
                    "error": query_results.get("error", "Query processing failed"),
                    "query": query,
                    "domain": self.domain_name
                }

            # Enhanced results processing (post-processing)
            enhanced_results = await self._enhance_query_results(
                query_results,
                enable_safety_warnings,
                progress_callback
            )

            # Convert search_results to dicts if present and not already dicts
            if hasattr(enhanced_results, 'get') and "search_results" in enhanced_results:
                enhanced_results["search_results"] = [
                    r.to_dict() if hasattr(r, 'to_dict') else r
                    for r in enhanced_results["search_results"]
                ]

            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time)

            # Return enhanced results - handle both dict and UniversalRAGResponse formats
            if hasattr(enhanced_results, 'answer'):
                # UniversalRAGResponse format
                return {
                    "success": True,
                    "query": query,
                    "domain": self.domain_name,
                    "generated_response": {"answer": enhanced_results.answer, "confidence": enhanced_results.confidence},
                    "search_results": getattr(enhanced_results, 'sources', []),
                    "processing_time": processing_time,
                    "system_stats": getattr(enhanced_results, 'processing_metadata', {}),
                    "safety_warnings": [],
                    "enhancement_metrics": {"citations": len(getattr(enhanced_results, 'citations', []))},
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Dictionary format
                return {
                    "success": True,
                    "query": query,
                    "domain": self.domain_name,
                    "generated_response": enhanced_results.get("response", {}),
                    "search_results": enhanced_results.get("search_results", []),
                    "processing_time": processing_time,
                    "system_stats": enhanced_results.get("system_stats", {}),
                    "safety_warnings": enhanced_results.get("safety_warnings", []),
                    "enhancement_metrics": enhanced_results.get("enhancement_metrics", {}),
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Enhanced Universal RAG query processing failed: {e}", exc_info=True)

            # Handle workflow failure
            if workflow_manager:
                await workflow_manager.fail_workflow(f"Enhanced RAG processing failed: {str(e)}")

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

            # Handle both dict and UniversalRAGResponse formats
            if hasattr(query_results, 'to_dict'):
                # UniversalRAGResponse object
                enhanced = query_results.to_dict()
            elif isinstance(query_results, dict):
                # Dictionary format
                enhanced = query_results.copy()
            else:
                # Convert to dict format
                enhanced = {
                    "response": query_results,
                    "success": True
                }

            # Add safety warnings if enabled
            enhanced["safety_warnings"] = self._generate_safety_warnings(query_results) if enable_safety_warnings else []

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
            return enhanced

    def _generate_safety_warnings(self, query_results) -> List[str]:
        """Generate safety warnings based on response content"""
        warnings = []

        # Handle different query_results types
        if hasattr(query_results, 'answer'):
            response_text = str(query_results.answer).lower()
        elif hasattr(query_results, 'to_dict'):
            result_dict = query_results.to_dict()
            response_text = str(result_dict.get("answer", "")).lower()
        elif isinstance(query_results, dict):
            response_content = query_results.get("response", {})
            if isinstance(response_content, dict):
                response_text = str(response_content.get("content", "")).lower()
            else:
                response_text = str(response_content).lower()
        else:
            response_text = str(query_results).lower()

        # Generate domain-appropriate safety warnings
        if any(term in response_text for term in ["danger", "risk", "hazard", "safety"]):
            warnings.append("⚠️ This response contains safety-related information. Follow all safety protocols.")

        if any(term in response_text for term in ["electrical", "voltage", "current"]):
            warnings.append("⚡ Electrical work should only be performed by qualified personnel.")

        if any(term in response_text for term in ["chemical", "toxic", "corrosive"]):
            warnings.append("🧪 Handle chemical substances according to safety data sheets.")

        return warnings

    def _calculate_quality_indicators(self, query_results) -> Dict[str, Any]:
        """Calculate quality indicators for the response"""

        # Handle different query_results types
        if hasattr(query_results, 'sources'):
            # UniversalRAGResponse object
            search_results = query_results.sources or []
            response_length = len(str(query_results.answer)) if hasattr(query_results, 'answer') else 0
            confidence_score = getattr(query_results, 'confidence', 0.0)
        elif isinstance(query_results, dict):
            # Dictionary format
            search_results = query_results.get("search_results", [])
            response_content = query_results.get("response", {})
            if isinstance(response_content, dict):
                response_length = len(str(response_content.get("content", "")))
            else:
                response_length = len(str(response_content))
            confidence_score = query_results.get("confidence_score", 0.0)
        else:
            # Fallback
            search_results = []
            response_length = len(str(query_results))
            confidence_score = 0.0

        return {
            "source_diversity": len(set(r.get("source", "") if isinstance(r, dict) else getattr(r, 'source', '') for r in search_results)),
            "result_count": len(search_results),
            "response_length": response_length,
            "confidence_score": confidence_score,
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


def get_enhanced_rag_instance(domain_name: str = "general") -> AzureRAGEnhancedPipeline:
    """
    Get or create Enhanced Universal RAG instance for a domain

    Args:
        domain_name: Domain name

    Returns:
        Enhanced Universal RAG instance
    """
    if domain_name not in enhanced_rag_instances:
        enhanced_rag_instances[domain_name] = AzureRAGEnhancedPipeline(domain_name)

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