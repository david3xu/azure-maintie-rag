"""
Enhanced RAG pipeline orchestrator
Coordinates all components to deliver intelligent maintenance responses
Now uses separated multi-modal and structured RAG implementations
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.models.maintenance_models import (
    RAGResponse, EnhancedQuery, SearchResult, QueryAnalysis
)
from config.settings import settings
from .rag_multi_modal import MaintIEMultiModalRAG
from .rag_structured import MaintIEStructuredRAG


logger = logging.getLogger(__name__)


class MaintIEEnhancedRAG:
    """Main RAG pipeline orchestrator for maintenance intelligence"""

    def __init__(self):
        """Initialize enhanced RAG pipeline"""
        # Initialize both RAG implementations
        self.multi_modal_rag = MaintIEMultiModalRAG()
        self.structured_rag = MaintIEStructuredRAG()

        # Track which implementation is active
        self.active_implementation = "multi_modal"  # Default to original approach

        # Track initialization status
        self.components_initialized = False
        self.knowledge_loaded = False

        logger.info("MaintIEEnhancedRAG pipeline initialized with dual implementations")

    def initialize_components(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize all pipeline components"""
        logger.info("Initializing dual RAG pipeline components...")

        initialization_results = {
            "multi_modal_rag": False,
            "structured_rag": False,
            "knowledge_loaded": False,
            "total_documents": 0,
            "total_entities": 0,
            "index_built": False
        }

        try:
            # Initialize multi-modal RAG (original approach)
            logger.info("Step 1: Initializing multi-modal RAG...")
            multi_modal_results = self.multi_modal_rag.initialize_components(force_rebuild)
            initialization_results["multi_modal_rag"] = multi_modal_results.get("data_transformer", False)

            # Initialize structured RAG (optimized approach)
            logger.info("Step 2: Initializing structured RAG...")
            structured_results = self.structured_rag.initialize_components(force_rebuild)
            initialization_results["structured_rag"] = structured_results.get("data_transformer", False)

            # Use results from either implementation
            if multi_modal_results.get("knowledge_loaded"):
                initialization_results["knowledge_loaded"] = True
                self.knowledge_loaded = True
                initialization_results["total_documents"] = multi_modal_results.get("total_documents", 0)
                initialization_results["total_entities"] = multi_modal_results.get("total_entities", 0)
                initialization_results["index_built"] = multi_modal_results.get("index_built", False)
            elif structured_results.get("knowledge_loaded"):
                initialization_results["knowledge_loaded"] = True
                self.knowledge_loaded = True
                initialization_results["total_documents"] = structured_results.get("total_documents", 0)
                initialization_results["total_entities"] = structured_results.get("total_entities", 0)
                initialization_results["index_built"] = structured_results.get("index_built", False)

            # Set components_initialized based on both implementations
            self.components_initialized = (
                self.multi_modal_rag.components_initialized and
                self.structured_rag.components_initialized
            )

            logger.info("All dual RAG pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing dual RAG pipeline: {e}")
            initialization_results["error"] = str(e)

        return initialization_results

    def process_query(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Process maintenance query using active RAG implementation"""

        # Ensure components are initialized
        if not self.components_initialized:
            logger.warning("Components not initialized, initializing now...")
            self.initialize_components()

        # Delegate to the active implementation
        if self.active_implementation == "multi_modal":
            return self.multi_modal_rag.process_query(
                query, max_results, include_explanations, enable_safety_warnings
            )
        else:
            # For structured RAG, we need to use the async method
            import asyncio
            return asyncio.run(self.structured_rag.process_query_optimized(
                query, max_results, include_explanations, enable_safety_warnings
            ))

    def set_active_implementation(self, implementation: str) -> None:
        """Set the active RAG implementation"""
        if implementation in ["multi_modal", "structured"]:
            self.active_implementation = implementation
            logger.info(f"Active RAG implementation set to: {implementation}")
        else:
            raise ValueError("Implementation must be 'multi_modal' or 'structured'")

    def get_active_implementation(self) -> str:
        """Get the currently active RAG implementation"""
        return self.active_implementation

    def process_query_multi_modal(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Process query using multi-modal RAG (original approach)"""
        return self.multi_modal_rag.process_query(
            query, max_results, include_explanations, enable_safety_warnings
        )

    async def process_query_structured(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Process query using structured RAG (optimized approach)"""
        return await self.structured_rag.process_query_optimized(
            query, max_results, include_explanations, enable_safety_warnings
        )

    async def process_query_optimized(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Process query using optimized structured RAG (for comparison endpoint)"""
        return await self.structured_rag.process_query_optimized(
            query, max_results, include_explanations, enable_safety_warnings
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        status = {
            "active_implementation": self.active_implementation,
            "multi_modal_rag": self.multi_modal_rag.get_system_status(),
            "structured_rag": self.structured_rag.get_system_status(),
            "components_initialized": (
                self.multi_modal_rag.components_initialized and
                self.structured_rag.components_initialized
            ),
            "knowledge_loaded": (
                self.multi_modal_rag.knowledge_loaded or
                self.structured_rag.knowledge_loaded
            )
        }

        return status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            "active_implementation": self.active_implementation,
            "multi_modal_rag": self.multi_modal_rag.get_performance_metrics(),
            "structured_rag": self.structured_rag.get_performance_metrics(),
            "max_query_time_setting": settings.max_query_time
        }

    def validate_pipeline_health(self) -> Dict[str, Any]:
        """Validate pipeline health and readiness"""
        health = {
            "overall_status": "healthy",
            "active_implementation": self.active_implementation,
            "multi_modal_rag": self.multi_modal_rag.validate_pipeline_health() if hasattr(self.multi_modal_rag, 'validate_pipeline_health') else {"status": "not_implemented"},
            "structured_rag": self.structured_rag.validate_pipeline_health() if hasattr(self.structured_rag, 'validate_pipeline_health') else {"status": "not_implemented"},
            "issues": [],
            "recommendations": []
        }

        # Check if both implementations are initialized
        if not self.multi_modal_rag.components_initialized:
            health["issues"].append("Multi-modal RAG not initialized")
            health["overall_status"] = "unhealthy"

        if not self.structured_rag.components_initialized:
            health["issues"].append("Structured RAG not initialized")
            health["overall_status"] = "unhealthy"

        # Add recommendations based on issues
        if health["issues"]:
            if "not initialized" in str(health["issues"]):
                health["recommendations"].append("Run initialize_components() to set up pipeline")

        return health

    async def process_query_async(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Async version of query processing"""
        # Delegate to the active implementation
        if self.active_implementation == "multi_modal":
            # For multi-modal, wrap the sync method
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.multi_modal_rag.process_query,
                query,
                max_results,
                include_explanations,
                enable_safety_warnings
            )
        else:
            # For structured, use the async method directly
            return await self.structured_rag.process_query_optimized(
                query, max_results, include_explanations, enable_safety_warnings
            )


# Global RAG instance
_rag_instance: Optional[MaintIEEnhancedRAG] = None


def get_rag_instance() -> MaintIEEnhancedRAG:
    """Get or create global RAG instance (singleton pattern)"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MaintIEEnhancedRAG()
    return _rag_instance


def initialize_rag_system(force_rebuild: bool = False) -> Dict[str, Any]:
    """Initialize the RAG system with all components"""
    rag = get_rag_instance()
    return rag.initialize_components(force_rebuild)


if __name__ == "__main__":
    # Example usage and testing
    import json

    # Initialize RAG system
    print("Initializing MaintIE Enhanced RAG system...")
    rag = MaintIEEnhancedRAG()
    init_results = rag.initialize_components()

    print("Initialization Results:")
    print(json.dumps(init_results, indent=2))

    # Test query processing with both implementations
    if init_results.get("multi_modal_rag") and init_results.get("structured_rag"):
        print("\nTesting query processing...")

        test_queries = [
            "How to troubleshoot pump seal failure?",
            "Preventive maintenance schedule for centrifugal pump",
            "Safety procedures for motor bearing replacement"
        ]

        for query in test_queries:
            print(f"\nProcessing: {query}")

            # Test multi-modal approach
            print("  Multi-modal approach:")
            try:
                rag.set_active_implementation("multi_modal")
                response = rag.process_query(query, max_results=5)
                print(f"    Response: {response.generated_response[:200]}...")
                print(f"    Confidence: {response.confidence_score:.2f}")
                print(f"    Processing time: {response.processing_time:.2f}s")
                print(f"    Sources: {len(response.sources)}")
            except Exception as e:
                print(f"    Error: {e}")

            # Test structured approach
            print("  Structured approach:")
            try:
                rag.set_active_implementation("structured")
                response = rag.process_query(query, max_results=5)
                print(f"    Response: {response.generated_response[:200]}...")
                print(f"    Confidence: {response.confidence_score:.2f}")
                print(f"    Processing time: {response.processing_time:.2f}s")
                print(f"    Sources: {len(response.sources)}")
            except Exception as e:
                print(f"    Error: {e}")

    # Show system status
    print("\nSystem Status:")
    status = rag.get_system_status()
    print(json.dumps(status, indent=2))
