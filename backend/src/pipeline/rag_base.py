"""
Base RAG pipeline class
Shared functionality for both multi-modal and structured RAG approaches
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.models.maintenance_models import (
    RAGResponse, EnhancedQuery, SearchResult, QueryAnalysis
)
from src.knowledge.data_transformer import MaintIEDataTransformer
from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer
from src.retrieval.vector_search import MaintenanceVectorSearch
from src.generation.llm_interface import MaintenanceLLMInterface
from config.settings import settings


logger = logging.getLogger(__name__)


class MaintIERAGBase:
    """Base class for MaintIE RAG pipelines"""

    def __init__(self, pipeline_type: str):
        """Initialize base RAG pipeline"""
        self.pipeline_type = pipeline_type
        self.components_initialized = False
        self.knowledge_loaded = False

        # Core components
        self.data_transformer: Optional[MaintIEDataTransformer] = None
        self.query_analyzer: Optional[MaintenanceQueryAnalyzer] = None
        self.vector_search: Optional[MaintenanceVectorSearch] = None
        self.llm_interface: Optional[MaintenanceLLMInterface] = None

        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0

        logger.info(f"MaintIE{pipeline_type}RAG pipeline initialized")

    def initialize_components(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize all pipeline components"""
        logger.info(f"Initializing {self.pipeline_type} RAG pipeline components...")

        initialization_results = {
            "data_transformer": False,
            "query_analyzer": False,
            "vector_search": False,
            "llm_interface": False,
            "knowledge_loaded": False,
            "total_documents": 0,
            "total_entities": 0,
            "index_built": False
        }

        try:
            # 1. Initialize data transformer and load knowledge
            logger.info("Step 1: Initializing data transformer...")
            self.data_transformer = MaintIEDataTransformer()

            # Check if processed data exists or force rebuild
            processed_entities_path = settings.processed_data_dir / "maintenance_entities.json"

            if not processed_entities_path.exists() or force_rebuild:
                logger.info("Processing MaintIE data...")
                knowledge_stats = self.data_transformer.extract_maintenance_knowledge()
                initialization_results.update(knowledge_stats)
            else:
                logger.info("Using existing processed data")

            initialization_results["data_transformer"] = True
            self.knowledge_loaded = True
            initialization_results["knowledge_loaded"] = True

            # 2. Initialize query analyzer
            logger.info("Step 2: Initializing query analyzer...")
            self.query_analyzer = MaintenanceQueryAnalyzer(self.data_transformer)
            initialization_results["query_analyzer"] = True

            # 3. Initialize vector search
            logger.info("Step 3: Initializing vector search...")
            self.vector_search = MaintenanceVectorSearch()

            # Load documents and build/load index
            documents = self._load_documents()
            initialization_results["total_documents"] = len(documents)

            if documents:
                # Check if index exists
                index_path = settings.indices_dir / "faiss_index.bin"
                if not index_path.exists() or force_rebuild:
                    logger.info("Building vector search index...")
                    try:
                        self.vector_search.build_index(documents)
                        initialization_results["index_built"] = True
                        logger.info("Vector search index built successfully")
                    except Exception as e:
                        logger.warning(f"Could not build vector search index: {e}")
                        logger.info("Vector search will be available but may have limited functionality")
                else:
                    logger.info("Using existing vector search index")
                    # Load existing documents into search
                    self.vector_search.documents = documents

            initialization_results["vector_search"] = True

            # 4. Initialize LLM interface
            logger.info("Step 4: Initializing LLM interface...")
            self.llm_interface = MaintenanceLLMInterface()
            initialization_results["llm_interface"] = True

            self.components_initialized = True
            logger.info(f"All {self.pipeline_type} RAG pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing {self.pipeline_type} RAG pipeline: {e}")
            initialization_results["error"] = str(e)

        return initialization_results

    def _load_documents(self) -> Dict[str, Any]:
        """Load maintenance documents from processed data"""
        try:
            from src.retrieval.vector_search import load_documents_from_processed_data
            documents = load_documents_from_processed_data()
            logger.info(f"Loaded {len(documents)} documents for {self.pipeline_type} RAG pipeline")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return {}

    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance tracking metrics"""
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.query_count

    def _create_error_response(self, query: str, error_message: str, processing_time: float) -> RAGResponse:
        """Create error response when processing fails"""

        # Create minimal analysis
        analysis = QueryAnalysis(
            original_query=query,
            query_type="informational",
            entities=[],
            intent="error",
            complexity="unknown"
        )

        enhanced_query = EnhancedQuery(
            analysis=analysis,
            expanded_concepts=[],
            related_entities=[],
            domain_context={},
            structured_search="",
            safety_considerations=[]
        )

        error_response = (
            f"I apologize, but I encountered an error while processing your maintenance query: '{query}'. "
            f"Please try rephrasing your question or contact technical support if the issue persists.\n\n"
            f"In the meantime, please ensure you follow all safety procedures and consult qualified "
            f"maintenance personnel for any critical equipment issues."
        )

        return RAGResponse(
            query=query,
            enhanced_query=enhanced_query,
            search_results=[],
            generated_response=error_response,
            confidence_score=0.1,
            processing_time=processing_time,
            sources=[],
            safety_warnings=["Always follow proper safety procedures", "Consult qualified personnel"],
            citations=[]
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        status = {
            "components_initialized": self.components_initialized,
            "knowledge_loaded": self.knowledge_loaded,
            "total_queries_processed": self.query_count,
            "average_processing_time": round(self.average_processing_time, 3),
            "pipeline_type": self.pipeline_type,
            "components": {
                "data_transformer": self.data_transformer is not None,
                "query_analyzer": self.query_analyzer is not None,
                "vector_search": self.vector_search is not None,
                "llm_interface": self.llm_interface is not None
            }
        }

        # Add vector search stats if available
        if self.vector_search:
            try:
                vector_stats = self.vector_search.get_index_stats()
                status["vector_search_stats"] = vector_stats
            except:
                status["vector_search_stats"] = {"error": "Could not retrieve stats"}

        return status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            "query_count": self.query_count,
            "total_processing_time": round(self.total_processing_time, 3),
            "average_processing_time": round(self.average_processing_time, 3),
            "max_query_time_setting": settings.max_query_time,
            "performance_within_target": self.average_processing_time <= settings.max_query_time,
            "pipeline_type": self.pipeline_type
        }