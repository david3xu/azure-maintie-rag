"""
Enhanced RAG pipeline orchestrator
Coordinates all components to deliver intelligent maintenance responses
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.models.maintenance_models import (
    RAGResponse, EnhancedQuery, SearchResult, QueryAnalysis
)
from src.knowledge.data_transformer import MaintIEDataTransformer
from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer
from src.retrieval.vector_search import MaintenanceVectorSearch
from src.generation.llm_interface import MaintenanceLLMInterface
from config.settings import settings


logger = logging.getLogger(__name__)


class MaintIEEnhancedRAG:
    """Main RAG pipeline orchestrator for maintenance intelligence"""

    def __init__(self):
        """Initialize enhanced RAG pipeline"""
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

        logger.info("MaintIEEnhancedRAG pipeline initialized")

    def initialize_components(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize all pipeline components"""
        logger.info("Initializing RAG pipeline components...")

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
            logger.info("All RAG pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            initialization_results["error"] = str(e)

        return initialization_results

    def _load_documents(self) -> Dict[str, Any]:
        """Load maintenance documents from processed data"""
        try:
            from src.retrieval.vector_search import load_documents_from_processed_data
            documents = load_documents_from_processed_data()
            logger.info(f"Loaded {len(documents)} documents for RAG pipeline")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return {}

    def process_query(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Process maintenance query through complete RAG pipeline"""

        if not self.components_initialized:
            logger.warning("Components not initialized, initializing now...")
            self.initialize_components()

        start_time = time.time()
        self.query_count += 1

        logger.info(f"Processing query #{self.query_count}: {query}")

        try:
            # Step 1: Analyze and enhance query
            logger.info("Step 1: Analyzing query...")
            if not self.query_analyzer:
                raise ValueError("Query analyzer not initialized")

            analysis = self.query_analyzer.analyze_query(query)
            enhanced_query = self.query_analyzer.enhance_query(analysis)

            # Step 2: Multi-modal retrieval
            logger.info("Step 2: Retrieving relevant documents...")
            search_results = self._multi_modal_retrieval(enhanced_query, max_results)

            # Step 3: Generate response
            logger.info("Step 3: Generating enhanced response...")
            if not self.llm_interface:
                raise ValueError("LLM interface not initialized")

            generation_result = self.llm_interface.generate_response(
                enhanced_query=enhanced_query,
                search_results=search_results,
                include_citations=True,
                include_safety_warnings=enable_safety_warnings
            )

            # Step 4: Create final response
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)

            response = RAGResponse(
                query=query,
                enhanced_query=enhanced_query,
                search_results=search_results,
                generated_response=generation_result["generated_response"],
                confidence_score=generation_result["confidence_score"],
                processing_time=processing_time,
                sources=generation_result["sources"],
                safety_warnings=generation_result["safety_warnings"],
                citations=generation_result["citations"]
            )

            logger.info(f"Query processed successfully in {processing_time:.2f}s with confidence {response.confidence_score:.2f}")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(query, str(e), time.time() - start_time)

    def _multi_modal_retrieval(self, enhanced_query: EnhancedQuery, max_results: int) -> List[SearchResult]:
        """Perform multi-modal retrieval combining vector, entity, and graph search"""

        if not self.vector_search:
            logger.error("Vector search not initialized")
            return []

        try:
            # Vector-based semantic search
            vector_results = self.vector_search.search(
                enhanced_query.analysis.original_query,
                top_k=max_results
            )

            # Entity-based search (simplified - using vector search with entity terms)
            entity_query = " ".join(enhanced_query.analysis.entities)
            entity_results = self.vector_search.search(
                entity_query,
                top_k=max_results // 2
            ) if entity_query else []

            # Concept expansion search
            concept_query = " ".join(enhanced_query.expanded_concepts[:10])
            concept_results = self.vector_search.search(
                concept_query,
                top_k=max_results // 2
            ) if concept_query else []

            # Combine and rank results
            combined_results = self._fuse_search_results(
                vector_results, entity_results, concept_results
            )

            return combined_results[:max_results]

        except Exception as e:
            logger.error(f"Error in multi-modal retrieval: {e}")
            return []

    def _fuse_search_results(
        self,
        vector_results: List[SearchResult],
        entity_results: List[SearchResult],
        concept_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fuse results from different search strategies"""

        # Create a dictionary to store combined scores
        doc_scores: Dict[str, Dict[str, Any]] = {}

        # Process vector results (highest weight)
        for result in vector_results:
            doc_scores[result.doc_id] = {
                "result": result,
                "vector_score": result.score,
                "entity_score": 0.0,
                "concept_score": 0.0
            }

        # Add entity search scores
        for result in entity_results:
            if result.doc_id in doc_scores:
                doc_scores[result.doc_id]["entity_score"] = result.score
            else:
                doc_scores[result.doc_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "entity_score": result.score,
                    "concept_score": 0.0
                }

        # Add concept search scores
        for result in concept_results:
            if result.doc_id in doc_scores:
                doc_scores[result.doc_id]["concept_score"] = result.score
            else:
                doc_scores[result.doc_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "entity_score": 0.0,
                    "concept_score": result.score
                }

        # Calculate fusion scores using weighted combination
        vector_weight = settings.vector_weight
        entity_weight = settings.entity_weight
        graph_weight = settings.graph_weight

        fused_results = []
        for doc_id, scores in doc_scores.items():
            fusion_score = (
                scores["vector_score"] * vector_weight +
                scores["entity_score"] * entity_weight +
                scores["concept_score"] * graph_weight
            )

            # Create new result with fusion score
            result = scores["result"]
            fused_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,
                content=result.content,
                score=fusion_score,
                source="hybrid_fusion",
                metadata={
                    **result.metadata,
                    "vector_score": scores["vector_score"],
                    "entity_score": scores["entity_score"],
                    "concept_score": scores["concept_score"],
                    "fusion_weights": {
                        "vector": vector_weight,
                        "entity": entity_weight,
                        "concept": graph_weight
                    }
                },
                entities=result.entities
            )
            fused_results.append(fused_result)

        # Sort by fusion score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Fused {len(fused_results)} search results")
        return fused_results

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
            "performance_within_target": self.average_processing_time <= settings.max_query_time
        }

    def validate_pipeline_health(self) -> Dict[str, Any]:
        """Validate pipeline health and readiness"""
        health = {
            "overall_status": "healthy",
            "components": {},
            "issues": [],
            "recommendations": []
        }

        # Check component initialization
        components = {
            "data_transformer": self.data_transformer,
            "query_analyzer": self.query_analyzer,
            "vector_search": self.vector_search,
            "llm_interface": self.llm_interface
        }

        for name, component in components.items():
            if component is None:
                health["components"][name] = "not_initialized"
                health["issues"].append(f"{name} not initialized")
                health["overall_status"] = "unhealthy"
            else:
                health["components"][name] = "healthy"

        # Check data availability
        if self.vector_search and hasattr(self.vector_search, 'documents'):
            doc_count = len(self.vector_search.documents)
            if doc_count == 0:
                health["issues"].append("No documents loaded in vector search")
                health["overall_status"] = "degraded"
            else:
                health["components"]["document_store"] = f"{doc_count} documents loaded"

        # Check performance
        if self.average_processing_time > settings.max_query_time:
            health["issues"].append(f"Average processing time ({self.average_processing_time:.2f}s) exceeds target ({settings.max_query_time}s)")
            health["recommendations"].append("Consider optimizing retrieval or generation parameters")

        # Add recommendations based on issues
        if health["issues"]:
            if "not_initialized" in str(health["issues"]):
                health["recommendations"].append("Run initialize_components() to set up pipeline")
            if "No documents" in str(health["issues"]):
                health["recommendations"].append("Ensure MaintIE data is processed and documents are loaded")

        return health

    async def process_query_async(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Async version of query processing"""
        # For now, just wrap the sync method
        # In production, could implement true async processing
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_query,
            query,
            max_results,
            include_explanations,
            enable_safety_warnings
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

    # Test query processing
    if init_results.get("data_transformer") and init_results.get("vector_search"):
        print("\nTesting query processing...")

        test_queries = [
            "How to troubleshoot pump seal failure?",
            "Preventive maintenance schedule for centrifugal pump",
            "Safety procedures for motor bearing replacement"
        ]

        for query in test_queries:
            print(f"\nProcessing: {query}")
            try:
                response = rag.process_query(query, max_results=5)
                print(f"Response: {response.generated_response[:200]}...")
                print(f"Confidence: {response.confidence_score:.2f}")
                print(f"Processing time: {response.processing_time:.2f}s")
                print(f"Sources: {len(response.sources)}")
            except Exception as e:
                print(f"Error: {e}")

    # Show system status
    print("\nSystem Status:")
    status = rag.get_system_status()
    print(json.dumps(status, indent=2))
