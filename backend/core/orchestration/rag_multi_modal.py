"""
Multi-modal RAG pipeline implementation
Original approach using 3 separate vector searches (query + entities + concepts)
Used for research comparison and fallback scenarios
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.models.maintenance_models import (
    RAGResponse, EnhancedQuery, SearchResult, QueryAnalysis
)
from src.retrieval.vector_search import MaintenanceVectorSearch
from config.settings import settings
from .rag_base import MaintIERAGBase


logger = logging.getLogger(__name__)


class MaintIEMultiModalRAG(MaintIERAGBase):
    """Multi-modal RAG pipeline for maintenance intelligence (original approach)"""

    def __init__(self):
        """Initialize multi-modal RAG pipeline"""
        super().__init__("MultiModal")
        self.retrieval_method = "multi_modal_retrieval"
        self.api_calls_per_query = 3

    def initialize_components(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize all pipeline components"""
        return super().initialize_components(force_rebuild)

    def process_query(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """Process maintenance query through multi-modal RAG pipeline (3 API calls)"""

        if not self.components_initialized:
            logger.warning("Components not initialized, initializing now...")
            self.initialize_components()

        start_time = time.time()
        self.query_count += 1

        logger.info(f"Processing multi-modal query #{self.query_count}: {query}")

        try:
            # Step 1: Analyze and enhance query
            logger.info("Step 1: Analyzing query...")
            if not self.query_analyzer:
                raise ValueError("Query analyzer not initialized")

            analysis = self.query_analyzer.analyze_query(query)
            enhanced_query = self.query_analyzer.enhance_query(analysis)

            # Step 2: Multi-modal retrieval (3 separate vector searches)
            logger.info("Step 2: Multi-modal retrieval (3 API calls)...")
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

            logger.info(f"Multi-modal query processed successfully in {processing_time:.2f}s with confidence {response.confidence_score:.2f}")
            return response

        except Exception as e:
            logger.error(f"Error processing multi-modal query: {e}")
            return self._create_error_response(query, str(e), time.time() - start_time)

    def _multi_modal_retrieval(self, enhanced_query: EnhancedQuery, max_results: int) -> List[SearchResult]:
        """Perform multi-modal retrieval combining vector, entity, and concept search (3 API calls)"""

        if not self.vector_search:
            logger.error("Vector search not initialized")
            return []

        try:
            # Vector-based semantic search (API call 1)
            vector_results = self.vector_search.search(
                enhanced_query.analysis.original_query,
                top_k=max_results
            )

            # Entity-based search (API call 2)
            entity_query = " ".join(enhanced_query.analysis.entities)
            entity_results = self.vector_search.search(
                entity_query,
                top_k=max_results // 2
            ) if entity_query else []

            # Concept expansion search (API call 3)
            concept_query = " ".join(enhanced_query.expanded_concepts[:10])
            concept_results = self.vector_search.search(
                concept_query,
                top_k=max_results // 2
            ) if concept_query else []

            # Combine and rank results
            combined_results = self._fuse_search_results(
                vector_results, entity_results, concept_results
            )

            logger.info(f"Multi-modal retrieval: 3 API calls, {len(combined_results)} results")
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
                source="multi_modal_fusion",
                metadata={
                    **result.metadata,
                    "vector_score": scores["vector_score"],
                    "entity_score": scores["entity_score"],
                    "concept_score": scores["concept_score"],
                    "fusion_weights": {
                        "vector": vector_weight,
                        "entity": entity_weight,
                        "concept": graph_weight
                    },
                    "retrieval_method": "multi_modal_retrieval"
                },
                entities=result.entities
            )
            fused_results.append(fused_result)

        # Sort by fusion score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Fused {len(fused_results)} search results from multi-modal retrieval")
        return fused_results



    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        status = super().get_system_status()
        status.update({
            "retrieval_method": self.retrieval_method,
            "api_calls_per_query": self.api_calls_per_query
        })
        return status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        metrics = super().get_performance_metrics()
        metrics.update({
            "retrieval_method": self.retrieval_method,
            "api_calls_per_query": self.api_calls_per_query
        })
        return metrics