"""
Structured RAG pipeline implementation
Optimized approach using 1 vector search + graph-enhanced ranking
Used for production performance and enhanced domain understanding
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


class MaintIEStructuredRAG(MaintIERAGBase):
    """Structured RAG pipeline for maintenance intelligence (optimized approach)"""

    def __init__(self):
        """Initialize structured RAG pipeline"""
        super().__init__("Structured")
        self.retrieval_method = "optimized_structured_rag"
        self.api_calls_per_query = 1

    def initialize_components(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize all pipeline components"""
        return super().initialize_components(force_rebuild)

    async def process_query_optimized(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ) -> RAGResponse:
        """
        Process maintenance query using optimized structured RAG approach (1 API call + graph operations)

        This method implements the three innovation points:
        1. Domain Understanding: Enhanced query analysis with maintenance context
        2. Structured Knowledge: Graph-enhanced retrieval instead of multiple vector calls
        3. Intelligent Retrieval: Single API call with structured ranking
        """

        if not self.components_initialized:
            logger.warning("Components not initialized, initializing now...")
            self.initialize_components()

        start_time = time.time()
        self.query_count += 1

        logger.info(f"Processing structured query #{self.query_count}: {query}")

        try:
            # Step 1: Enhanced domain understanding (same as original but with maintenance context)
            logger.info("Step 1: Enhanced domain analysis...")
            if not self.query_analyzer:
                raise ValueError("Query analyzer not initialized")

            analysis = self.query_analyzer.analyze_query(query)
            enhanced_query = self.query_analyzer.enhance_query(analysis)

            # Step 2: Structured retrieval (NEW - replaces multi-modal with single optimized call)
            logger.info("Step 2: Structured retrieval (1 API call)...")
            search_results = self._structured_retrieval(enhanced_query, max_results)

            # Step 3: Generate response (same as original for quality consistency)
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

            logger.info(f"Structured query processed successfully in {processing_time:.2f}s with confidence {response.confidence_score:.2f}")
            return response

        except Exception as e:
            logger.error(f"Error in structured processing: {e}")
            return self._create_error_response(query, str(e), time.time() - start_time)

    def process_query(
        self,
        query: str,
        max_results: int = 10,
        include_explanations: bool = True,
        enable_safety_warnings: bool = True
    ):
        """Synchronous wrapper for process_query_optimized for compatibility with test suite and orchestrator."""
        import asyncio
        return asyncio.run(self.process_query_optimized(
            query, max_results, include_explanations, enable_safety_warnings
        ))

    def _structured_retrieval(self, enhanced_query: EnhancedQuery, max_results: int) -> List[SearchResult]:
        """
        Structured RAG: Single API call + graph-enhanced ranking

        Innovation Point 2 & 3: Structured Knowledge + Intelligent Retrieval
        - Builds comprehensive query using domain knowledge
        - Single vector search instead of 3 separate calls
        - Applies knowledge graph intelligence to ranking
        """

        if not self.vector_search:
            logger.error("Vector search not initialized")
            return []

        try:
            # Step 1: Build structured query using domain knowledge
            structured_query = self._build_structured_query(enhanced_query)

            # Step 2: Single vector search with expanded context
            base_results = self.vector_search.search(structured_query, top_k=max_results * 2)

            # Step 3: Apply knowledge graph intelligence to ranking
            enhanced_results = self._apply_knowledge_graph_ranking(base_results, enhanced_query)

            logger.info(f"Structured retrieval: 1 API call, {len(enhanced_results)} results")
            return enhanced_results[:max_results]

        except Exception as e:
            logger.error(f"Error in structured retrieval: {e}")
            # Fallback to simple search
            return self.vector_search.search(enhanced_query.analysis.original_query, top_k=max_results)

    def _build_structured_query(self, enhanced_query: EnhancedQuery) -> str:
        """
        Build comprehensive query using domain knowledge

        Innovation Point 1: Domain Understanding
        - Combines original query with relevant entities and concepts
        - Uses maintenance domain context to prioritize terms
        - Creates single comprehensive query for vector search
        """

        # Base query
        query_parts = [enhanced_query.analysis.original_query]

        # Add entities with domain context (limit to prevent noise)
        if enhanced_query.analysis.entities:
            important_entities = enhanced_query.analysis.entities[:5]
            query_parts.extend(important_entities)

        # Add concept expansion with graph structure
        if enhanced_query.expanded_concepts:
            # Use knowledge graph to select most relevant concepts
            relevant_concepts = self._select_relevant_concepts(
                enhanced_query.expanded_concepts,
                enhanced_query.analysis.entities
            )
            query_parts.extend(relevant_concepts[:5])  # Limit to top concepts

        structured_query = " ".join(query_parts)
        logger.info(f"Structured query built: {structured_query[:100]}...")
        return structured_query

    def _select_relevant_concepts(self, concepts: List[str], entities: List[str]) -> List[str]:
        """
        Use knowledge graph to select most relevant concepts

        TODO: Replace with actual knowledge graph operations
        Current implementation uses simple term matching
        """

        if not concepts:
            return []

        # For now, prioritize concepts that relate to identified entities
        # TODO: Replace with actual knowledge graph traversal
        relevant_concepts = []
        entity_terms = set(e.lower() for e in entities)

        for concept in concepts:
            concept_terms = set(concept.lower().split())
            # If concept shares terms with entities, it's likely relevant
            if entity_terms.intersection(concept_terms):
                relevant_concepts.append(concept)
            elif len(relevant_concepts) < 3:  # Always include some concepts
                relevant_concepts.append(concept)

        return relevant_concepts[:5]

    def _apply_knowledge_graph_ranking(self, base_results: List[SearchResult],
                                     enhanced_query: EnhancedQuery) -> List[SearchResult]:
        """
        Apply structured knowledge to enhance ranking

        Innovation Point 2: Structured Knowledge
        - Calculates knowledge graph relevance scores
        - Combines vector similarity with domain knowledge
        - Enhances ranking based on maintenance context
        """

        # Extract query context for ranking
        query_entities = set(e.lower() for e in enhanced_query.analysis.entities)
        query_concepts = set(c.lower() for c in enhanced_query.expanded_concepts)

        enhanced_results = []
        for result in base_results:

            # Calculate knowledge graph relevance score
            kg_score = self._calculate_knowledge_relevance(
                result, query_entities, query_concepts
            )

            # Combine vector similarity with knowledge graph relevance
            # Preserve original vector score but enhance with domain knowledge
            enhanced_score = 0.7 * result.score + 0.3 * kg_score

            enhanced_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,
                content=result.content,
                score=enhanced_score,
                source="structured_rag",
                metadata={
                    **result.metadata,
                    "original_vector_score": result.score,
                    "knowledge_graph_score": kg_score,
                    "enhancement_method": "structured_ranking",
                    "retrieval_method": "optimized_structured_rag"
                },
                entities=result.entities
            )
            enhanced_results.append(enhanced_result)

        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x.score, reverse=True)
        return enhanced_results

    def _calculate_knowledge_relevance(self, result: SearchResult,
                                     query_entities: set, query_concepts: set) -> float:
        """
        Calculate knowledge graph-based relevance score

        TODO: Replace with actual knowledge graph operations
        Current implementation uses simple term matching
        """

        # Extract document terms
        doc_text = f"{result.title} {result.content}".lower()
        doc_terms = set(doc_text.split())

        # Entity match score
        entity_matches = len(query_entities.intersection(doc_terms)) / max(len(query_entities), 1)

        # Concept match score
        concept_matches = len(query_concepts.intersection(doc_terms)) / max(len(query_concepts), 1)

        # Combined knowledge relevance (can be enhanced with actual graph operations)
        knowledge_score = 0.6 * entity_matches + 0.4 * concept_matches

        return min(knowledge_score, 1.0)  # Normalize to [0,1]



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