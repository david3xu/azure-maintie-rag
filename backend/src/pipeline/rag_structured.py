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
        """Initialize structured RAG pipeline with graph optimization and caching"""
        super().__init__("Structured")
        self.retrieval_method = "optimized_structured_rag"
        self.api_calls_per_query = 1

        # Initialize caching
        self.response_cache = None
        self.caching_enabled = True
        self._init_caching()

        # Initialize graph optimization components
        self.data_transformer = None
        self.entity_index = None
        self.graph_ranker = None
        self.graph_operations_enabled = False

        self._init_graph_components()

    def _init_caching(self):
        """Initialize response caching"""
        try:
            from src.cache.response_cache import ResponseCache
            self.response_cache = ResponseCache()
            logger.info("Response caching initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize caching: {e}")
            self.caching_enabled = False

    def _init_graph_components(self):
        """Initialize graph optimization components with better error handling"""
        try:
            from src.knowledge.data_transformer import MaintIEDataTransformer
            from src.knowledge.entity_document_index import EntityDocumentIndex
            from src.retrieval.graph_enhanced_ranking import GraphEnhancedRanker

            logger.info("Initializing graph optimization components...")

            # Initialize data transformer with caching
            self.data_transformer = MaintIEDataTransformer()

            # Check if knowledge graph already exists
            if (hasattr(self.data_transformer, 'knowledge_graph') and
                self.data_transformer.knowledge_graph):
                logger.info("Knowledge graph already exists, skipping extraction")
            else:
                # Try to load existing data first
                if self.data_transformer.load_existing_processed_data():
                    logger.info("Successfully loaded existing processed data")
                else:
                    logger.info("No existing data found, extracting knowledge...")
                    self.data_transformer.extract_maintenance_knowledge()

            # Initialize entity-document index with validation
            self.entity_index = EntityDocumentIndex(self.data_transformer)

            # Only build index if we have valid data
            if (self.data_transformer and
                hasattr(self.data_transformer, 'documents') and
                self.data_transformer.documents):

                index_stats = self.entity_index.build_index()
                logger.info(f"Entity index built: {index_stats}")
            else:
                logger.warning("No documents available for entity index building")

            # Initialize graph ranker only if prerequisites are met
            if (self.data_transformer and
                getattr(self.data_transformer, 'knowledge_graph', None) and
                self.entity_index and
                getattr(self.entity_index, 'index_built', False)):

                self.graph_ranker = GraphEnhancedRanker(
                    self.data_transformer, self.entity_index
                )
                self.graph_operations_enabled = True
                logger.info("Graph optimization enabled")
            else:
                logger.warning("Graph optimization disabled: missing prerequisites")
                self.graph_operations_enabled = False

        except Exception as e:
            logger.error(f"Error initializing graph components: {e}")
            self.graph_operations_enabled = False

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
        """Process query with caching optimization"""
        from src.monitoring.pipeline_monitor import get_monitor

        # Initialize monitoring
        monitor = get_monitor()
        query_id = monitor.start_query(query, "structured")

        # Check cache first
        if self.caching_enabled and self.response_cache:
            with monitor.track_sub_step("Cache Check", "MaintIEStructuredRAG", query):
                cached_response = self.response_cache.get_cached_response(query, max_results)
                if cached_response:
                    logger.info(f"Returning cached response for: {query[:50]}...")
                    monitor.track_cache_hit("Cache Check")
                    monitor.end_query(
                        confidence_score=cached_response.confidence_score,
                        sources_count=len(cached_response.sources),
                        safety_warnings_count=len(cached_response.safety_warnings)
                    )
                    return cached_response

        # Process query normally
        start_time = time.time()
        self.query_count += 1

        logger.info(f"Processing structured query #{self.query_count}: {query}")

        try:
            # Step 1: Enhanced domain understanding (same as original but with maintenance context)
            logger.info("Step 1: Enhanced domain analysis...")
            if not self.query_analyzer:
                raise ValueError("Query analyzer not initialized")

            analysis_start = time.time()
            with monitor.track_sub_step("Query Analysis and Enhancement", "MaintIEStructuredRAG", query):
                analysis = self.query_analyzer.analyze_query(query)
                enhanced_query = self.query_analyzer.enhance_query(analysis)
                monitor.add_custom_metric("Query Analysis and Enhancement", "entities_count", len(analysis.entities))
                monitor.add_custom_metric("Query Analysis and Enhancement", "concepts_count", len(enhanced_query.expanded_concepts))
                # Defensive check: if enhanced_query is None, create a minimal one
                if enhanced_query is None:
                    logger.warning(f"Enhanced query is None for query: {query}, creating fallback")
                    enhanced_query = EnhancedQuery(
                        analysis=analysis,
                        expanded_concepts=analysis.entities if analysis.entities else [],
                        related_entities=[],
                        domain_context={},
                        structured_search=query,
                        safety_considerations=[],
                        safety_critical=False,
                        safety_warnings=[],
                        equipment_category=None,
                        maintenance_context={}
                    )
            analysis_time = time.time() - analysis_start
            if analysis_time > 0.1:
                logger.warning(f"Slow query analysis: {analysis_time:.3f}s")

            # Step 2: Structured retrieval (NEW - replaces multi-modal with single optimized call)
            logger.info("Step 2: Structured retrieval (1 API call)...")
            retrieval_start = time.time()
            with monitor.track_sub_step("Structured Retrieval", "MaintIEStructuredRAG", enhanced_query):
                search_results = self._optimized_structured_retrieval(enhanced_query, max_results)
                monitor.add_custom_metric("Structured Retrieval", "results_count", len(search_results))
                monitor.add_custom_metric("Structured Retrieval", "top_score", search_results[0].score if search_results else 0.0)
            retrieval_time = time.time() - retrieval_start
            if retrieval_time > 0.1:
                logger.warning(f"Slow retrieval: {retrieval_time:.3f}s")

            # Step 3: Generate response (same as original for quality consistency)
            logger.info("Step 3: Generating enhanced response...")
            if not self.llm_interface:
                raise ValueError("LLM interface not initialized")
            generation_start = time.time()
            with monitor.track_sub_step("Response Generation", "MaintIEStructuredRAG", search_results):
                generation_result = self.llm_interface.generate_response(
                    enhanced_query=enhanced_query,
                    search_results=search_results,
                    include_citations=True,
                    include_safety_warnings=enable_safety_warnings
                )
                monitor.add_custom_metric("Response Generation", "response_length", len(generation_result["generated_response"]))
                monitor.add_custom_metric("Response Generation", "confidence_score", generation_result["confidence_score"])
                monitor.add_custom_metric("Response Generation", "sources_count", len(generation_result["sources"]))
            generation_time = time.time() - generation_start
            if generation_time > 0.1:
                logger.warning(f"Slow response generation: {generation_time:.3f}s")

            # Step 4: Create final response
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)

            with monitor.track_sub_step("Response Assembly", "MaintIEStructuredRAG"):
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

            # Cache the response if high confidence
            if (self.caching_enabled and self.response_cache and
                response.confidence_score > 0.6):
                with monitor.track_sub_step("Response Caching", "MaintIEStructuredRAG"):
                    self.response_cache.cache_response(query, response, max_results)
                    monitor.add_custom_metric("Response Caching", "cached", True)

            # End monitoring and get metrics
            metrics = monitor.end_query(
                confidence_score=response.confidence_score,
                sources_count=len(response.sources),
                safety_warnings_count=len(response.safety_warnings)
            )

            logger.info(f"Structured query processed successfully in {processing_time:.2f}s with confidence {response.confidence_score:.2f}")
            logger.info(f"Pipeline metrics: {metrics.total_steps} steps, {metrics.total_api_calls} API calls, {metrics.cache_hits} cache hits")
            return response

        except Exception as e:
            logger.error(f"Error in structured processing: {e}")
            # End monitoring with error
            monitor.end_query()
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

    def _optimized_structured_retrieval(self, enhanced_query: EnhancedQuery,
                                       max_results: int) -> List[SearchResult]:
        """Optimized structured retrieval using entity index and graph ranking"""
        from src.monitoring.pipeline_monitor import get_monitor

        monitor = get_monitor()

        if not self.vector_search:
            logger.error("Vector search not initialized")
            return []

        try:
            # Step 1: Build optimized query
            with monitor.track_sub_step("Query Building", "MaintIEStructuredRAG", enhanced_query):
                structured_query = self._build_structured_query(enhanced_query)
                monitor.add_custom_metric("Query Building", "query_length", len(structured_query))
                monitor.add_custom_metric("Query Building", "entities_count", len(enhanced_query.analysis.entities))

            # Step 2: Single vector search (optimization: 1 API call instead of 3)
            with monitor.track_sub_step("Vector Search", "MaintIEStructuredRAG", structured_query):
                base_results = self.vector_search.search(structured_query, top_k=max_results * 2)
                monitor.add_custom_metric("Vector Search", "base_results_count", len(base_results))
                monitor.add_custom_metric("Vector Search", "top_base_score", base_results[0].score if base_results else 0.0)

            # Step 3: Apply graph-enhanced ranking if available
            with monitor.track_sub_step("Result Enhancement", "MaintIEStructuredRAG", base_results):
                if self.graph_operations_enabled and self.graph_ranker:
                    enhanced_results = self.graph_ranker.enhance_ranking(base_results, enhanced_query)
                    monitor.add_custom_metric("Result Enhancement", "enhancement_method", "graph_ranker")
                else:
                    # Fallback to knowledge graph ranking
                    enhanced_results = self._apply_knowledge_graph_ranking(base_results, enhanced_query)
                    monitor.add_custom_metric("Result Enhancement", "enhancement_method", "knowledge_graph")

                monitor.add_custom_metric("Result Enhancement", "enhanced_results_count", len(enhanced_results))

            logger.info(f"Optimized retrieval: 1 API call, {len(enhanced_results)} results")
            return enhanced_results[:max_results]

        except Exception as e:
            logger.error(f"Error in optimized retrieval: {e}")
            # Fallback to simple search
            with monitor.track_sub_step("Fallback Search", "MaintIEStructuredRAG", enhanced_query.analysis.original_query):
                fallback_results = self.vector_search.search(enhanced_query.analysis.original_query, top_k=max_results)
                monitor.add_custom_metric("Fallback Search", "fallback_results_count", len(fallback_results))
                return fallback_results

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

        REPLACES TODO: Now uses actual NetworkX graph operations from data_transformer.py
        """
        if not concepts:
            return []

        # Use graph operations if available
        if self.graph_operations_enabled and self.data_transformer:
            return self._graph_based_concept_selection(concepts, entities)
        else:
            # Fallback to term matching
            return self._term_based_concept_selection(concepts, entities)

    def _graph_based_concept_selection(self, concepts: List[str], entities: List[str]) -> List[str]:
        """Select concepts using NetworkX graph operations"""
        relevant_concepts = []
        concept_scores = {}

        # For each concept, calculate relevance using graph operations
        for concept in concepts:
            concept_score = 0.0

            # Find concept entity in knowledge graph
            concept_entity = self.data_transformer.get_entity_by_text(concept)
            if concept_entity:
                concept_id = concept_entity.entity_id if hasattr(concept_entity, 'entity_id') else concept

                # Calculate relevance based on graph distance to query entities
                for entity in entities:
                    query_entity = self.data_transformer.get_entity_by_text(entity)
                    if query_entity:
                        query_id = query_entity.entity_id if hasattr(query_entity, 'entity_id') else entity

                        # Use existing get_related_entities to check connectivity
                        related_entities = self.data_transformer.get_related_entities(query_id, max_distance=2)

                        if concept_id in related_entities:
                            # Score based on graph distance (closer = higher score)
                            try:
                                import networkx as nx
                                if self.data_transformer.knowledge_graph:
                                    distance = nx.shortest_path_length(
                                        self.data_transformer.knowledge_graph,
                                        query_id,
                                        concept_id
                                    )
                                    # Inverse distance scoring: distance 1 = 1.0, distance 2 = 0.5
                                    concept_score += 1.0 / distance
                            except (nx.NetworkXNoPath, nx.NetworkXError):
                                # If no path, give small score for entity match
                                concept_score += 0.1

                    # Also check entity type compatibility
                    if (concept_entity and query_entity and
                        hasattr(concept_entity, 'entity_type') and
                        hasattr(query_entity, 'entity_type') and
                        concept_entity.entity_type == query_entity.entity_type):
                        concept_score += 0.3

            if concept_score > 0:
                concept_scores[concept] = concept_score

        # Sort by relevance score and return top concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_concepts = [concept for concept, score in sorted_concepts]

        # Ensure we always return some concepts (fallback)
        if len(relevant_concepts) < 3 and len(concepts) >= 3:
            for concept in concepts:
                if concept not in relevant_concepts:
                    relevant_concepts.append(concept)
                if len(relevant_concepts) >= 5:
                    break

        logger.debug(f"Graph-based concept selection: {len(relevant_concepts)} concepts selected")
        return relevant_concepts[:5]

    def _term_based_concept_selection(self, concepts: List[str], entities: List[str]) -> List[str]:
        """Fallback: Simple term matching (original logic)"""
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

        REPLACES TODO: Now uses actual NetworkX graph operations from data_transformer.py
        """

        # Use graph operations if available
        if self.graph_operations_enabled and self.data_transformer:
            return self._graph_based_relevance_scoring(result, query_entities, query_concepts)
        else:
            # Fallback to term matching
            return self._term_based_relevance_scoring(result, query_entities, query_concepts)

    def _graph_based_relevance_scoring(self, result: SearchResult,
                                      query_entities: set, query_concepts: set) -> float:
        """Calculate relevance using NetworkX graph operations"""

        # Extract document terms for entity matching
        doc_text = f"{result.title} {result.content}".lower()
        doc_terms = set(doc_text.split())

        graph_score = 0.0
        entity_graph_score = 0.0
        concept_graph_score = 0.0

        # 1. Entity graph relevance
        entity_matches = 0
        for query_entity in query_entities:
            # Find entity in knowledge graph
            entity_obj = self.data_transformer.get_entity_by_text(query_entity)
            if entity_obj:
                entity_id = entity_obj.entity_id if hasattr(entity_obj, 'entity_id') else query_entity

                # Get related entities from graph (2-hop neighborhood)
                related_entities = self.data_transformer.get_related_entities(entity_id, max_distance=2)

                # Check if document contains related entities
                for related_id in related_entities:
                    if (self.data_transformer.knowledge_graph and
                        related_id in self.data_transformer.knowledge_graph):

                        related_text = self.data_transformer.knowledge_graph.nodes[related_id].get('text', '').lower()
                        if related_text and any(word in doc_terms for word in related_text.split()):
                            # Score based on graph distance
                            try:
                                import networkx as nx
                                distance = nx.shortest_path_length(
                                    self.data_transformer.knowledge_graph,
                                    entity_id,
                                    related_id
                                )
                                # Higher score for closer entities: distance 1 = 1.0, distance 2 = 0.5
                                entity_matches += 1.0 / distance
                            except (nx.NetworkXNoPath, nx.NetworkXError):
                                entity_matches += 0.1  # Small score for unconnected matches

            # Fallback: Direct entity mention in document
            if query_entity.lower() in doc_text:
                entity_matches += 1.0

        # Normalize entity score
        entity_graph_score = min(entity_matches / max(len(query_entities), 1), 1.0)

        # 2. Concept graph relevance (similar logic for concepts)
        concept_matches = 0
        for query_concept in query_concepts:
            concept_obj = self.data_transformer.get_entity_by_text(query_concept)
            if concept_obj:
                concept_id = concept_obj.entity_id if hasattr(concept_obj, 'entity_id') else query_concept

                # Get related concepts from graph
                related_concepts = self.data_transformer.get_related_entities(concept_id, max_distance=2)

                # Check if document contains related concepts
                for related_id in related_concepts:
                    if (self.data_transformer.knowledge_graph and
                        related_id in self.data_transformer.knowledge_graph):

                        related_text = self.data_transformer.knowledge_graph.nodes[related_id].get('text', '').lower()
                        if related_text and any(word in doc_terms for word in related_text.split()):
                            try:
                                import networkx as nx
                                distance = nx.shortest_path_length(
                                    self.data_transformer.knowledge_graph,
                                    concept_id,
                                    related_id
                                )
                                concept_matches += 1.0 / distance
                            except (nx.NetworkXNoPath, nx.NetworkXError):
                                concept_matches += 0.1

            # Fallback: Direct concept mention in document
            if query_concept.lower() in doc_text:
                concept_matches += 1.0

        # Normalize concept score
        concept_graph_score = min(concept_matches / max(len(query_concepts), 1), 1.0)

        # 3. Combined graph relevance score
        # Weight entity relevance higher than concept relevance for maintenance domain
        graph_score = 0.7 * entity_graph_score + 0.3 * concept_graph_score

        # 4. Add bonus for document metadata if available
        if hasattr(result, 'metadata') and result.metadata:
            # Check if document has maintenance-relevant metadata
            metadata_bonus = 0.0
            if 'equipment_type' in result.metadata or 'maintenance_type' in result.metadata:
                metadata_bonus = 0.1
            graph_score = min(graph_score + metadata_bonus, 1.0)

        logger.debug(f"Graph relevance for doc {result.doc_id}: entity_score={entity_graph_score:.3f}, "
                    f"concept_score={concept_graph_score:.3f}, final_score={graph_score:.3f}")

        return graph_score

    def _term_based_relevance_scoring(self, result: SearchResult,
                                     query_entities: set, query_concepts: set) -> float:
        """Fallback: Simple term matching (original logic)"""

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
            "api_calls_per_query": self.api_calls_per_query,
            "graph_operations_enabled": self.graph_operations_enabled,
            "data_transformer_available": self.data_transformer is not None,
            "caching_enabled": self.caching_enabled
        })
        if self.caching_enabled and self.response_cache:
            status["cache_stats"] = self.response_cache.get_cache_stats()
        return status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        metrics = super().get_performance_metrics()
        metrics.update({
            "retrieval_method": self.retrieval_method,
            "api_calls_per_query": self.api_calls_per_query
        })
        return metrics