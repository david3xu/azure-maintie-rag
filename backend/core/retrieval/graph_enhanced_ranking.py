"""
Graph-based ranking enhancement
Improves document relevance ranking based on knowledge graph context
"""

import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from core.models.maintenance_models import SearchResult, EnhancedQuery
from core.knowledge.data_transformer import MaintIEDataTransformer
from core.knowledge.entity_document_index import EntityDocumentIndex

logger = logging.getLogger(__name__)

class GraphEnhancedRanker:
    """Enhance document ranking using graph operations"""

    def __init__(self, data_transformer: MaintIEDataTransformer,
                 entity_index: EntityDocumentIndex):
        """Initialize with graph and entity index"""
        self.data_transformer = data_transformer
        self.entity_index = entity_index
        self.graph = data_transformer.knowledge_graph if data_transformer else None

        # Ranking parameters (configurable)
        self.entity_weight = 0.4
        self.concept_weight = 0.3
        self.graph_distance_weight = 0.3

        logger.info("GraphEnhancedRanker initialized")

    def enhance_ranking(self, search_results: List[SearchResult],
                       enhanced_query: EnhancedQuery) -> List[SearchResult]:
        """Enhance search results ranking using graph operations"""

        if not self.graph or not self.entity_index.index_built:
            logger.warning("Graph or entity index not available, returning original ranking")
            return search_results

        logger.info(f"Enhancing ranking for {len(search_results)} results")

        # Calculate graph-based scores for each result
        enhanced_results = []
        for result in search_results:
            graph_score = self._calculate_graph_score(result, enhanced_query)

            # Combine original score with graph score
            combined_score = self._combine_scores(result.score, graph_score)

            # Create enhanced result
            enhanced_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,
                content=result.content,
                score=combined_score,
                source=result.source,
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "graph_score": graph_score,
                    "ranking_method": "graph_enhanced"
                },
                entities=result.entities
            )

            enhanced_results.append(enhanced_result)

        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Graph-enhanced ranking completed")
        return enhanced_results

    def _calculate_graph_score(self, result: SearchResult,
                              enhanced_query: EnhancedQuery) -> float:
        """Calculate graph-based relevance score"""

        # Get entities in this document
        doc_entities = self.entity_index.get_entities_for_document(result.doc_id)
        if not doc_entities:
            return 0.0

        # Calculate entity-based score
        entity_score = self._calculate_entity_score(
            doc_entities, enhanced_query.analysis.entities
        )

        # Calculate concept-based score
        concept_score = self._calculate_concept_score(
            doc_entities, enhanced_query.expanded_concepts
        )

        # Calculate graph distance score
        distance_score = self._calculate_distance_score(
            doc_entities, enhanced_query.analysis.entities
        )

        # Combine scores
        graph_score = (
            self.entity_weight * entity_score +
            self.concept_weight * concept_score +
            self.graph_distance_weight * distance_score
        )

        return min(graph_score, 1.0)  # Normalize to [0,1]

    def _calculate_entity_score(self, doc_entities: List[str],
                               query_entities: List[str]) -> float:
        """Calculate entity overlap score"""
        if not doc_entities or not query_entities:
            return 0.0

        doc_entity_set = set(e.lower() for e in doc_entities)
        query_entity_set = set(e.lower() for e in query_entities)

        # Calculate Jaccard similarity
        intersection = len(doc_entity_set.intersection(query_entity_set))
        union = len(doc_entity_set.union(query_entity_set))

        return intersection / union if union > 0 else 0.0

    def _calculate_concept_score(self, doc_entities: List[str],
                                expanded_concepts: List[str]) -> float:
        """Calculate expanded concept score"""
        if not doc_entities or not expanded_concepts:
            return 0.0

        doc_entity_set = set(e.lower() for e in doc_entities)
        concept_set = set(c.lower() for c in expanded_concepts)

        # Count concept matches
        matches = len(doc_entity_set.intersection(concept_set))

        return matches / len(concept_set) if concept_set else 0.0

    def _calculate_distance_score(self, doc_entities: List[str],
                                 query_entities: List[str]) -> float:
        """Calculate graph distance-based score"""
        if not self.graph or not doc_entities or not query_entities:
            return 0.0

        distance_scores = []

        for doc_entity in doc_entities:
            for query_entity in query_entities:
                # Find entities in graph
                doc_entity_obj = self.data_transformer.get_entity_by_text(doc_entity)
                query_entity_obj = self.data_transformer.get_entity_by_text(query_entity)

                if doc_entity_obj and query_entity_obj:
                    doc_id = doc_entity_obj.entity_id if hasattr(doc_entity_obj, 'entity_id') else doc_entity
                    query_id = query_entity_obj.entity_id if hasattr(query_entity_obj, 'entity_id') else query_entity

                    try:
                        # Calculate shortest path distance
                        distance = nx.shortest_path_length(self.graph, query_id, doc_id)
                        # Convert distance to score (closer = higher score)
                        score = 1.0 / (distance + 1)  # +1 to avoid division by zero
                        distance_scores.append(score)

                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        # No path exists
                        continue

        return max(distance_scores) if distance_scores else 0.0

    def _combine_scores(self, original_score: float, graph_score: float) -> float:
        """Combine original vector score with graph score"""
        # Weighted combination: 70% original, 30% graph
        combined = 0.7 * original_score + 0.3 * graph_score
        return min(combined, 1.0)  # Ensure [0,1] range