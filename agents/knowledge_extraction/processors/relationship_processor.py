"""
Relationship Processor - Specialized Relationship Extraction Logic

This module provides specialized relationship extraction processing that focuses
exclusively on identifying relationships between entities with high accuracy.

Key Features:
- Multi-strategy relationship extraction (syntactic, semantic, pattern-based)
- Domain-aware relationship type classification
- Confidence scoring and relationship validation
- Graph-ready relationship formatting
- Integration with entity extraction results

Architecture Integration:
- Used by Knowledge Extraction Agent for relationship extraction delegation
- Integrates with entity extraction results for relationship discovery
- Provides structured relationship results for knowledge graph construction
- Supports relationship confidence thresholds and quality filtering
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

# Interface contracts
from config.extraction_interface import ExtractionConfiguration

logger = logging.getLogger(__name__)


@dataclass
class RelationshipMatch:
    """Individual relationship match result"""

    source_entity: str
    relation_type: str
    target_entity: str
    confidence: float
    extraction_method: str

    # Position information
    start_position: int
    end_position: int
    context: str = ""

    # Additional metadata
    metadata: Dict[str, Any] = None
    relation_direction: str = (
        "bidirectional"  # "source_to_target", "target_to_source", "bidirectional"
    )

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_triple(self) -> Tuple[str, str, str]:
        """Convert to (source, relation, target) triple"""
        return (self.source_entity, self.relation_type, self.target_entity)


class RelationshipExtractionResult(BaseModel):
    """Results of relationship extraction processing"""

    relationships: List[Dict[str, Any]] = Field(
        ..., description="Extracted relationships"
    )
    extraction_method: str = Field(..., description="Primary extraction method used")
    confidence_distribution: Dict[str, int] = Field(
        ..., description="Confidence score distribution"
    )
    relation_type_counts: Dict[str, int] = Field(
        ..., description="Count by relation type"
    )
    processing_time: float = Field(..., description="Processing time in seconds")

    # Metrics
    total_relationships: int = Field(..., description="Total relationships found")
    high_confidence_relationships: int = Field(
        ..., description="High confidence relationships"
    )
    unique_entity_pairs: int = Field(..., description="Unique entity pairs connected")
    average_confidence: float = Field(..., description="Average confidence score")

    # Graph metrics
    graph_density: float = Field(..., description="Relationship graph density")
    connected_components: int = Field(..., description="Number of connected components")

    # Quality metrics
    validation_passed: bool = Field(..., description="Whether validation passed")
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )


class RelationshipProcessor:
    """
    Specialized processor for relationship extraction with multiple strategies
    and graph-aware optimization.
    """

    def __init__(self):
        self._relation_patterns_cache: Dict[str, List[re.Pattern]] = {}
        self._performance_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "average_processing_time": 0.0,
            "method_performance": {
                "syntactic": {"count": 0, "avg_time": 0.0, "avg_relationships": 0.0},
                "semantic": {"count": 0, "avg_time": 0.0, "avg_relationships": 0.0},
                "pattern_based": {
                    "count": 0,
                    "avg_time": 0.0,
                    "avg_relationships": 0.0,
                },
                "hybrid": {"count": 0, "avg_time": 0.0, "avg_relationships": 0.0},
            },
        }

    async def extract_relationships(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration,
        extraction_method: str = "hybrid",
    ) -> RelationshipExtractionResult:
        """
        Extract relationships from content using entities and configuration.

        Args:
            content: Text content to process
            entities: Previously extracted entities
            config: Extraction configuration with parameters
            extraction_method: Method to use ("syntactic", "semantic", "pattern_based", "hybrid")

        Returns:
            RelationshipExtractionResult: Structured relationship extraction results
        """
        start_time = time.time()

        try:
            # Choose extraction strategy
            if extraction_method == "syntactic":
                relationships = await self._extract_relationships_syntactic(
                    content, entities, config
                )
            elif extraction_method == "semantic":
                relationships = await self._extract_relationships_semantic(
                    content, entities, config
                )
            elif extraction_method == "pattern_based":
                relationships = await self._extract_relationships_pattern_based(
                    content, entities, config
                )
            else:  # hybrid
                relationships = await self._extract_relationships_hybrid(
                    content, entities, config
                )

            # Filter by confidence threshold
            filtered_relationships = [
                r
                for r in relationships
                if r.confidence >= config.relationship_confidence_threshold
            ]

            # Validate results
            validation_result = self._validate_relationship_results(
                filtered_relationships, config
            )

            # Calculate graph metrics
            graph_metrics = self._calculate_graph_metrics(
                filtered_relationships, entities
            )

            processing_time = time.time() - start_time

            # Create result
            result = RelationshipExtractionResult(
                relationships=[
                    self._relationship_to_dict(r) for r in filtered_relationships
                ],
                extraction_method=extraction_method,
                confidence_distribution=self._calculate_confidence_distribution(
                    filtered_relationships
                ),
                relation_type_counts=self._calculate_type_counts(
                    filtered_relationships
                ),
                processing_time=processing_time,
                total_relationships=len(filtered_relationships),
                high_confidence_relationships=len(
                    [r for r in filtered_relationships if r.confidence > 0.8]
                ),
                unique_entity_pairs=len(
                    set(
                        (r.source_entity, r.target_entity)
                        for r in filtered_relationships
                    )
                ),
                average_confidence=sum(r.confidence for r in filtered_relationships)
                / len(filtered_relationships)
                if filtered_relationships
                else 0.0,
                graph_density=graph_metrics["density"],
                connected_components=graph_metrics["components"],
                validation_passed=validation_result["passed"],
                validation_warnings=validation_result["warnings"],
            )

            # Update performance statistics
            self._update_performance_stats(
                extraction_method, processing_time, len(filtered_relationships), True
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Relationship extraction failed: {e}")
            self._update_performance_stats(extraction_method, processing_time, 0, False)

            # Return empty result
            return RelationshipExtractionResult(
                relationships=[],
                extraction_method=extraction_method,
                confidence_distribution={},
                relation_type_counts={},
                processing_time=processing_time,
                total_relationships=0,
                high_confidence_relationships=0,
                unique_entity_pairs=0,
                average_confidence=0.0,
                graph_density=0.0,
                connected_components=0,
                validation_passed=False,
                validation_warnings=[f"Extraction failed: {str(e)}"],
            )

    async def _extract_relationships_syntactic(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using syntactic patterns and dependency parsing"""
        relationships = []

        # Create entity lookup for fast matching
        entity_texts = [e["name"] for e in entities]
        entity_positions = self._find_entity_positions(content, entity_texts)

        # Common syntactic patterns for relationships
        syntactic_patterns = [
            # Entity1 VERB Entity2
            r"({entity1})\s+(\w+)\s+({entity2})",
            # Entity1 is/has/contains Entity2
            r"({entity1})\s+(is|has|contains|includes|uses|implements)\s+({entity2})",
            # Entity1 and Entity2 VERB
            r"({entity1})\s+and\s+({entity2})\s+(\w+)",
            # Entity1 of Entity2
            r"({entity1})\s+of\s+({entity2})",
            # Entity1 in Entity2
            r"({entity1})\s+in\s+({entity2})",
            # Entity1 with Entity2
            r"({entity1})\s+with\s+({entity2})",
            # Entity1 to Entity2
            r"({entity1})\s+to\s+({entity2})",
            # Entity1 from Entity2
            r"({entity1})\s+from\s+({entity2})",
        ]

        # Extract relationships using syntactic patterns
        for entity1 in entity_texts:
            for entity2 in entity_texts:
                if entity1 == entity2:
                    continue

                for pattern_template in syntactic_patterns:
                    # Create specific pattern for this entity pair
                    pattern = pattern_template.format(
                        entity1=re.escape(entity1), entity2=re.escape(entity2)
                    )

                    matches = re.finditer(pattern, content, re.IGNORECASE)

                    for match in matches:
                        relation_type = self._determine_syntactic_relation_type(
                            match, pattern_template
                        )
                        confidence = self._calculate_syntactic_confidence(
                            match, content, entity1, entity2
                        )

                        if confidence >= config.relationship_confidence_threshold:
                            relationships.append(
                                RelationshipMatch(
                                    source_entity=entity1,
                                    relation_type=relation_type,
                                    target_entity=entity2,
                                    confidence=confidence,
                                    extraction_method="syntactic",
                                    start_position=match.start(),
                                    end_position=match.end(),
                                    context=self._extract_relationship_context(
                                        content, match.start(), match.end()
                                    ),
                                    metadata={
                                        "pattern": pattern,
                                        "match_text": match.group(),
                                    },
                                )
                            )

        return relationships

    async def _extract_relationships_semantic(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using semantic analysis"""
        relationships = []

        # This would integrate with semantic analysis services
        # For now, implementing a simplified approach based on domain knowledge

        entity_texts = [e["name"] for e in entities]

        # Semantic relationship indicators based on domain
        semantic_indicators = {
            "implements": ["implements", "implements", "extends", "inherits"],
            "uses": ["uses", "utilizes", "employs", "applies", "leverages"],
            "contains": ["contains", "includes", "has", "comprises"],
            "processes": ["processes", "handles", "manages", "operates"],
            "connects_to": ["connects to", "links to", "integrates with"],
            "depends_on": ["depends on", "relies on", "requires", "needs"],
            "configures": ["configures", "sets up", "initializes", "defines"],
            "monitors": ["monitors", "tracks", "observes", "watches"],
            "triggers": ["triggers", "initiates", "starts", "launches"],
            "validates": ["validates", "verifies", "checks", "confirms"],
        }

        # Extract relationships based on semantic indicators
        for entity1 in entity_texts:
            for entity2 in entity_texts:
                if entity1 == entity2:
                    continue

                # Find sentences containing both entities
                sentences = self._find_sentences_with_entities(
                    content, entity1, entity2
                )

                for sentence in sentences:
                    for relation_type, indicators in semantic_indicators.items():
                        for indicator in indicators:
                            if indicator in sentence.lower():
                                confidence = self._calculate_semantic_confidence(
                                    sentence, entity1, entity2, indicator
                                )

                                if (
                                    confidence
                                    >= config.relationship_confidence_threshold
                                ):
                                    # Find position in original content
                                    position = content.lower().find(sentence.lower())

                                    relationships.append(
                                        RelationshipMatch(
                                            source_entity=entity1,
                                            relation_type=relation_type,
                                            target_entity=entity2,
                                            confidence=confidence,
                                            extraction_method="semantic",
                                            start_position=position,
                                            end_position=position + len(sentence),
                                            context=sentence,
                                            metadata={
                                                "indicator": indicator,
                                                "sentence": sentence,
                                            },
                                        )
                                    )
                                break  # Only use first matching indicator per sentence

        return relationships

    async def _extract_relationships_pattern_based(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using predefined relationship patterns"""
        relationships = []

        # Get relationship patterns from configuration
        relationship_patterns = config.relationship_patterns
        entity_texts = [e["name"] for e in entities]

        # Process each relationship pattern
        for pattern_str in relationship_patterns:
            # Parse pattern: "entity1 relation entity2"
            pattern_parts = pattern_str.split()
            if len(pattern_parts) >= 3:
                relation_type = pattern_parts[1]

                # Create regex pattern
                pattern = self._create_relationship_regex(pattern_str)

                # Find matches in content
                matches = pattern.finditer(content)

                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        entity1, entity2 = groups[0], groups[-1]

                        # Verify entities are in our entity list
                        if any(
                            e1.lower() in entity1.lower() for e1 in entity_texts
                        ) and any(e2.lower() in entity2.lower() for e2 in entity_texts):
                            confidence = self._calculate_pattern_confidence(
                                match, content, pattern_str
                            )

                            if confidence >= config.relationship_confidence_threshold:
                                relationships.append(
                                    RelationshipMatch(
                                        source_entity=entity1,
                                        relation_type=relation_type,
                                        target_entity=entity2,
                                        confidence=confidence,
                                        extraction_method="pattern_based",
                                        start_position=match.start(),
                                        end_position=match.end(),
                                        context=self._extract_relationship_context(
                                            content, match.start(), match.end()
                                        ),
                                        metadata={
                                            "pattern": pattern_str,
                                            "match_text": match.group(),
                                        },
                                    )
                                )

        return relationships

    async def _extract_relationships_hybrid(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using hybrid approach combining multiple methods"""

        # Run all extraction methods
        syntactic_relationships = await self._extract_relationships_syntactic(
            content, entities, config
        )
        semantic_relationships = await self._extract_relationships_semantic(
            content, entities, config
        )
        pattern_relationships = await self._extract_relationships_pattern_based(
            content, entities, config
        )

        # Combine all relationships
        all_relationships = (
            syntactic_relationships + semantic_relationships + pattern_relationships
        )

        # Deduplicate and enhance confidence
        deduplicated_relationships = self._deduplicate_relationships(all_relationships)
        enhanced_relationships = self._enhance_multi_method_confidence(
            deduplicated_relationships
        )

        return enhanced_relationships

    def _find_entity_positions(
        self, content: str, entity_texts: List[str]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Find all positions of entities in content"""
        positions = {}

        for entity in entity_texts:
            entity_positions = []
            start = 0

            while True:
                pos = content.lower().find(entity.lower(), start)
                if pos == -1:
                    break
                entity_positions.append((pos, pos + len(entity)))
                start = pos + 1

            positions[entity] = entity_positions

        return positions

    def _determine_syntactic_relation_type(
        self, match: re.Match, pattern_template: str
    ) -> str:
        """Determine relation type from syntactic pattern match"""

        # Map pattern templates to relation types
        pattern_relations = {
            r"({entity1})\s+(\w+)\s+({entity2})": "interacts_with",
            r"({entity1})\s+(is|has|contains|includes|uses|implements)\s+({entity2})": "has_relationship",
            r"({entity1})\s+and\s+({entity2})\s+(\w+)": "associated_with",
            r"({entity1})\s+of\s+({entity2})": "part_of",
            r"({entity1})\s+in\s+({entity2})": "contained_in",
            r"({entity1})\s+with\s+({entity2})": "associated_with",
            r"({entity1})\s+to\s+({entity2})": "connected_to",
            r"({entity1})\s+from\s+({entity2})": "derived_from",
        }

        for pattern, relation in pattern_relations.items():
            if pattern == pattern_template:
                return relation

        return "related_to"  # Default relation type

    def _calculate_syntactic_confidence(
        self, match: re.Match, content: str, entity1: str, entity2: str
    ) -> float:
        """Calculate confidence for syntactic relationship match"""

        # Base confidence factors
        match_text = match.group()
        base_confidence = 0.6

        # Distance factor (closer entities get higher confidence)
        entity1_pos = match_text.find(entity1)
        entity2_pos = match_text.find(entity2)
        distance = (
            abs(entity2_pos - entity1_pos)
            if entity1_pos != -1 and entity2_pos != -1
            else len(match_text)
        )
        distance_factor = max(0.3, 1.0 - (distance / 100))

        # Context quality factor
        context = self._extract_relationship_context(
            content, match.start(), match.end(), window=100
        )
        context_factor = 0.7

        # Check for relationship-supporting words in context
        support_words = [
            "relationship",
            "connection",
            "interaction",
            "dependency",
            "association",
        ]
        if any(word in context.lower() for word in support_words):
            context_factor = 0.9

        confidence = (
            base_confidence * 0.4 + distance_factor * 0.3 + context_factor * 0.3
        )

        return min(1.0, confidence)

    def _find_sentences_with_entities(
        self, content: str, entity1: str, entity2: str
    ) -> List[str]:
        """Find sentences containing both entities"""

        sentences = re.split(r"[.!?]+", content)
        matching_sentences = []

        for sentence in sentences:
            if (
                entity1.lower() in sentence.lower()
                and entity2.lower() in sentence.lower()
            ):
                matching_sentences.append(sentence.strip())

        return matching_sentences

    def _calculate_semantic_confidence(
        self, sentence: str, entity1: str, entity2: str, indicator: str
    ) -> float:
        """Calculate confidence for semantic relationship match"""

        base_confidence = 0.7

        # Indicator strength
        strong_indicators = ["implements", "contains", "processes", "depends on"]
        if indicator in strong_indicators:
            base_confidence = 0.8

        # Sentence length factor (shorter sentences are more reliable)
        length_factor = max(0.5, 1.0 - (len(sentence.split()) / 50))

        # Entity prominence in sentence
        entity1_count = sentence.lower().count(entity1.lower())
        entity2_count = sentence.lower().count(entity2.lower())
        prominence_factor = min(1.0, (entity1_count + entity2_count) / 4)

        confidence = (
            base_confidence * 0.5 + length_factor * 0.3 + prominence_factor * 0.2
        )

        return min(1.0, confidence)

    def _create_relationship_regex(self, pattern_str: str) -> re.Pattern:
        """Create regex pattern from relationship pattern string"""

        # Simple pattern parsing: "entity1 relation entity2" -> regex
        parts = pattern_str.split()

        if len(parts) >= 3:
            # Create flexible pattern
            pattern = (
                r"(\w+(?:\s+\w+)*)\s+" + re.escape(parts[1]) + r"\s+(\w+(?:\s+\w+)*)"
            )
        else:
            # Fallback pattern
            pattern = r"(\w+)\s+(\w+)\s+(\w+)"

        return re.compile(pattern, re.IGNORECASE)

    def _calculate_pattern_confidence(
        self, match: re.Match, content: str, pattern_str: str
    ) -> float:
        """Calculate confidence for pattern-based relationship match"""

        base_confidence = 0.8  # Pattern-based matches are generally reliable

        # Pattern specificity factor
        pattern_parts = pattern_str.split()
        specificity_factor = min(1.0, len(pattern_parts) / 5)

        # Context support
        context = self._extract_relationship_context(
            content, match.start(), match.end()
        )
        context_factor = 0.7

        # Check for domain-specific terms in context
        domain_terms = ["system", "process", "component", "service", "interface"]
        if any(term in context.lower() for term in domain_terms):
            context_factor = 0.9

        confidence = (
            base_confidence * 0.6 + specificity_factor * 0.2 + context_factor * 0.2
        )

        return min(1.0, confidence)

    def _extract_relationship_context(
        self, content: str, start: int, end: int, window: int = 50
    ) -> str:
        """Extract surrounding context for a relationship match"""

        context_start = max(0, start - window)
        context_end = min(len(content), end + window)

        return content[context_start:context_end].strip()

    def _deduplicate_relationships(
        self, relationships: List[RelationshipMatch]
    ) -> List[RelationshipMatch]:
        """Remove duplicate relationships keeping the highest confidence ones"""

        relationship_map: Dict[Tuple[str, str, str], RelationshipMatch] = {}

        for relationship in relationships:
            key = relationship.to_triple()

            if (
                key not in relationship_map
                or relationship.confidence > relationship_map[key].confidence
            ):
                relationship_map[key] = relationship

        return list(relationship_map.values())

    def _enhance_multi_method_confidence(
        self, relationships: List[RelationshipMatch]
    ) -> List[RelationshipMatch]:
        """Enhance confidence for relationships found by multiple methods"""

        triple_method_map: Dict[Tuple[str, str, str], List[RelationshipMatch]] = {}

        # Group by triple
        for relationship in relationships:
            triple = relationship.to_triple()
            if triple not in triple_method_map:
                triple_method_map[triple] = []
            triple_method_map[triple].append(relationship)

        enhanced_relationships = []

        for triple, relationship_list in triple_method_map.items():
            if len(relationship_list) > 1:
                # Multiple methods found this relationship - boost confidence
                best_relationship = max(relationship_list, key=lambda r: r.confidence)
                best_relationship.confidence = min(
                    1.0, best_relationship.confidence * 1.3
                )
                best_relationship.extraction_method = "hybrid_multi_method"
                best_relationship.metadata["multi_method_count"] = len(
                    relationship_list
                )
                best_relationship.metadata["methods_used"] = [
                    r.extraction_method for r in relationship_list
                ]
                enhanced_relationships.append(best_relationship)
            else:
                enhanced_relationships.append(relationship_list[0])

        return enhanced_relationships

    def _validate_relationship_results(
        self, relationships: List[RelationshipMatch], config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Validate relationship extraction results"""

        warnings = []
        passed = True

        # Check minimum relationship count
        min_relationships = config.validation_criteria.get(
            "min_relationships_per_document", 0
        )
        if len(relationships) < min_relationships:
            warnings.append(
                f"Only {len(relationships)} relationships found, minimum {min_relationships} expected"
            )
            passed = False

        # Check confidence distribution
        if relationships:
            avg_confidence = sum(r.confidence for r in relationships) / len(
                relationships
            )
            if avg_confidence < config.relationship_confidence_threshold:
                warnings.append(
                    f"Average confidence {avg_confidence:.2f} below threshold {config.relationship_confidence_threshold}"
                )

        # Check for relationship diversity
        unique_relations = set(r.relation_type for r in relationships)
        if len(unique_relations) < 2 and len(relationships) > 5:
            warnings.append("Low relationship type diversity detected")

        return {"passed": passed, "warnings": warnings}

    def _calculate_graph_metrics(
        self, relationships: List[RelationshipMatch], entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate graph-based metrics for relationships"""

        if not relationships or not entities:
            return {"density": 0.0, "components": 0}

        # Build adjacency list
        adjacency = {}
        all_entities = set(e["name"] for e in entities)

        for relationship in relationships:
            source = relationship.source_entity
            target = relationship.target_entity

            if source not in adjacency:
                adjacency[source] = set()
            if target not in adjacency:
                adjacency[target] = set()

            adjacency[source].add(target)
            adjacency[target].add(
                source
            )  # Treat as undirected for component calculation

        # Calculate density
        num_entities = len(all_entities)
        max_edges = num_entities * (num_entities - 1) / 2
        actual_edges = len(relationships)
        density = actual_edges / max_edges if max_edges > 0 else 0.0

        # Calculate connected components using DFS
        visited = set()
        components = 0

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in adjacency.get(node, []):
                dfs(neighbor)

        for entity in all_entities:
            if entity not in visited:
                dfs(entity)
                components += 1

        return {"density": density, "components": components}

    def _calculate_confidence_distribution(
        self, relationships: List[RelationshipMatch]
    ) -> Dict[str, int]:
        """Calculate distribution of confidence scores"""

        distribution = {
            "very_high": 0,  # 0.9+
            "high": 0,  # 0.8-0.9
            "medium": 0,  # 0.6-0.8
            "low": 0,  # <0.6
        }

        for relationship in relationships:
            if relationship.confidence >= 0.9:
                distribution["very_high"] += 1
            elif relationship.confidence >= 0.8:
                distribution["high"] += 1
            elif relationship.confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _calculate_type_counts(
        self, relationships: List[RelationshipMatch]
    ) -> Dict[str, int]:
        """Calculate count by relation type"""

        type_counts = {}
        for relationship in relationships:
            type_counts[relationship.relation_type] = (
                type_counts.get(relationship.relation_type, 0) + 1
            )

        return type_counts

    def _relationship_to_dict(self, relationship: RelationshipMatch) -> Dict[str, Any]:
        """Convert RelationshipMatch to dictionary format"""

        return {
            "source": relationship.source_entity,
            "relation": relationship.relation_type,
            "target": relationship.target_entity,
            "confidence": relationship.confidence,
            "start_position": relationship.start_position,
            "end_position": relationship.end_position,
            "extraction_method": relationship.extraction_method,
            "context": relationship.context,
            "direction": relationship.relation_direction,
            "metadata": relationship.metadata,
        }

    def _update_performance_stats(
        self,
        method: str,
        processing_time: float,
        relationship_count: int,
        success: bool,
    ):
        """Update performance statistics"""

        self._performance_stats["total_extractions"] += 1
        if success:
            self._performance_stats["successful_extractions"] += 1

        # Update method-specific stats
        if method in self._performance_stats["method_performance"]:
            method_stats = self._performance_stats["method_performance"][method]
            method_stats["count"] += 1

            # Update average processing time
            current_avg_time = method_stats["avg_time"]
            count = method_stats["count"]
            method_stats["avg_time"] = (
                current_avg_time * (count - 1) + processing_time
            ) / count

            # Update average relationship count
            current_avg_relationships = method_stats["avg_relationships"]
            method_stats["avg_relationships"] = (
                current_avg_relationships * (count - 1) + relationship_count
            ) / count

        # Update overall average processing time
        total_extractions = self._performance_stats["total_extractions"]
        current_avg = self._performance_stats["average_processing_time"]
        self._performance_stats["average_processing_time"] = (
            current_avg * (total_extractions - 1) + processing_time
        ) / total_extractions

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""

        return {
            **self._performance_stats,
            "success_rate": (
                self._performance_stats["successful_extractions"]
                / self._performance_stats["total_extractions"]
                if self._performance_stats["total_extractions"] > 0
                else 0.0
            ),
        }


# Export main components
__all__ = ["RelationshipProcessor", "RelationshipMatch", "RelationshipExtractionResult"]
