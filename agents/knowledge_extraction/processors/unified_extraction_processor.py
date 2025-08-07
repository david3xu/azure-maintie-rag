"""
Unified Extraction Processor - Consolidated Entity and Relationship Extraction

This module combines the functionality of entity and relationship processors
into a single, streamlined extraction pipeline that eliminates redundancy
while preserving all functionality.

Key Features:
- Unified entity and relationship extraction in single pass
- Consolidated confidence calculation logic
- Integrated validation and quality assessment
- Performance optimization through reduced redundancy
- Centralized configuration management

Architecture Integration:
- Replaces separate EntityProcessor and RelationshipProcessor
- Maintains backward compatibility with existing interfaces
- Integrates with centralized configuration system
- Used by Knowledge Extraction Agent for complete extraction workflow
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

# Interface contracts - use fallback to avoid circular imports during consolidation
try:
    from services.interfaces.extraction_interface import ExtractionConfiguration
except ImportError:
    # Fallback definition to avoid circular imports during restructuring
    from pydantic import BaseModel
    from typing import List, Dict, Any

    class ExtractionConfiguration(BaseModel):
        """Fallback extraction configuration model"""

        domain_name: str = "general"
        entity_confidence_threshold: float = 0.7
        relationship_confidence_threshold: float = 0.65
        expected_entity_types: List[str] = []
        technical_vocabulary: List[str] = []
        key_concepts: List[str] = []
        minimum_quality_score: float = 0.6
        enable_caching: bool = True
        validation_criteria: Dict[str, Any] = {}


# Clean configuration imports (CODING_STANDARDS compliant)
from config.centralized_config import get_extraction_config

# Import validation processor
from .validation_processor import SimpleValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Individual entity match result"""

    text: str
    entity_type: str
    start_position: int
    end_position: int
    confidence: float
    extraction_method: str
    context: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RelationshipMatch:
    """Individual relationship match result"""

    source_entity: str
    relation_type: str
    target_entity: str
    confidence: float
    extraction_method: str
    start_position: int
    end_position: int
    context: str = ""
    metadata: Dict[str, Any] = None
    relation_direction: str = "bidirectional"

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_triple(self) -> Tuple[str, str, str]:
        """Convert to (source, relation, target) triple"""
        return (self.source_entity, self.relation_type, self.target_entity)


class UnifiedExtractionResult(BaseModel):
    """Results of unified extraction processing"""

    # Entity results
    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    entity_confidence_distribution: Dict[str, int] = Field(
        ..., description="Entity confidence distribution"
    )
    entity_type_counts: Dict[str, int] = Field(..., description="Count by entity type")
    total_entities: int = Field(..., description="Total entities found")
    high_confidence_entities: int = Field(..., description="High confidence entities")
    average_entity_confidence: float = Field(
        ..., description="Average entity confidence"
    )

    # Relationship results
    relationships: List[Dict[str, Any]] = Field(
        ..., description="Extracted relationships"
    )
    relationship_confidence_distribution: Dict[str, int] = Field(
        ..., description="Relationship confidence distribution"
    )
    relation_type_counts: Dict[str, int] = Field(
        ..., description="Count by relation type"
    )
    total_relationships: int = Field(..., description="Total relationships found")
    high_confidence_relationships: int = Field(
        ..., description="High confidence relationships"
    )
    average_relationship_confidence: float = Field(
        ..., description="Average relationship confidence"
    )

    # Unified results
    extraction_method: str = Field(..., description="Primary extraction method used")
    processing_time: float = Field(..., description="Processing time in seconds")

    # Graph metrics
    unique_entity_pairs: int = Field(..., description="Unique entity pairs connected")
    graph_density: float = Field(..., description="Relationship graph density")
    connected_components: int = Field(..., description="Number of connected components")

    # Quality metrics
    validation_passed: bool = Field(..., description="Whether validation passed")
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    coverage_percentage: float = Field(..., description="Document coverage percentage")


class UnifiedExtractionProcessor:
    """
    Unified processor combining entity and relationship extraction.
    Eliminates overlap while preserving all functionality.
    """

    def __init__(self):
        # Load configurations
        # Use clean configuration (CODING_STANDARDS: Essential parameters only)
        self.extraction_config = None  # Will be loaded lazily when needed

        # Backward compatibility aliases (CODING_STANDARDS: Gradual migration)
        self.entity_config = self.extraction_config
        self.relationship_config = self.extraction_config

    def _get_config(self, domain_name: str = "general"):
        """Get extraction configuration lazily to avoid circular imports"""
        if self.extraction_config is None:
            try:
                self.extraction_config = get_extraction_config(domain_name)
            except Exception:
                # Use safe defaults during initialization
                from types import SimpleNamespace

                self.extraction_config = SimpleNamespace(
                    entity_confidence_threshold=0.7,
                    relationship_confidence_threshold=0.65,
                    chunk_size=1000,
                    max_entities_per_chunk=15,
                    minimum_quality_score=0.8,
                )
        return self.extraction_config

        # Initialize after _get_config method definition
        self._init_components()

    def _init_components(self):
        """Initialize components after _get_config method is defined"""
        # Pattern caches
        self._entity_patterns_cache: Dict[str, List[re.Pattern]] = {}
        self._relationship_patterns_cache: Dict[str, List[re.Pattern]] = {}

        # Performance statistics
        self._performance_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "average_processing_time": 0.0,
            "method_performance": {
                "pattern_based": {
                    "count": 0,
                    "avg_time": 0.0,
                    "avg_entities": 0.0,
                    "avg_relationships": 0.0,
                },
                "nlp_based": {
                    "count": 0,
                    "avg_time": 0.0,
                    "avg_entities": 0.0,
                    "avg_relationships": 0.0,
                },
                "hybrid": {
                    "count": 0,
                    "avg_time": 0.0,
                    "avg_entities": 0.0,
                    "avg_relationships": 0.0,
                },
            },
        }

        # Initialize validator
        self.validator = SimpleValidator()

    async def extract_knowledge_complete(
        self,
        content: str,
        config: ExtractionConfiguration,
        extraction_method: str = "hybrid",
    ) -> UnifiedExtractionResult:
        """
        Single method for complete knowledge extraction combining entities and relationships.

        Args:
            content: Text content to process
            config: Extraction configuration with parameters
            extraction_method: Method to use ("pattern_based", "nlp_based", "hybrid")

        Returns:
            UnifiedExtractionResult: Complete extraction results
        """
        start_time = time.time()

        try:
            # Phase 1: Entity extraction (consolidated from EntityProcessor)
            entities = await self._extract_entities_unified(
                content, config, extraction_method
            )

            # Phase 2: Relationship extraction (consolidated from RelationshipProcessor)
            relationships = await self._extract_relationships_unified(
                content, entities, config, extraction_method
            )

            # Phase 3: Cross-validation and enhancement
            validated_result = await self._validate_and_enhance(
                entities, relationships, config, content
            )

            processing_time = time.time() - start_time

            # Create unified result
            result = self._create_unified_result(
                entities,
                relationships,
                validated_result,
                extraction_method,
                processing_time,
                content,
            )

            # Update performance statistics
            self._update_performance_stats(
                extraction_method,
                processing_time,
                len(entities),
                len(relationships),
                True,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Unified extraction failed: {e}")
            self._update_performance_stats(
                extraction_method, processing_time, 0, 0, False
            )

            # Return empty result
            return self._create_empty_result(extraction_method, processing_time, str(e))

    async def _extract_entities_unified(
        self, content: str, config: ExtractionConfiguration, extraction_method: str
    ) -> List[EntityMatch]:
        """Unified entity extraction combining all strategies"""

        if extraction_method == "pattern_based":
            return await self._extract_entities_pattern_based(content, config)
        elif extraction_method == "nlp_based":
            return await self._extract_entities_nlp_based(content, config)
        else:  # hybrid
            return await self._extract_entities_hybrid(content, config)

    async def _extract_entities_pattern_based(
        self, content: str, config: ExtractionConfiguration
    ) -> List[EntityMatch]:
        """Extract entities using pattern-based approach"""
        entities = []

        # Get or create patterns for expected entity types
        patterns = self._get_entity_patterns(
            config.expected_entity_types, config.technical_vocabulary
        )

        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = pattern.finditer(content)

                for match in matches:
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_entity_confidence(
                        match, content, entity_type, "pattern_based"
                    )

                    if confidence >= config.entity_confidence_threshold:
                        entities.append(
                            EntityMatch(
                                text=match.group(),
                                entity_type=entity_type,
                                start_position=match.start(),
                                end_position=match.end(),
                                confidence=confidence,
                                extraction_method="pattern_based",
                                context=self._extract_context(
                                    content, match.start(), match.end()
                                ),
                                metadata={
                                    "pattern": pattern.pattern,
                                    "match_groups": match.groups(),
                                },
                            )
                        )

        return entities

    async def _extract_entities_nlp_based(
        self, content: str, config: ExtractionConfiguration
    ) -> List[EntityMatch]:
        """Extract entities using NLP-based approach"""
        entities = []

        # Extract capitalized phrases (potential proper nouns)
        capitalized_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

        for match in capitalized_pattern.finditer(content):
            text = match.group()

            # Classify entity type based on context and vocabulary
            entity_type = self._classify_entity_type(text, config)
            confidence = self._calculate_entity_confidence(
                match, content, entity_type, "nlp_based", text
            )

            if confidence >= config.entity_confidence_threshold:
                entities.append(
                    EntityMatch(
                        text=text,
                        entity_type=entity_type,
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=confidence,
                        extraction_method="nlp_based",
                        context=self._extract_context(
                            content, match.start(), match.end()
                        ),
                        metadata={"classification_method": "linguistic_pattern"},
                    )
                )

        # Extract technical terms from vocabulary
        for term in config.technical_vocabulary:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)

            for match in pattern.finditer(content):
                confidence = self.extraction_config.entity_confidence_threshold

                entities.append(
                    EntityMatch(
                        text=match.group(),
                        entity_type="technical_term",
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=confidence,
                        extraction_method="nlp_based",
                        context=self._extract_context(
                            content, match.start(), match.end()
                        ),
                        metadata={"source": "technical_vocabulary"},
                    )
                )

        return entities

    async def _extract_entities_hybrid(
        self, content: str, config: ExtractionConfiguration
    ) -> List[EntityMatch]:
        """Extract entities using hybrid approach combining multiple methods"""

        # Run both pattern-based and NLP-based extraction
        pattern_entities = await self._extract_entities_pattern_based(content, config)
        nlp_entities = await self._extract_entities_nlp_based(content, config)

        # Combine and deduplicate results
        all_entities = pattern_entities + nlp_entities
        deduplicated_entities = self._deduplicate_entities(all_entities)

        # Boost confidence for entities found by multiple methods
        enhanced_entities = self._enhance_multi_method_confidence(
            deduplicated_entities, "entity"
        )

        return enhanced_entities

    async def _extract_relationships_unified(
        self,
        content: str,
        entities: List[EntityMatch],
        config: ExtractionConfiguration,
        extraction_method: str,
    ) -> List[RelationshipMatch]:
        """Unified relationship extraction using entities"""

        if extraction_method == "pattern_based":
            return await self._extract_relationships_pattern_based(
                content, entities, config
            )
        elif extraction_method == "nlp_based":
            return await self._extract_relationships_semantic(content, entities, config)
        else:  # hybrid
            return await self._extract_relationships_hybrid(content, entities, config)

    async def _extract_relationships_pattern_based(
        self,
        content: str,
        entities: List[EntityMatch],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using pattern-based approach"""
        relationships = []

        # Create entity lookup for fast matching
        entity_texts = [e.text for e in entities]

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
        for entity1_match in entities:
            for entity2_match in entities:
                if entity1_match.text == entity2_match.text:
                    continue

                entity1 = entity1_match.text
                entity2 = entity2_match.text

                for pattern_template in syntactic_patterns:
                    # Create specific pattern for this entity pair
                    pattern = pattern_template.format(
                        entity1=re.escape(entity1), entity2=re.escape(entity2)
                    )

                    matches = re.finditer(pattern, content, re.IGNORECASE)

                    for match in matches:
                        relation_type = self._determine_relation_type(
                            match, pattern_template
                        )
                        confidence = self._calculate_relationship_confidence(
                            match, content, entity1, entity2, "pattern_based"
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
                                    context=self._extract_context(
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
        entities: List[EntityMatch],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using semantic analysis"""
        relationships = []

        entity_texts = [e.text for e in entities]

        # Semantic relationship indicators based on domain
        semantic_indicators = {
            "implements": ["implements", "extends", "inherits"],
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
        for entity1_match in entities:
            for entity2_match in entities:
                if entity1_match.text == entity2_match.text:
                    continue

                entity1 = entity1_match.text
                entity2 = entity2_match.text

                # Find sentences containing both entities
                sentences = self._find_sentences_with_entities(
                    content, entity1, entity2
                )

                for sentence in sentences:
                    for relation_type, indicators in semantic_indicators.items():
                        for indicator in indicators:
                            if indicator in sentence.lower():
                                confidence = self._calculate_relationship_confidence(
                                    None,
                                    sentence,
                                    entity1,
                                    entity2,
                                    "semantic",
                                    indicator,
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

    async def _extract_relationships_hybrid(
        self,
        content: str,
        entities: List[EntityMatch],
        config: ExtractionConfiguration,
    ) -> List[RelationshipMatch]:
        """Extract relationships using hybrid approach combining multiple methods"""

        # Run all extraction methods
        pattern_relationships = await self._extract_relationships_pattern_based(
            content, entities, config
        )
        semantic_relationships = await self._extract_relationships_semantic(
            content, entities, config
        )

        # Combine all relationships
        all_relationships = pattern_relationships + semantic_relationships

        # Deduplicate and enhance confidence
        deduplicated_relationships = self._deduplicate_relationships(all_relationships)
        enhanced_relationships = self._enhance_multi_method_confidence(
            deduplicated_relationships, "relationship"
        )

        return enhanced_relationships

    def _get_entity_patterns(
        self, entity_types: List[str], technical_vocabulary: List[str]
    ) -> Dict[str, List[re.Pattern]]:
        """Get compiled regex patterns for entity types (consolidated from EntityProcessor)"""

        # Cache key for patterns
        cache_key = f"{hash(tuple(entity_types))}_{hash(tuple(technical_vocabulary))}"

        if cache_key in self._entity_patterns_cache:
            return self._entity_patterns_cache[cache_key]

        patterns = {}

        # Common pattern templates for different entity types
        pattern_templates = {
            "identifier": [
                r"\b[A-Z][A-Z0-9_]{3,}\b",  # Simple caps pattern (CODING_STANDARDS: No over-engineering)
                r"\b[a-z]+_[a-z0-9_]+\b",
                r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b",
            ],
            "concept": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
                r"\b(?:process|method|system|approach|strategy)\b",
            ],
            "technical_term": [
                rf"\b{re.escape(term)}\b"
                for term in technical_vocabulary[
                    :20
                ]  # Simple limit (CODING_STANDARDS: No over-engineering)
            ],
            "api_interface": [
                r"\b[A-Z][a-zA-Z]*(?:API|Interface|Service|Client)\b",
                r"\b[a-z]+\.[a-z]+\(\)",
                r"\b[A-Z][a-zA-Z]*\.[A-Z][a-zA-Z]*\b",
            ],
            "system_component": [
                r"\b[A-Z][a-zA-Z]*(?:Manager|Handler|Controller|Processor)\b",
                r"\b(?:Azure|AWS|GCP)\s+[A-Z][a-zA-Z\s]+\b",
            ],
        }

        # Compile patterns for each entity type
        for entity_type in entity_types:
            if entity_type in pattern_templates:
                patterns[entity_type] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in pattern_templates[entity_type]
                ]
            else:
                # Generic pattern for unknown types
                patterns[entity_type] = [
                    re.compile(rf"\b{re.escape(entity_type)}\b", re.IGNORECASE)
                ]

        # Cache the compiled patterns
        self._entity_patterns_cache[cache_key] = patterns

        return patterns

    def _calculate_entity_confidence(
        self,
        match: re.Match,
        content: str,
        entity_type: str,
        method: str,
        text: str = None,
    ) -> float:
        """Unified entity confidence calculation"""

        if text is None:
            text = match.group()

        if method == "pattern_based":
            return self._calculate_pattern_entity_confidence(
                match, content, entity_type
            )
        else:  # nlp_based
            return self._calculate_nlp_entity_confidence(text, entity_type, content)

    def _calculate_pattern_entity_confidence(
        self, match: re.Match, content: str, entity_type: str
    ) -> float:
        """Calculate confidence score for pattern-based entity matches"""

        text = match.group()

        # Simple confidence calculation - no over-engineering
        length_factor = min(1.0, len(text) / 20.0)  # Normalize text length
        position_factor = (
            0.8 if match.start() < len(content) / 4 else 0.6
        )  # Early vs late position

        # Context analysis
        context = self._extract_context(
            content,
            match.start(),
            match.end(),
            window=50,  # Simple context window (CODING_STANDARDS: No over-engineering)
        )
        context_factor = 1.1  # Simple context boost (CODING_STANDARDS: Data-driven)

        # Check for surrounding context that supports the entity type
        context_indicators = {
            "identifier": ["variable", "parameter", "constant", "define"],
            "concept": ["concept", "idea", "approach", "method"],
            "technical_term": ["technology", "tool", "framework", "library"],
            "api_interface": ["interface", "API", "endpoint", "service"],
            "system_component": ["component", "module", "system", "service"],
        }

        if entity_type in context_indicators:
            for indicator in context_indicators[entity_type]:
                if indicator.lower() in context.lower():
                    context_factor = self.entity_config.context_factor_high
                    break

        # Case sensitivity bonus
        case_factor = self.entity_config.case_factor_default
        if text.isupper() or text.istitle():
            case_factor = self.entity_config.case_factor_high

        # Calculate final confidence
        confidence = (
            length_factor * self.entity_config.length_weight
            + position_factor * self.entity_config.position_weight
            + context_factor * self.entity_config.context_weight
            + case_factor * self.entity_config.case_weight
        )

        return min(self.entity_config.max_confidence_value, confidence)

    def _calculate_nlp_entity_confidence(
        self, text: str, entity_type: str, content: str
    ) -> float:
        """Calculate confidence score for NLP-based entity matches"""

        base_confidence = self.entity_config.base_nlp_confidence

        # Length factor
        if len(text) > self.entity_config.min_entity_length:
            base_confidence += self.entity_config.length_bonus_small
        if len(text) > self.entity_config.long_entity_threshold:
            base_confidence += self.entity_config.length_bonus_large

        # Capitalization pattern
        if text.istitle() or text.isupper():
            base_confidence += self.entity_config.frequency_bonus

        # Frequency in document (rare terms get higher confidence)
        frequency = content.lower().count(text.lower())
        if frequency == self.entity_config.single_frequency:
            base_confidence += self.entity_config.frequency_bonus
        elif frequency <= self.entity_config.low_frequency_threshold:
            base_confidence += self.entity_config.frequency_bonus_small

        return min(self.entity_config.max_confidence_value, base_confidence)

    def _calculate_relationship_confidence(
        self,
        match: Optional[re.Match],
        content: str,
        entity1: str,
        entity2: str,
        method: str,
        indicator: str = None,
    ) -> float:
        """Unified relationship confidence calculation"""

        if method == "pattern_based":
            return self._calculate_syntactic_confidence(
                match, content, entity1, entity2
            )
        else:  # semantic
            return self._calculate_semantic_confidence(
                content, entity1, entity2, indicator
            )

    def _calculate_syntactic_confidence(
        self, match: re.Match, content: str, entity1: str, entity2: str
    ) -> float:
        """Calculate confidence for syntactic relationship match"""

        # Base confidence factors
        match_text = match.group()
        base_confidence = self.relationship_config.base_syntactic_confidence

        # Distance factor (closer entities get higher confidence)
        entity1_pos = match_text.find(entity1)
        entity2_pos = match_text.find(entity2)
        distance = (
            abs(entity2_pos - entity1_pos)
            if entity1_pos != -1 and entity2_pos != -1
            else len(match_text)
        )
        distance_factor = max(
            self.relationship_config.min_distance_factor,
            self.relationship_config.max_distance_factor
            - (distance / self.relationship_config.distance_divisor),
        )

        # Context quality factor
        context = self._extract_context(
            content,
            match.start(),
            match.end(),
            window=self.relationship_config.context_window_size,
        )
        context_factor = self.relationship_config.context_factor_default

        # Check for relationship-supporting words in context
        support_words = [
            "relationship",
            "connection",
            "interaction",
            "dependency",
            "association",
        ]
        if any(word in context.lower() for word in support_words):
            context_factor = self.relationship_config.context_factor_high

        confidence = (
            base_confidence * self.relationship_config.syntactic_base_weight
            + distance_factor * self.relationship_config.syntactic_distance_weight
            + context_factor * self.relationship_config.syntactic_context_weight
        )

        return min(self.relationship_config.max_confidence_value, confidence)

    def _calculate_semantic_confidence(
        self, sentence: str, entity1: str, entity2: str, indicator: str
    ) -> float:
        """Calculate confidence for semantic relationship match"""

        base_confidence = self.relationship_config.base_semantic_confidence

        # Indicator strength
        strong_indicators = ["implements", "contains", "processes", "depends on"]
        if indicator in strong_indicators:
            base_confidence = self.relationship_config.high_semantic_confidence

        # Sentence length factor (shorter sentences are more reliable)
        length_factor = max(
            self.relationship_config.min_length_factor,
            self.relationship_config.max_distance_factor
            - (
                len(sentence.split())
                / self.relationship_config.max_sentence_length_divisor
            ),
        )

        # Entity prominence in sentence
        entity1_count = sentence.lower().count(entity1.lower())
        entity2_count = sentence.lower().count(entity2.lower())
        prominence_factor = min(
            self.relationship_config.max_distance_factor,
            (entity1_count + entity2_count)
            / self.relationship_config.max_prominence_divisor,
        )

        confidence = (
            base_confidence * self.relationship_config.semantic_base_weight
            + length_factor * self.relationship_config.semantic_length_weight
            + prominence_factor * self.relationship_config.semantic_prominence_weight
        )

        return min(self.relationship_config.max_confidence_value, confidence)

    def _classify_entity_type(self, text: str, config: ExtractionConfiguration) -> str:
        """Classify entity type based on text characteristics and configuration"""

        # Check if it matches any expected entity types
        text_lower = text.lower()

        for entity_type in config.expected_entity_types:
            if entity_type.lower() in text_lower or text_lower in entity_type.lower():
                return entity_type

        # Check technical vocabulary
        if text in config.technical_vocabulary:
            return "technical_term"

        # Heuristic classification
        if text.isupper() and len(text) > self.entity_config.caps_min_length:
            return "identifier"
        elif text.istitle() and " " in text:
            return "concept"
        elif any(char in text for char in "._()"):
            return "code_element"
        else:
            return "concept"  # Default classification

    def _determine_relation_type(self, match: re.Match, pattern_template: str) -> str:
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

    def _extract_context(
        self, content: str, start: int, end: int, window: int = None
    ) -> str:
        """Extract surrounding context for a match"""

        if window is None:
            window = self.entity_config.context_window_small

        context_start = max(0, start - window)
        context_end = min(len(content), end + window)

        return content[context_start:context_end].strip()

    def _deduplicate_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove duplicate entities keeping the highest confidence ones"""

        entity_map: Dict[Tuple[str, str], EntityMatch] = {}

        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)

            if key not in entity_map or entity.confidence > entity_map[key].confidence:
                entity_map[key] = entity

        return list(entity_map.values())

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

    def _enhance_multi_method_confidence(self, items: List, item_type: str) -> List:
        """Enhance confidence for items found by multiple methods"""

        if item_type == "entity":
            return self._enhance_entity_multi_method_confidence(items)
        else:
            return self._enhance_relationship_multi_method_confidence(items)

    def _enhance_entity_multi_method_confidence(
        self, entities: List[EntityMatch]
    ) -> List[EntityMatch]:
        """Enhance confidence for entities found by multiple methods"""

        text_method_map: Dict[str, List[EntityMatch]] = {}

        # Group by text
        for entity in entities:
            text_key = entity.text.lower()
            if text_key not in text_method_map:
                text_method_map[text_key] = []
            text_method_map[text_key].append(entity)

        enhanced_entities = []

        for text_key, entity_list in text_method_map.items():
            if len(entity_list) > 1:
                # Multiple methods found this entity - boost confidence
                best_entity = max(entity_list, key=lambda e: e.confidence)
                best_entity.confidence = min(
                    self.entity_config.max_confidence_value,
                    best_entity.confidence * self.entity_config.confidence_boost_factor,
                )
                best_entity.extraction_method = "hybrid_multi_method"
                best_entity.metadata["multi_method_count"] = len(entity_list)
                best_entity.metadata["methods_used"] = [
                    e.extraction_method for e in entity_list
                ]
                enhanced_entities.append(best_entity)
            else:
                enhanced_entities.append(entity_list[0])

        return enhanced_entities

    def _enhance_relationship_multi_method_confidence(
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

    async def _validate_and_enhance(
        self,
        entities: List[EntityMatch],
        relationships: List[RelationshipMatch],
        config: ExtractionConfiguration,
        content: str,
    ) -> ValidationResult:
        """Cross-validation and enhancement using SimpleValidator"""

        # Convert to dict format for validation
        entity_dicts = [self._entity_to_dict(e) for e in entities]
        relationship_dicts = [self._relationship_to_dict(r) for r in relationships]

        # Use the simple validator
        validation_result = self.validator.validate_extraction(
            entity_dicts,
            relationship_dicts,
            config.entity_confidence_threshold,
            config.relationship_confidence_threshold,
        )

        return validation_result

    def _create_unified_result(
        self,
        entities: List[EntityMatch],
        relationships: List[RelationshipMatch],
        validation_result: ValidationResult,
        extraction_method: str,
        processing_time: float,
        content: str,
    ) -> UnifiedExtractionResult:
        """Create unified extraction result"""

        # Calculate graph metrics
        graph_metrics = self._calculate_graph_metrics(relationships, entities)

        return UnifiedExtractionResult(
            # Entity results
            entities=[self._entity_to_dict(e) for e in entities],
            entity_confidence_distribution=self._calculate_entity_confidence_distribution(
                entities
            ),
            entity_type_counts=self._calculate_entity_type_counts(entities),
            total_entities=len(entities),
            high_confidence_entities=len(
                [
                    e
                    for e in entities
                    if e.confidence > self.entity_config.high_confidence_threshold
                ]
            ),
            average_entity_confidence=(
                sum(e.confidence for e in entities) / len(entities) if entities else 0.0
            ),
            # Relationship results
            relationships=[self._relationship_to_dict(r) for r in relationships],
            relationship_confidence_distribution=self._calculate_relationship_confidence_distribution(
                relationships
            ),
            relation_type_counts=self._calculate_relation_type_counts(relationships),
            total_relationships=len(relationships),
            high_confidence_relationships=len(
                [
                    r
                    for r in relationships
                    if r.confidence > self.relationship_config.high_confidence_threshold
                ]
            ),
            average_relationship_confidence=(
                sum(r.confidence for r in relationships) / len(relationships)
                if relationships
                else 0.0
            ),
            # Unified results
            extraction_method=extraction_method,
            processing_time=processing_time,
            # Graph metrics
            unique_entity_pairs=len(
                set((r.source_entity, r.target_entity) for r in relationships)
            ),
            graph_density=graph_metrics["density"],
            connected_components=graph_metrics["components"],
            # Quality metrics
            validation_passed=validation_result.is_valid,
            validation_warnings=validation_result.warnings,
            validation_errors=validation_result.errors,
            coverage_percentage=self._calculate_coverage_percentage(entities, content),
        )

    def _create_empty_result(
        self, extraction_method: str, processing_time: float, error_message: str
    ) -> UnifiedExtractionResult:
        """Create empty result for failed extraction"""

        return UnifiedExtractionResult(
            entities=[],
            entity_confidence_distribution={},
            entity_type_counts={},
            total_entities=0,
            high_confidence_entities=0,
            average_entity_confidence=0.0,
            relationships=[],
            relationship_confidence_distribution={},
            relation_type_counts={},
            total_relationships=0,
            high_confidence_relationships=0,
            average_relationship_confidence=0.0,
            extraction_method=extraction_method,
            processing_time=processing_time,
            unique_entity_pairs=0,
            graph_density=0.0,
            connected_components=0,
            validation_passed=False,
            validation_warnings=[],
            validation_errors=[f"Extraction failed: {error_message}"],
            coverage_percentage=0.0,
        )

    def _calculate_entity_confidence_distribution(
        self, entities: List[EntityMatch]
    ) -> Dict[str, int]:
        """Calculate distribution of entity confidence scores"""

        distribution = {"very_high": 0, "high": 0, "medium": 0, "low": 0}

        for entity in entities:
            if entity.confidence >= self.entity_config.confidence_very_high_threshold:
                distribution["very_high"] += 1
            elif entity.confidence >= self.entity_config.high_confidence_threshold:
                distribution["high"] += 1
            elif entity.confidence >= self.entity_config.base_nlp_confidence:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _calculate_relationship_confidence_distribution(
        self, relationships: List[RelationshipMatch]
    ) -> Dict[str, int]:
        """Calculate distribution of relationship confidence scores"""

        distribution = {"very_high": 0, "high": 0, "medium": 0, "low": 0}

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

    def _calculate_entity_type_counts(
        self, entities: List[EntityMatch]
    ) -> Dict[str, int]:
        """Calculate count by entity type"""

        type_counts = {}
        for entity in entities:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        return type_counts

    def _calculate_relation_type_counts(
        self, relationships: List[RelationshipMatch]
    ) -> Dict[str, int]:
        """Calculate count by relation type"""

        type_counts = {}
        for relationship in relationships:
            type_counts[relationship.relation_type] = (
                type_counts.get(relationship.relation_type, 0) + 1
            )

        return type_counts

    def _calculate_graph_metrics(
        self, relationships: List[RelationshipMatch], entities: List[EntityMatch]
    ) -> Dict[str, Any]:
        """Calculate graph-based metrics for relationships"""

        if not relationships or not entities:
            return {"density": 0.0, "components": 0}

        # Build adjacency list
        adjacency = {}
        all_entities = set(e.text for e in entities)

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

    def _calculate_coverage_percentage(
        self, entities: List[EntityMatch], content: str
    ) -> float:
        """Calculate what percentage of document is covered by entities"""

        if not entities or not content:
            return 0.0

        total_entity_chars = sum(len(e.text) for e in entities)
        coverage = (total_entity_chars / len(content)) * 100

        return min(100.0, coverage)

    def _entity_to_dict(self, entity: EntityMatch) -> Dict[str, Any]:
        """Convert EntityMatch to dictionary format"""

        return {
            "name": entity.text,
            "type": entity.entity_type,
            "confidence": entity.confidence,
            "start_position": entity.start_position,
            "end_position": entity.end_position,
            "extraction_method": entity.extraction_method,
            "context": entity.context,
            "metadata": entity.metadata,
        }

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
        entity_count: int,
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

            # Update average entity count
            current_avg_entities = method_stats["avg_entities"]
            method_stats["avg_entities"] = (
                current_avg_entities * (count - 1) + entity_count
            ) / count

            # Update average relationship count
            current_avg_relationships = method_stats.get("avg_relationships", 0.0)
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
__all__ = [
    "UnifiedExtractionProcessor",
    "UnifiedExtractionResult",
    "EntityMatch",
    "RelationshipMatch",
]
