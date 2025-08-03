"""
Entity Processor - Specialized Entity Extraction Logic

This module provides specialized entity extraction processing that focuses
exclusively on entity identification, classification, and confidence scoring.

Key Features:
- Multi-strategy entity extraction (regex, NLP, ML-based)
- Domain-aware entity type classification
- Confidence scoring and validation
- Performance optimization for large documents
- Integration with Azure Cognitive Services

Architecture Integration:
- Used by Knowledge Extraction Agent for entity extraction delegation
- Integrates with extraction configuration parameters
- Provides structured entity results for knowledge graph construction
- Supports confidence thresholds and quality filtering
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


class EntityExtractionResult(BaseModel):
    """Results of entity extraction processing"""

    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    extraction_method: str = Field(..., description="Primary extraction method used")
    confidence_distribution: Dict[str, int] = Field(
        ..., description="Confidence score distribution"
    )
    entity_type_counts: Dict[str, int] = Field(..., description="Count by entity type")
    processing_time: float = Field(..., description="Processing time in seconds")
    total_entities: int = Field(..., description="Total entities found")
    high_confidence_entities: int = Field(..., description="High confidence entities")

    # Quality metrics
    average_confidence: float = Field(..., description="Average confidence score")
    coverage_percentage: float = Field(..., description="Document coverage percentage")
    validation_passed: bool = Field(..., description="Whether validation passed")
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )


class EntityProcessor:
    """
    Specialized processor for entity extraction with multiple strategies
    and domain-aware optimization.
    """

    def __init__(self):
        self._entity_patterns_cache: Dict[str, List[re.Pattern]] = {}
        self._performance_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "average_processing_time": 0.0,
            "method_performance": {
                "pattern_based": {"count": 0, "avg_time": 0.0, "avg_entities": 0.0},
                "nlp_based": {"count": 0, "avg_time": 0.0, "avg_entities": 0.0},
                "hybrid": {"count": 0, "avg_time": 0.0, "avg_entities": 0.0},
            },
        }

    async def extract_entities(
        self,
        content: str,
        config: ExtractionConfiguration,
        extraction_method: str = "hybrid",
    ) -> EntityExtractionResult:
        """
        Extract entities from content using specified method and configuration.

        Args:
            content: Text content to process
            config: Extraction configuration with parameters
            extraction_method: Method to use ("pattern_based", "nlp_based", "hybrid")

        Returns:
            EntityExtractionResult: Structured extraction results
        """
        start_time = time.time()

        try:
            # Choose extraction strategy
            if extraction_method == "pattern_based":
                entities = await self._extract_entities_pattern_based(content, config)
            elif extraction_method == "nlp_based":
                entities = await self._extract_entities_nlp_based(content, config)
            else:  # hybrid
                entities = await self._extract_entities_hybrid(content, config)

            # Filter by confidence threshold
            filtered_entities = [
                e
                for e in entities
                if e.confidence >= config.entity_confidence_threshold
            ]

            # Validate results
            validation_result = self._validate_extraction_results(
                filtered_entities, config
            )

            processing_time = time.time() - start_time

            # Create result
            result = EntityExtractionResult(
                entities=[self._entity_to_dict(e) for e in filtered_entities],
                extraction_method=extraction_method,
                confidence_distribution=self._calculate_confidence_distribution(
                    filtered_entities
                ),
                entity_type_counts=self._calculate_type_counts(filtered_entities),
                processing_time=processing_time,
                total_entities=len(filtered_entities),
                high_confidence_entities=len(
                    [e for e in filtered_entities if e.confidence > 0.8]
                ),
                average_confidence=sum(e.confidence for e in filtered_entities)
                / len(filtered_entities)
                if filtered_entities
                else 0.0,
                coverage_percentage=self._calculate_coverage_percentage(
                    filtered_entities, content
                ),
                validation_passed=validation_result["passed"],
                validation_warnings=validation_result["warnings"],
            )

            # Update performance statistics
            self._update_performance_stats(
                extraction_method, processing_time, len(filtered_entities), True
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Entity extraction failed: {e}")
            self._update_performance_stats(extraction_method, processing_time, 0, False)

            # Return empty result
            return EntityExtractionResult(
                entities=[],
                extraction_method=extraction_method,
                confidence_distribution={},
                entity_type_counts={},
                processing_time=processing_time,
                total_entities=0,
                high_confidence_entities=0,
                average_confidence=0.0,
                coverage_percentage=0.0,
                validation_passed=False,
                validation_warnings=[f"Extraction failed: {str(e)}"],
            )

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
                    confidence = self._calculate_pattern_confidence(
                        match, content, entity_type
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
        """Extract entities using NLP-based approach (placeholder for actual NLP integration)"""
        entities = []

        # This would integrate with Azure Cognitive Services or other NLP services
        # For now, implementing a simplified approach based on linguistic patterns

        # Extract capitalized phrases (potential proper nouns)
        capitalized_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

        for match in capitalized_pattern.finditer(content):
            text = match.group()

            # Classify entity type based on context and vocabulary
            entity_type = self._classify_entity_type(text, config)
            confidence = self._calculate_nlp_confidence(text, entity_type, content)

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
                confidence = 0.8  # High confidence for known technical terms

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
        enhanced_entities = self._enhance_multi_method_confidence(deduplicated_entities)

        return enhanced_entities

    def _get_entity_patterns(
        self, entity_types: List[str], technical_vocabulary: List[str]
    ) -> Dict[str, List[re.Pattern]]:
        """Get compiled regex patterns for entity types"""

        # Cache key for patterns
        cache_key = f"{hash(tuple(entity_types))}_{hash(tuple(technical_vocabulary))}"

        if cache_key in self._entity_patterns_cache:
            return self._entity_patterns_cache[cache_key]

        patterns = {}

        # Common pattern templates for different entity types
        pattern_templates = {
            "identifier": [
                r"\b[A-Z][A-Z0-9_]{2,}\b",  # ALL_CAPS identifiers
                r"\b[a-z]+_[a-z0-9_]+\b",  # snake_case identifiers
                r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b",  # camelCase identifiers
            ],
            "concept": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Title Case Concepts
                r"\b(?:process|method|system|approach|strategy)\b",  # Process-related terms
            ],
            "technical_term": [
                rf"\b{re.escape(term)}\b"
                for term in technical_vocabulary[:50]  # Limit for performance
            ],
            "api_interface": [
                r"\b[A-Z][a-zA-Z]*(?:API|Interface|Service|Client)\b",
                r"\b[a-z]+\.[a-z]+\(\)",  # Method calls
                r"\b[A-Z][a-zA-Z]*\.[A-Z][a-zA-Z]*\b",  # Class.Method patterns
            ],
            "system_component": [
                r"\b[A-Z][a-zA-Z]*(?:Manager|Handler|Controller|Processor)\b",
                r"\b(?:Azure|AWS|GCP)\s+[A-Z][a-zA-Z\s]+\b",  # Cloud service names
            ],
            "code_element": [
                r"\b(?:function|class|method|variable|parameter)\s+[a-zA-Z_][a-zA-Z0-9_]*\b",
                r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)",  # Function calls
                r"\$\{[^}]+\}",  # Template variables
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

    def _calculate_pattern_confidence(
        self, match: re.Match, content: str, entity_type: str
    ) -> float:
        """Calculate confidence score for pattern-based matches"""

        text = match.group()

        # Base confidence factors
        length_factor = min(1.0, len(text) / 20)  # Longer matches get higher confidence
        position_factor = (
            0.8 if match.start() < len(content) / 4 else 0.6
        )  # Early matches get boost

        # Context analysis
        context = self._extract_context(content, match.start(), match.end(), window=50)
        context_factor = 0.7

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
                    context_factor = 0.9
                    break

        # Case sensitivity bonus
        case_factor = 0.8
        if text.isupper() or text.istitle():
            case_factor = 0.9

        # Calculate final confidence
        confidence = (
            length_factor * 0.3
            + position_factor * 0.2
            + context_factor * 0.3
            + case_factor * 0.2
        )

        return min(1.0, confidence)

    def _calculate_nlp_confidence(
        self, text: str, entity_type: str, content: str
    ) -> float:
        """Calculate confidence score for NLP-based matches"""

        # This would be enhanced with actual NLP confidence scores
        # For now, using heuristic-based confidence calculation

        base_confidence = 0.6

        # Length factor
        if len(text) > 3:
            base_confidence += 0.1
        if len(text) > 10:
            base_confidence += 0.1

        # Capitalization pattern
        if text.istitle() or text.isupper():
            base_confidence += 0.1

        # Frequency in document (rare terms get higher confidence)
        frequency = content.lower().count(text.lower())
        if frequency == 1:
            base_confidence += 0.1
        elif frequency <= 3:
            base_confidence += 0.05

        return min(1.0, base_confidence)

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
        if text.isupper() and len(text) > 2:
            return "identifier"
        elif text.istitle() and " " in text:
            return "concept"
        elif any(char in text for char in "._()"):
            return "code_element"
        else:
            return "concept"  # Default classification

    def _extract_context(
        self, content: str, start: int, end: int, window: int = 30
    ) -> str:
        """Extract surrounding context for an entity match"""

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

    def _enhance_multi_method_confidence(
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
                best_entity.confidence = min(1.0, best_entity.confidence * 1.2)
                best_entity.extraction_method = "hybrid_multi_method"
                best_entity.metadata["multi_method_count"] = len(entity_list)
                best_entity.metadata["methods_used"] = [
                    e.extraction_method for e in entity_list
                ]
                enhanced_entities.append(best_entity)
            else:
                enhanced_entities.append(entity_list[0])

        return enhanced_entities

    def _validate_extraction_results(
        self, entities: List[EntityMatch], config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Validate extraction results against configuration criteria"""

        warnings = []
        passed = True

        # Check minimum entity count
        min_entities = config.validation_criteria.get("min_entities_per_document", 0)
        if len(entities) < min_entities:
            warnings.append(
                f"Only {len(entities)} entities found, minimum {min_entities} expected"
            )
            passed = False

        # Check confidence distribution
        if entities:
            avg_confidence = sum(e.confidence for e in entities) / len(entities)
            if avg_confidence < config.entity_confidence_threshold:
                warnings.append(
                    f"Average confidence {avg_confidence:.2f} below threshold {config.entity_confidence_threshold}"
                )

        # Check entity type coverage
        expected_types = set(config.expected_entity_types)
        found_types = set(e.entity_type for e in entities)
        missing_types = expected_types - found_types

        if len(missing_types) > len(expected_types) / 2:
            warnings.append(
                f"Many expected entity types not found: {list(missing_types)[:5]}"
            )

        return {"passed": passed, "warnings": warnings}

    def _calculate_confidence_distribution(
        self, entities: List[EntityMatch]
    ) -> Dict[str, int]:
        """Calculate distribution of confidence scores"""

        distribution = {
            "very_high": 0,  # 0.9+
            "high": 0,  # 0.8-0.9
            "medium": 0,  # 0.6-0.8
            "low": 0,  # <0.6
        }

        for entity in entities:
            if entity.confidence >= 0.9:
                distribution["very_high"] += 1
            elif entity.confidence >= 0.8:
                distribution["high"] += 1
            elif entity.confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _calculate_type_counts(self, entities: List[EntityMatch]) -> Dict[str, int]:
        """Calculate count by entity type"""

        type_counts = {}
        for entity in entities:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        return type_counts

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

    def _update_performance_stats(
        self, method: str, processing_time: float, entity_count: int, success: bool
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
__all__ = ["EntityProcessor", "EntityMatch", "EntityExtractionResult"]
