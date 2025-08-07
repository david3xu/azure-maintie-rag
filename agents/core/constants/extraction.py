"""
Knowledge Extraction Constants
==============================

This module contains specialized constants for knowledge extraction algorithms and
processes. Many of these have medium-high automation potential and could be learned
from extraction performance analysis.

Key Areas:
1. Extraction Algorithm Constants - pattern matching and confidence calculation
2. Knowledge Extraction Agent Constants - processing parameters and thresholds
3. Extraction Processing - text analysis and statistical calculations

AUTO-GENERATION POTENTIAL: MEDIUM-HIGH
Many of these could be learned from extraction performance analysis and
domain-specific extraction patterns.
"""

# Base constants available if needed in future


class ExtractionAlgorithmConstants:
    """Specialized constants for extraction algorithms"""

    # AUTO-GENERATION POTENTIAL: MEDIUM-HIGH
    # Many of these could be learned from extraction performance analysis

    # Confidence Enhancement Factors - LEARNABLE from performance analysis
    CONFIDENCE_BOOST_FACTOR = 1.3  # LEARNABLE: optimal confidence adjustment


class KnowledgeExtractionConstants:
    """Knowledge Extraction Agent specific constants"""

    # Many of these could be consolidated and auto-generated

    MAX_ENTITIES_PER_DOCUMENT = 100  # LEARNABLE: document density analysis
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # LEARNABLE: domain confidence patterns
    DEFAULT_MAX_ENTITIES = 50  # LEARNABLE: optimal entity extraction count
    DEFAULT_MAX_RELATIONS = 30  # LEARNABLE: optimal relationship count
    LLM_EXTRACTION_CONFIDENCE = 0.8  # ADAPTIVE: LLM performance calibration
    MIN_RELATION_TEXT_LENGTH = 2  # Could be LEARNABLE: domain text patterns

    # Default values for immediate use (backward compatibility)
    DEFAULT_ENTITY_CONFIDENCE_THRESHOLD = 0.8  # Domain entity patterns
    DEFAULT_RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.7  # Domain relationship patterns
    DEFAULT_CHUNK_SIZE = 1000  # Optimal for document structure
    DEFAULT_CHUNK_OVERLAP = 200  # Calculated from chunk_size and content flow

    # Fallback Values - These should come from DomainAdaptiveConstants
    FALLBACK_ENTITY_CONFIDENCE_THRESHOLD = 0.85  # Reference to DomainAdaptiveConstants
    FALLBACK_RELATIONSHIP_CONFIDENCE_THRESHOLD = (
        0.75  # Reference to DomainAdaptiveConstants
    )
    FALLBACK_CHUNK_SIZE = 1000  # Reference to DomainAdaptiveConstants
    FALLBACK_CHUNK_OVERLAP = 200  # Reference to DomainAdaptiveConstants
    FALLBACK_BATCH_SIZE = 10  # Reference to PerformanceAdaptiveConstants
    FALLBACK_MAX_ENTITIES_PER_CHUNK = 20  # Reference to DomainAdaptiveConstants
    FALLBACK_MIN_RELATIONSHIP_STRENGTH = 0.5  # Reference to DomainAdaptiveConstants
    FALLBACK_QUALITY_VALIDATION_THRESHOLD = 0.8  # Reference to quality requirements

    # Extraction processing constants (from hardcoded values)
    STRATEGY_PATTERN_REDUCTION_PERCENT = 40  # 40% reduction through strategy pattern
    DEFAULT_STATISTICS_CONFIDENCE = 0.0  # Default confidence for statistics
    FREQUENCY_BOOST_MULTIPLIER = 10.0  # Used in frequency_boost calculation
    FREQUENCY_BOOST_BASE = 1.0  # Base value for frequency boost
    FREQUENCY_BOOST_MAX = 2.0  # Maximum frequency boost value
    MODEL_CONFIDENCE_WEIGHT = 0.25  # Weight for model confidence
    CONTEXT_CLARITY_WEIGHT = 0.15  # Weight for context clarity
    ENTITY_PRECISION_MULTIPLIER = 0.8  # Entity precision factor
    PATTERN_MATCH_WEIGHT = 0.1  # Pattern matching weight
    DOMAIN_RELEVANCE_WEIGHT = 0.1  # Domain relevance weight
    VALIDATION_WEIGHT = 0.1  # Validation score weight

    # Text clarity analysis constants
    CLARITY_WORD_DIVISOR = 20.0  # Divisor for word count in clarity calculation
    CLARITY_PUNCTUATION_MULTIPLIER = 2.0  # Multiplier for punctuation ratio
    MAX_PUNCTUATION_RATIO = 0.5  # Maximum punctuation ratio consideration
    MIN_CLARITY_SCORE = 0.1  # Minimum clarity score
    PRECISION_PENALTY = 0.2  # Penalty for precision issues

    # Type consistency scores
    PERSON_TYPE_CONFIDENCE_HIGH = 0.9  # High confidence for person entities
    PERSON_TYPE_CONFIDENCE_LOW = 0.6  # Low confidence for person entities
    ORG_TYPE_CONFIDENCE_HIGH = 0.9  # High confidence for organization entities
    ORG_TYPE_CONFIDENCE_LOW = 0.7  # Low confidence for organization entities
    LOCATION_TYPE_CONFIDENCE_HIGH = 0.9  # High confidence for location entities
    LOCATION_TYPE_CONFIDENCE_LOW = 0.6  # Low confidence for location entities
    TECHNICAL_TYPE_CONFIDENCE_HIGH = 0.9  # High confidence for technical entities
    TECHNICAL_TYPE_CONFIDENCE_LOW = 0.5  # Low confidence for technical entities

    # Relationship extraction constants
    DEFAULT_DOMAIN_PLAUSIBILITY = 0.7  # Default domain plausibility score
    MIN_RELATIONSHIP_CONFIDENCE = 0.5  # Minimum relationship confidence
    RELATIONSHIP_PROXIMITY_BASE = 0.1  # Base proximity score
    MIN_COHERENCE_SCORE = 0.3  # Minimum coherence score
    DEFAULT_COHERENCE_SCORE = 0.7  # Default coherence score
    DEFAULT_CHUNK_OVERLAP = 100  # Default chunk overlap
    DEFAULT_DOMAIN_RELEVANCE = 0.8  # Default domain relevance score
    MS_MULTIPLIER = 1000.0  # Milliseconds multiplier


class ExtractionQualityConstants:
    """Quality assessment constants for extraction operations"""

    # Text quality scoring
    DEFAULT_TEXT_QUALITY_SCORE = 0.8
    DEFAULT_REASONING_QUALITY = 0.8

    # Readability calculation constants (Flesch Reading Ease)
    MAX_READABILITY_SCORE = 100.0
    FLESCH_BASE_SCORE = 206.835
    FLESCH_SENTENCE_WEIGHT = 1.015
    FLESCH_SYLLABLE_WEIGHT = 84.6
    FLESCH_CHAR_DIVISOR = 4.7

    # Chunk size constraints
    MIN_CHUNK_SIZE = 50
    MAX_CHUNK_SIZE = 10000


# Export all constants
__all__ = [
    "ExtractionAlgorithmConstants",
    "KnowledgeExtractionConstants",
    "ExtractionQualityConstants",
]
