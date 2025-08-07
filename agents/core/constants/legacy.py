"""
Legacy Backward Compatibility Constants
=======================================

This module provides backward compatibility for existing constant access patterns.
These are aliases and consolidated classes that maintain the existing import structure
while the system transitions to the modular organization.

Key Compatibility Areas:
1. Backward Compatibility Aliases - maintain existing class names
2. Stub Constants - implementation transition helpers
3. Processing Constants - grouped operational constants
4. Cache Constants - cache-specific backward compatibility

This module ensures that all existing imports continue to work unchanged during
the modular constants refactoring transition.
"""

from .base import BaseScalingFactors, MathematicalConstants
from .domain import ContentAnalysisAdaptiveConstants
from .extraction import KnowledgeExtractionConstants
from .performance import PerformanceAdaptiveConstants
from .search import MLModelStaticConstants
from .system import InfrastructureConstants, SystemBoundaryConstants

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These provide compatibility during transition period


class CacheConstants:
    """Backward compatibility for cache constants"""

    DEFAULT_CACHE_TTL = PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL
    CACHE_CLEANUP_INTERVAL = PerformanceAdaptiveConstants.CACHE_CLEANUP_INTERVAL
    CACHE_OPTIMIZATION_MAX_SIZE = (
        PerformanceAdaptiveConstants.CACHE_OPTIMIZATION_MAX_SIZE
    )
    MS_PER_SECOND = MathematicalConstants.MS_PER_SECOND
    ZERO_FLOAT = MathematicalConstants.CONFIDENCE_MIN
    MAX_METRICS_HISTORY = SystemBoundaryConstants.MAX_METRICS_HISTORY
    BYTES_TO_GB_DIVISOR = MathematicalConstants.BYTES_PER_GB
    MEMORY_PRECISION_DECIMAL = 2  # Decimal precision for memory measurements
    TIME_PRECISION_DECIMAL = 1  # Decimal precision for time measurements
    PERCENTAGE_MULTIPLIER = MathematicalConstants.PERCENTAGE_MULTIPLIER
    CACHE_TTL_SECONDS = PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL
    ACCEPTABLE_PROCESSING_TIME = 3.0  # Acceptable processing time in seconds


class ProcessingConstants:
    """Backward compatibility for processing constants"""

    DEFAULT_TIMEOUT = PerformanceAdaptiveConstants.DEFAULT_TIMEOUT
    AZURE_SERVICE_TIMEOUT = PerformanceAdaptiveConstants.AZURE_SERVICE_TIMEOUT
    MAX_RETRIES = PerformanceAdaptiveConstants.MAX_RETRIES
    MAX_CONCURRENT_CHUNKS = PerformanceAdaptiveConstants.MAX_CONCURRENT_CHUNKS
    MAX_EXECUTION_TIME_LIMIT = SystemBoundaryConstants.MAX_EXECUTION_TIME_LIMIT
    MAX_EXECUTION_TIME_SECONDS = SystemBoundaryConstants.MAX_EXECUTION_TIME_LIMIT
    MAX_EXECUTION_TIME_MIN = SystemBoundaryConstants.MAX_EXECUTION_TIME_LIMIT / 60.0
    DEFAULT_MEMORY_LIMIT_MB = SystemBoundaryConstants.DEFAULT_MEMORY_LIMIT_MB
    MAX_AZURE_SERVICE_COST_USD = 100.0  # Maximum cost per operation
    MEMORY_CLEANUP_THRESHOLD = 0.8  # Memory cleanup threshold as percentage


class AzureServiceConstants:
    """Backward compatibility for Azure service constants"""

    DEFAULT_EMBEDDING_MODEL = InfrastructureConstants.DEFAULT_EMBEDDING_MODEL
    OPENAI_API_VERSION = InfrastructureConstants.OPENAI_API_VERSION
    MAX_TOKENS_GPT4 = InfrastructureConstants.MAX_TOKENS_GPT4
    MAX_SEARCH_RESULTS = InfrastructureConstants.MAX_SEARCH_RESULTS
    COGNITIVE_SERVICES_SCOPE = InfrastructureConstants.COGNITIVE_SERVICES_SCOPE


class ContentAnalysisConstants:
    """Backward compatibility for content analysis constants"""

    MEDIUM_COMPLEXITY_THRESHOLD = (
        ContentAnalysisAdaptiveConstants.MEDIUM_COMPLEXITY_THRESHOLD
    )
    HIGH_SIMILARITY_THRESHOLD = (
        ContentAnalysisAdaptiveConstants.HIGH_SIMILARITY_THRESHOLD
    )
    TECHNICAL_DENSITY_THRESHOLD = (
        ContentAnalysisAdaptiveConstants.TECHNICAL_DENSITY_THRESHOLD
    )
    SHORT_DOCUMENT_WORD_THRESHOLD = (
        ContentAnalysisAdaptiveConstants.SHORT_DOCUMENT_WORD_THRESHOLD
    )
    CHUNK_SIZE_MAX_FALLBACK = ContentAnalysisAdaptiveConstants.OPTIMAL_CHUNK_SIZE_MAX
    CHUNK_SIZE_MIN_FALLBACK = ContentAnalysisAdaptiveConstants.OPTIMAL_CHUNK_SIZE_MIN
    EXCELLENT_RESPONSE_TIME = 1.0  # Response time threshold for excellent performance
    GOOD_RESPONSE_TIME = 2.0  # Response time threshold for good performance


class MLModelConstants:
    """Backward compatibility for ML model constants"""

    GNN_HIDDEN_DIM = MLModelStaticConstants.GNN_HIDDEN_DIM
    GNN_LEARNING_RATE = MLModelStaticConstants.GNN_LEARNING_RATE
    BATCH_SIZE = MLModelStaticConstants.BATCH_SIZE
    EMBEDDING_DIMENSION = SystemBoundaryConstants.EMBEDDING_DIMENSION


# =============================================================================
# STUB CONSTANTS FOR IMPLEMENTATION TRANSITION
# =============================================================================


class StubConstants:
    """Constants for stub functions during implementation transition"""

    # Confidence calculation stubs
    STUB_ADAPTIVE_CONFIDENCE = (
        MathematicalConstants.BASE_CONFIDENCE
        + BaseScalingFactors.HIGH_CONFIDENCE_OFFSET
    )  # 0.8
    STUB_COMPLEXITY_SCORE = 0.5  # Medium complexity default

    # Text analysis stubs
    DEFAULT_MEDIUM_COMPLEXITY = "medium"

    # Dynamic configuration mathematical factors
    RELATIONSHIP_THRESHOLD_FACTOR = (
        0.9  # Factor to derive relationship threshold from entity threshold
    )
    MAX_WORDS_BASE = 500  # Base maximum words
    MAX_WORDS_MULTIPLIER = 20  # Multiplier for average sentence length

    # Extraction processor fallback configuration
    FALLBACK_ENTITY_THRESHOLD = (
        KnowledgeExtractionConstants.DEFAULT_CONFIDENCE_THRESHOLD
    )
    FALLBACK_RELATIONSHIP_THRESHOLD = (
        KnowledgeExtractionConstants.DEFAULT_RELATIONSHIP_CONFIDENCE_THRESHOLD
    )
    FALLBACK_CHUNK_SIZE = KnowledgeExtractionConstants.DEFAULT_CHUNK_SIZE
    FALLBACK_MAX_ENTITIES_PER_CHUNK = 20  # Reference to DomainAdaptiveConstants
    FALLBACK_MINIMUM_QUALITY_SCORE = MathematicalConstants.BASE_CONFIDENCE

    # Position and confidence factors for extraction
    EARLY_POSITION_FACTOR = 0.8  # Factor for entities found in first quarter of text
    LATE_POSITION_FACTOR = 0.6  # Factor for entities found later in text

    # Confidence distribution thresholds
    VERY_HIGH_CONFIDENCE_THRESHOLD = 0.9  # Threshold for very high confidence
    HIGH_CONFIDENCE_THRESHOLD = MathematicalConstants.BASE_CONFIDENCE
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6  # Threshold for medium confidence

    # Default statistical values
    DEFAULT_ZERO_FLOAT = 0.0  # Default zero value for statistical calculations
    PERCENTAGE_MULTIPLIER = 100  # Multiplier to convert ratio to percentage
    MAX_COVERAGE_PERCENTAGE = 100.0  # Maximum coverage percentage
    DEFAULT_MEMORY_LIMIT_MB = SystemBoundaryConstants.DEFAULT_MEMORY_LIMIT_MB
    MAX_AZURE_SERVICE_COST_USD = 100.0  # Maximum cost per operation
    MEMORY_CLEANUP_THRESHOLD = 0.8  # Memory cleanup threshold as percentage

    # Text processing constants
    DEFAULT_MAX_TEXT_LENGTH = int(
        MathematicalConstants.BASE_CHUNK_SIZE * BaseScalingFactors.STANDARD_CHUNK_FACTOR
    )  # 1000
    DEFAULT_TEXT_SUFFIX = "..."  # Default truncation suffix

    # Content analysis constants
    READABILITY_SCORE_DIVISOR = MathematicalConstants.PERCENTAGE_MULTIPLIER  # 100.0
    WORDS_PER_THOUSAND = MathematicalConstants.MS_PER_SECOND  # 1000.0
    AVG_WORDS_PER_SENTENCE_THRESHOLD = 20.0  # Threshold for average words per sentence

    # Frequency and scoring constants
    FREQUENCY_WEIGHT = 0.4  # Weight for frequency score in combined scoring
    TFIDF_WEIGHT = 0.6  # Weight for TF-IDF score in combined scoring

    # Content complexity tier weights
    COMPLEXITY_SIMPLE_WEIGHT = 0.2  # Weight for simple complexity
    COMPLEXITY_MODERATE_WEIGHT = 0.6  # Weight for moderate complexity
    COMPLEXITY_COMPLEX_WEIGHT = 0.8  # Weight for complex complexity
    COMPLEXITY_DEFAULT_WEIGHT = 0.5  # Default weight for unknown complexity

    # TF-IDF and analysis constants
    TFIDF_FEATURES_DIVISOR = 50.0  # Divisor for TF-IDF feature normalization
    TEXT_STATS_WORD_DIVISOR = 200.0  # Divisor for text statistics word normalization

    # Content scoring thresholds
    CONTENT_HIGH_SCORE_THRESHOLD = 0.8  # High content quality score threshold
    CONTENT_GOOD_SCORE_THRESHOLD = 0.6  # Good content quality score threshold
    CONTENT_FAIR_SCORE_THRESHOLD = 0.4  # Fair content quality score threshold

    # Pattern and model agreement constants
    MODEL_AGREEMENT_THRESHOLD = (
        0.8  # Threshold for model agreement in integration tests
    )
    PROCESSING_TIME_MS_MULTIPLIER = MathematicalConstants.MS_PER_SECOND  # 1000

    # Vocabulary richness thresholds
    VOCABULARY_HIGH_RICHNESS = 0.6  # High vocabulary richness threshold
    VOCABULARY_LOW_RICHNESS = 0.3  # Low vocabulary richness threshold

    # Content analysis confidence adjustments
    HIGH_WORD_COUNT_THRESHOLD = MathematicalConstants.MS_PER_SECOND  # 1000 words
    HIGH_WORD_COUNT_CONFIDENCE_BOOST = 0.2  # Confidence boost for high word count
    MEDIUM_WORD_COUNT_CONFIDENCE_BOOST = 0.1  # Confidence boost for medium word count

    # ML Model constants
    GNN_HIDDEN_DIM_STANDARD = 128  # Standard GNN hidden dimension
    GNN_HIDDEN_DIM_LARGE = 256  # Large GNN hidden dimension
    DEFAULT_DROPOUT_RATE = 0.5  # Standard dropout rate for ML models
    DEFAULT_LEARNING_RATE = 0.001  # Standard learning rate for training

    # Minimum text length thresholds
    MIN_CLEANED_TEXT_LENGTH = (
        MathematicalConstants.PERCENTAGE_MULTIPLIER
    )  # 100 characters

    # Performance and execution constants
    USER_COUNT_SUPPORTED = "100+"  # Number of concurrent users supported
    COMPLIANCE_PERFECT_SCORE = MathematicalConstants.PERCENTAGE_MULTIPLIER  # 100.0

    # Database limits and pagination
    DEFAULT_QUERY_LIMIT = MathematicalConstants.MS_PER_SECOND  # 1000 records

    # Statistical initialization values
    STAT_INITIAL_ZERO = MathematicalConstants.CONFIDENCE_MIN  # 0.0
    STAT_INITIAL_COUNT = 0  # Initial count value

    # Embedding model constants
    DEFAULT_EMBEDDING_MODEL_NAME = (
        "text-embedding-ada-002"  # Default Azure OpenAI embedding model
    )


# Export all backward compatibility classes
__all__ = [
    "CacheConstants",
    "ProcessingConstants",
    "AzureServiceConstants",
    "ContentAnalysisConstants",
    "MLModelConstants",
    "StubConstants",
]
