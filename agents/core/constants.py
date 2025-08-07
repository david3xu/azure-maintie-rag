"""
Centralized Constants for Azure Universal RAG Agents
===================================================

This file centralizes constants organized by their potential for automation and interdependency.
Supports the zero-hardcoded-values philosophy with strategic categorization for future enhancement.

ORGANIZATION STRATEGY:
======================
1. STATIC SYSTEM CONSTANTS - Never change, system-level constraints
2. DOMAIN-LEARNABLE CONSTANTS - Can be learned from domain analysis and data
3. PERFORMANCE-ADAPTIVE CONSTANTS - Should adapt based on performance metrics and feedback
4. INFRASTRUCTURE CONSTANTS - Environment-dependent, can be auto-configured
5. WORKFLOW INTERDEPENDENT CONSTANTS - Work together, should be optimized as groups

Each category is marked with automation potential and interdependency notes.
"""

from typing import Dict, List, Tuple, Any

# =============================================================================
# CATEGORY 0: MATHEMATICAL AND BASE CONSTANTS
# =============================================================================
# Foundation constants that other constants derive from to eliminate duplication


class MathematicalConstants:
    """Core mathematical constants used throughout the system"""

    # Byte conversion factors
    BYTES_PER_KB = 1024
    BYTES_PER_MB = BYTES_PER_KB * 1024  # 1,048,576
    BYTES_PER_GB = BYTES_PER_MB * 1024  # 1,073,741,824

    # Time conversion factors
    MS_PER_SECOND = 1000.0
    SECONDS_PER_MINUTE = 60.0

    # Percentage and scoring
    PERCENTAGE_MULTIPLIER = 100.0
    CONFIDENCE_MIN = 0.0
    CONFIDENCE_MAX = 1.0

    # Base scaling factors to reduce duplication
    BASE_TIMEOUT = 30  # Base timeout in seconds
    BASE_RESULT_COUNT = 10  # Base result count for searches
    BASE_CHUNK_SIZE = 500  # Base chunk size in characters
    BASE_CONFIDENCE = 0.7  # Base confidence threshold


class BaseScalingFactors:
    """Scaling factors to derive related constants from base values"""

    # Timeout scaling
    AZURE_TIMEOUT_FACTOR = 2.0  # 60 = 30 * 2
    MAX_TIMEOUT_FACTOR = 20.0  # 600 = 30 * 20

    # Result count scaling
    SEARCH_RESULT_FACTOR = 2.0  # 20 = 10 * 2
    ENTITY_RESULT_FACTOR = 5.0  # 50 = 10 * 5

    # Chunk size scaling
    STANDARD_CHUNK_FACTOR = 2.0  # 1000 = 500 * 2
    LARGE_CHUNK_FACTOR = 4.0  # 2000 = 500 * 4

    # Confidence scaling
    HIGH_CONFIDENCE_OFFSET = 0.1  # 0.8 = 0.7 + 0.1
    STATISTICAL_CONFIDENCE_OFFSET = 0.05  # 0.75 = 0.7 + 0.05


# =============================================================================
# CATEGORY 1: STATIC SYSTEM CONSTANTS
# =============================================================================
# These constants represent fundamental system limitations and should NOT be auto-generated.
# They define physical/logical boundaries of the system architecture.


class SystemBoundaryConstants:
    """Fundamental system limits that should never be auto-generated"""

    # Physical/API Limits - Set by external systems
    EMBEDDING_DIMENSION = 1536  # OpenAI embedding model dimension

    # System Resource Limits - Hardware constraints (derived from base constants)
    MAX_EXECUTION_TIME_LIMIT = (
        MathematicalConstants.BASE_TIMEOUT * BaseScalingFactors.MAX_TIMEOUT_FACTOR
    )  # 600.0
    DEFAULT_MEMORY_LIMIT_MB = 200.0  # Memory constraint per operation
    # Aliases for backward compatibility
    BYTES_TO_GB_DIVISOR = (
        MathematicalConstants.BYTES_PER_GB
    )  # Alias for backward compatibility
    BYTES_TO_MB_DIVISOR = (
        MathematicalConstants.BYTES_PER_MB
    )  # Alias for backward compatibility

    # Data Structure Limits - Prevent system overload
    MAX_CONCURRENT_REQUESTS = 10  # Prevent resource exhaustion
    MAX_METRICS_HISTORY = 1000  # Prevent unbounded growth

    # Cache and System Constants - Core system behavior
    MIN_WORD_LENGTH_FOR_INDEXING = 3  # Minimum word length for search indexing
    PHRASE_MATCH_SCORE_MULTIPLIER = 2.0  # Phrase matching score multiplier
    DOMAIN_MATCH_TOP_RESULTS = 5  # Maximum domain matches to return
    CACHE_EVICTION_PERCENTAGE = 0.25  # Cache eviction percentage (25%)
    FIRST_ACCESS_COUNT = 1  # Initial access count for cache entries
    INITIAL_ACCESS_COUNT = 1  # Initial access count for new entries

    # Default Values and Fallbacks - Core system fallbacks (derived from base constants)
    DEFAULT_FALLBACK_DOMAIN = "general"  # Default domain when none detected
    DEFAULT_FALLBACK_CONFIDENCE = 0.5  # Default confidence when none available
    # Aliases for backward compatibility
    MAX_CONFIDENCE = (
        MathematicalConstants.CONFIDENCE_MAX
    )  # Alias for backward compatibility
    ZERO_FLOAT = (
        MathematicalConstants.CONFIDENCE_MIN
    )  # Alias for backward compatibility
    CACHE_PATTERN_SCORE_THRESHOLD = 0.1  # Cache pattern scoring threshold

    # Error Handling Constants - Core system error management
    FAILURE_COUNT_INITIAL = 0  # Initial failure count
    FAILURE_COUNT_INCREMENT = 1  # Failure count increment
    CRITICAL_ERROR_THRESHOLD = 50  # Critical error threshold
    WARNING_ERROR_THRESHOLD = 20  # Warning error threshold
    MIN_RECOVERY_RATE_THRESHOLD = 50  # Minimum recovery rate threshold


class InfrastructureConstants:
    """Infrastructure and service configuration - CAN BE AUTO-CONFIGURED"""

    # AUTO-GENERATION POTENTIAL: HIGH
    # These could be discovered from Azure resource deployment or environment scanning

    # Azure Service Configuration - Required for service authentication
    COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
    OPENAI_API_VERSION = "2024-08-01-preview"  # Could query latest supported version

    # Model Deployments - LEARNABLE from deployment scanning
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"  # Could discover from deployment

    # Service Limits and Performance - LEARNABLE from deployment monitoring
    MIN_SERVICES_FOR_OPERATIONAL = 3  # Minimum services needed for operations
    # PERCENTAGE_MULTIPLIER now in MathematicalConstants
    MAX_TOKENS_GPT4 = 8192  # OpenAI GPT-4 token limit
    MAX_SEARCH_RESULTS = int(
        MathematicalConstants.BASE_RESULT_COUNT
        * BaseScalingFactors.ENTITY_RESULT_FACTOR
    )  # 50
    METRICS_COLLECTION_INTERVAL = 60  # Metrics collection interval (seconds)
    HEALTH_CHECK_INTERVAL = 30  # Health check interval (seconds)


class FileSystemConstants:
    """File system paths and patterns - STATIC or CONFIGURABLE"""

    # AUTO-GENERATION POTENTIAL: LOW (conventional paths)

    # Configuration File Patterns - Conventional naming
    EXTRACTION_CONFIG_SUFFIX = "_extraction_config.yaml"
    SEARCH_CONFIG_SUFFIX = "_search_config.yaml"
    GENERAL_CONFIG_SUFFIX = "_config.yaml"


# =============================================================================
# CATEGORY 2: DOMAIN-LEARNABLE CONSTANTS
# =============================================================================
# These constants should be learned from domain analysis and document corpus.
# HIGH AUTOMATION POTENTIAL - Domain Intelligence Agent should generate these.


class DomainAdaptiveConstants:
    """Constants that should be learned from domain analysis"""

    # AUTO-GENERATION POTENTIAL: VERY HIGH
    # These work together as a group and should be optimized collectively

    # INTERDEPENDENT GROUP 1: Entity Extraction Thresholds
    # These thresholds should be learned from domain-specific entity density and quality
    ENTITY_CONFIDENCE_THRESHOLD = 0.8  # LEARNABLE: domain entity patterns
    RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.7  # LEARNABLE: domain relationship patterns
    MIN_RELATIONSHIP_STRENGTH = 0.5  # LEARNABLE: domain relationship quality

    # INTERDEPENDENT GROUP 2: Document Processing Parameters (derived from base constants)
    # These should be optimized together based on document characteristics
    DEFAULT_CHUNK_SIZE = int(
        MathematicalConstants.BASE_CHUNK_SIZE * BaseScalingFactors.STANDARD_CHUNK_FACTOR
    )  # 1000
    DEFAULT_CHUNK_OVERLAP = int(DEFAULT_CHUNK_SIZE * 0.2)  # 200 (20% of chunk size)
    MAX_ENTITIES_PER_CHUNK = int(MathematicalConstants.BASE_RESULT_COUNT * 2)  # 20
    MIN_ENTITY_LENGTH = 2  # LEARNABLE: domain-specific entity patterns
    MAX_ENTITY_LENGTH = 100  # LEARNABLE: domain-specific entity patterns

    # INTERDEPENDENT GROUP 3: Search Quality Thresholds
    # These work together to define search quality for specific domains
    RESULT_RELEVANCE_THRESHOLD = 0.6  # LEARNABLE: domain relevance patterns
    MIN_CONFIDENCE_THRESHOLD = 0.1  # LEARNABLE: minimum viable confidence for domain

    # INTERDEPENDENT GROUP 4: Domain Classification Thresholds (derived from base confidence)
    # These should be learned from multi-domain analysis
    DOMAIN_DETECTION_THRESHOLD = (
        MathematicalConstants.BASE_CONFIDENCE
        + BaseScalingFactors.STATISTICAL_CONFIDENCE_OFFSET
    )  # 0.75
    MIN_DOMAIN_CONFIDENCE = MathematicalConstants.BASE_CONFIDENCE  # 0.7
    DOMAIN_CLASSIFICATION_THRESHOLD = 0.6  # LEARNABLE: domain boundary definitions
    TECHNICAL_CONTENT_SIMILARITY_THRESHOLD = (
        0.8  # LEARNABLE: technical vs general content patterns
    )


class ContentAnalysisAdaptiveConstants:
    """Content analysis parameters that should adapt to corpus characteristics"""

    # AUTO-GENERATION POTENTIAL: HIGH
    # These should be learned from corpus analysis and document structure patterns

    # INTERDEPENDENT GROUP: Document Size Classifications
    # Only keeping used thresholds
    SHORT_DOCUMENT_WORD_THRESHOLD = 100  # LEARNABLE: corpus length distribution

    # INTERDEPENDENT GROUP: Complexity Analysis Thresholds
    # These should be learned from content complexity patterns in the domain
    MEDIUM_COMPLEXITY_THRESHOLD = 0.4  # LEARNABLE: domain complexity distribution
    HIGH_SIMILARITY_THRESHOLD = 0.4  # LEARNABLE: domain similarity patterns
    TECHNICAL_DENSITY_THRESHOLD = 0.1  # LEARNABLE: domain technical content patterns

    # LEARNABLE: Optimal chunk sizes based on document structure analysis (derived from base)
    OPTIMAL_CHUNK_SIZE_MIN = MathematicalConstants.BASE_CHUNK_SIZE  # 500
    OPTIMAL_CHUNK_SIZE_MAX = int(
        MathematicalConstants.BASE_CHUNK_SIZE * BaseScalingFactors.LARGE_CHUNK_FACTOR
    )  # 2000


# =============================================================================
# CATEGORY 3: PERFORMANCE-ADAPTIVE CONSTANTS
# =============================================================================
# These constants should adapt based on performance metrics and system feedback.
# MEDIUM-HIGH AUTOMATION POTENTIAL - Should be optimized by performance monitoring.


class PerformanceAdaptiveConstants:
    """Constants that should adapt based on system performance metrics"""

    # AUTO-GENERATION POTENTIAL: HIGH
    # These should be continuously optimized based on actual performance data

    # INTERDEPENDENT GROUP 1: Timeout and Retry Strategy (derived from base constants)
    # These work together to define resilience behavior - should be optimized as a group
    DEFAULT_TIMEOUT = MathematicalConstants.BASE_TIMEOUT  # 30
    AZURE_SERVICE_TIMEOUT = int(
        MathematicalConstants.BASE_TIMEOUT * BaseScalingFactors.AZURE_TIMEOUT_FACTOR
    )  # 60
    MAX_RETRIES = 3  # ADAPTIVE: based on failure patterns
    RETRY_DELAY = 1.0  # ADAPTIVE: based on recovery time patterns
    EXPONENTIAL_BACKOFF_MULTIPLIER = 2.0  # ADAPTIVE: based on service recovery patterns

    # INTERDEPENDENT GROUP 2: Batch Processing Optimization
    # These should be optimized together based on throughput vs latency tradeoffs
    DEFAULT_BATCH_SIZE = 10  # ADAPTIVE: optimal batch size for performance
    MAX_BATCH_SIZE = 100  # ADAPTIVE: based on memory and performance limits
    PARALLEL_WORKERS = 4  # ADAPTIVE: based on CPU cores and I/O patterns
    MAX_CONCURRENT_CHUNKS = 5  # ADAPTIVE: based on memory usage and throughput

    # INTERDEPENDENT GROUP 3: Cache Performance Tuning
    # These should be optimized together based on cache hit rates and memory usage
    DEFAULT_CACHE_TTL = 3600  # ADAPTIVE: based on typical access patterns
    SHORT_CACHE_TTL = 300  # ADAPTIVE: based on dynamic data patterns
    LONG_CACHE_TTL = 86400  # ADAPTIVE: based on stable data patterns
    TARGET_CACHE_HIT_RATE = 0.6  # ADAPTIVE: target based on performance analysis
    CACHE_CLEANUP_INTERVAL = 300  # ADAPTIVE: based on memory management needs

    # Cache Performance Thresholds - ADAPTIVE based on performance analysis
    CACHE_HIGH_PERFORMANCE_THRESHOLD = 0.8  # ADAPTIVE: high performance cache hit rate
    CACHE_EXCELLENT_PERFORMANCE_THRESHOLD = 0.9  # ADAPTIVE: excellent cache utilization
    CACHE_LOW_PERFORMANCE_THRESHOLD = 0.4  # ADAPTIVE: low performance threshold
    CACHE_VERY_HIGH_PERFORMANCE_THRESHOLD = (
        0.7  # ADAPTIVE: very high performance threshold
    )
    CACHE_OPTIMIZATION_MAX_SIZE = 1000  # ADAPTIVE: optimal cache size for memory

    # Time and Measurement Constants - use MathematicalConstants.MS_PER_SECOND
    SUB_MILLISECOND_THRESHOLD = 0.001  # Performance measurement threshold

    # SLA Targets - ADAPTIVE based on actual performance capability


class SearchPerformanceAdaptiveConstants:
    """Search-specific performance constants that should adapt"""

    # AUTO-GENERATION POTENTIAL: HIGH
    # These should be optimized based on search quality vs performance tradeoffs

    # INTERDEPENDENT GROUP: Tri-Modal Search Weights
    # These must sum to 1.0 and should be optimized together based on search quality
    MULTI_MODAL_WEIGHT_VECTOR = 0.4  # ADAPTIVE: based on vector search effectiveness
    MULTI_MODAL_WEIGHT_GRAPH = 0.3  # ADAPTIVE: based on graph search effectiveness
    MULTI_MODAL_WEIGHT_GNN = 0.3  # ADAPTIVE: based on GNN search effectiveness

    # INTERDEPENDENT GROUP: Search Result Limits
    # These work together to balance quality vs quantity
    DEFAULT_VECTOR_TOP_K = 10  # ADAPTIVE: optimal result count for quality
    MAX_SEARCH_RESULTS = 20  # ADAPTIVE: based on user interaction patterns
    DEFAULT_MAX_RESULTS_PER_MODALITY = 10  # ADAPTIVE: balanced across search types

    # INTERDEPENDENT GROUP: Search Processing Delays
    # These should reflect actual processing times for capacity planning


# =============================================================================
# CATEGORY 4: ALGORITHM AND MODEL CONSTANTS
# =============================================================================
# Constants for ML models and algorithms - Some static, some learnable.


class MLModelStaticConstants:
    """ML model constants that are algorithmically determined"""

    # AUTO-GENERATION POTENTIAL: LOW-MEDIUM

    # GNN Architecture - Could be learned through architecture search
    GNN_HIDDEN_DIM = 128  # POTENTIALLY LEARNABLE: architecture optimization
    GNN_NUM_LAYERS = 2  # POTENTIALLY LEARNABLE: architecture optimization

    # Training Hyperparameters - Should be learned through hyperparameter optimization
    GNN_LEARNING_RATE = 0.001  # LEARNABLE: hyperparameter optimization
    BATCH_SIZE = 32  # LEARNABLE: memory vs convergence optimization

    # Vector Search Configuration - Should be optimized for specific use case


class StatisticalConstants:
    """Statistical analysis constants - mix of standard values and learnable thresholds"""

    # AUTO-GENERATION POTENTIAL: MEDIUM

    # Standard Statistical Values - These are mathematically standard
    CHI_SQUARE_SIGNIFICANCE_ALPHA = 0.05  # STANDARD: statistical convention

    # Domain-Specific Statistical Thresholds - LEARNABLE
    STATISTICAL_CONFIDENCE_THRESHOLD = 0.75  # LEARNABLE: domain confidence requirements
    STATISTICAL_CONFIDENCE_MIN = 0.0  # Minimum confidence bound
    STATISTICAL_CONFIDENCE_MAX = 1.0  # Maximum confidence bound
    MIN_PATTERN_FREQUENCY = 3  # LEARNABLE: significant pattern frequency

    # Data Quality Thresholds - LEARNABLE from data characteristics

    # Domain Classification Thresholds - Referenced by data models
    MIN_DOMAIN_CONFIDENCE = DomainAdaptiveConstants.MIN_DOMAIN_CONFIDENCE
    DOMAIN_CLASSIFICATION_THRESHOLD = (
        DomainAdaptiveConstants.DOMAIN_CLASSIFICATION_THRESHOLD
    )
    TECHNICAL_CONTENT_SIMILARITY_THRESHOLD = (
        DomainAdaptiveConstants.TECHNICAL_CONTENT_SIMILARITY_THRESHOLD
    )
    RICH_VOCABULARY_FACTOR = 0.6  # Factor for rich vocabulary content analysis


# =============================================================================
# CATEGORY 5: WORKFLOW INTERDEPENDENT CONSTANTS
# =============================================================================
# Constants that work together in workflows and should be optimized as groups.


class WorkflowCoordinationConstants:
    """Workflow constants that must work together as coordinated groups"""

    # AUTO-GENERATION POTENTIAL: MEDIUM-HIGH
    # These should be optimized as interdependent groups

    # INTERDEPENDENT GROUP 1: Concurrency and Queue Management
    # These must be balanced together to prevent resource contention

    # INTERDEPENDENT GROUP 2: State Management and Persistence
    # These work together for workflow reliability

    # INTERDEPENDENT GROUP 3: Performance Grading and Quality Gates
    # These define coordinated quality thresholds across the system
    EXCELLENT_PERFORMANCE_THRESHOLD = (
        1.0  # COORDINATED: with good/acceptable thresholds
    )
    GOOD_PERFORMANCE_THRESHOLD = 2.0  # COORDINATED: with SLA targets
    ACCEPTABLE_PERFORMANCE_THRESHOLD = 3.0  # COORDINATED: with SLA limits
    MIN_EXTRACTION_ACCURACY = 0.85  # COORDINATED: with search relevance requirements
    MIN_SEARCH_RELEVANCE = 0.7  # COORDINATED: with extraction accuracy

    # INTERDEPENDENT GROUP 4: Multi-Modal Search Coordination
    # These must be coordinated for effective tri-modal search
    CONFIDENCE_WEIGHT = 0.4  # COORDINATED: with agreement and quality weights
    AGREEMENT_WEIGHT = 0.3  # COORDINATED: synthesis weight coordination
    QUALITY_WEIGHT = 0.3  # COORDINATED: synthesis weight coordination


class ErrorHandlingCoordinatedConstants:
    """Error handling constants that work together for system resilience"""

    # AUTO-GENERATION POTENTIAL: MEDIUM
    # These should be coordinated for consistent error handling behavior

    # INTERDEPENDENT GROUP: Circuit Breaker and Fallback Coordination
    DEFAULT_FAILURE_THRESHOLD = 5  # COORDINATED: with retry attempts and recovery
    CRITICAL_ERROR_THRESHOLD = 50  # COORDINATED: with warning threshold
    WARNING_ERROR_THRESHOLD = 20  # COORDINATED: with critical threshold
    DEFAULT_CONFIDENCE_FALLBACK = 0.3  # COORDINATED: with minimum confidence thresholds
    MIN_RECOVERY_RATE_THRESHOLD = 50  # COORDINATED: with failure thresholds


# =============================================================================
# CATEGORY 6: SECURITY AND HASHING CONSTANTS
# =============================================================================
# Security-related constants - mostly static for consistency and security.


class SecurityConstants:
    """Security constants - mostly static for consistency"""

    # AUTO-GENERATION POTENTIAL: LOW (security requires consistency)

    # Cryptographic Standards - STATIC for security consistency
    DEFAULT_HASH_ALGORITHM = "sha256"  # STATIC: security standard
    HASH_ENCODING = "utf-8"  # STATIC: encoding standard
    CACHE_KEY_SEPARATOR = "|"  # STATIC: consistent key format
    JSON_SORT_KEYS = True  # STATIC: consistent serialization

    # Service Access Levels - STATIC security classifications


# =============================================================================
# CATEGORY 7: SPECIALIZED EXTRACTION CONSTANTS
# =============================================================================
# Constants specific to knowledge extraction algorithms.


class ExtractionAlgorithmConstants:
    """Specialized constants for extraction algorithms"""

    # AUTO-GENERATION POTENTIAL: MEDIUM-HIGH
    # Many of these could be learned from extraction performance analysis

    # INTERDEPENDENT GROUP 1: Entity Recognition Weights
    # These work together for entity scoring and should be optimized as a group

    # INTERDEPENDENT GROUP 2: Confidence Calculation Components
    # These work together for relationship confidence and should be coordinated

    # Pattern Matching Limits - LEARNABLE from corpus analysis

    # Confidence Enhancement Factors - LEARNABLE from performance analysis
    CONFIDENCE_BOOST_FACTOR = 1.3  # LEARNABLE: optimal confidence adjustment


# =============================================================================
# CATEGORY 8: DATA STRUCTURE AND VALIDATION CONSTANTS
# =============================================================================
# Constants that define required data structures - mostly static for consistency.


class DataModelConstants:
    """Required keys and structure definitions for centralized data models"""

    # AUTO-GENERATION POTENTIAL: LOW (structural consistency requirements)

    # Configuration Structure Keys - STATIC for API consistency


# =============================================================================
# SPECIALIZED AGENT CONSTANTS (LEGACY ORGANIZATION)
# =============================================================================
# These maintain backward compatibility but could be reorganized into above categories.


class DomainIntelligenceConstants:
    """Domain Intelligence Agent specific constants"""

    # Many of these should move to DomainAdaptiveConstants for auto-generation


class KnowledgeExtractionConstants:
    """Knowledge Extraction Agent specific constants"""

    # Many of these are duplicated in DomainAdaptiveConstants and should be consolidated

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


class UniversalSearchConstants:
    """Universal Search Agent specific constants"""

    # Many of these are duplicated in SearchPerformanceAdaptiveConstants

    DEFAULT_GRAPH_HOP_COUNT = 2  # LEARNABLE: optimal graph traversal depth
    DEFAULT_GNN_NODE_EMBEDDINGS = 128  # LEARNABLE: optimal embedding dimensionality
    GRAPH_MIN_RELATIONSHIP_STRENGTH = 0.5  # LEARNABLE: domain relationship quality
    GNN_MIN_PREDICTION_CONFIDENCE = 0.6  # ADAPTIVE: GNN calibration

    # Search Result Processing - Could be performance adaptive
    DEFAULT_VECTOR_TOP_K = 10  # ADAPTIVE: optimal result count for quality
    DEFAULT_MAX_DEPTH = 3  # Maximum graph traversal depth
    DEFAULT_MAX_ENTITIES = 50  # Maximum entities to extract
    DEFAULT_MAX_PREDICTIONS = 20  # Maximum GNN predictions
    DEFAULT_PATTERN_THRESHOLD = 0.7  # GNN pattern recognition threshold
    DEFAULT_MIN_TRAINING_EXAMPLES = 100  # Minimum examples for GNN training
    DEFAULT_RELATIONSHIP_THRESHOLD = 0.5  # Default relationship confidence threshold
    VECTOR_SIMILARITY_THRESHOLD = 0.7  # LEARNABLE: domain semantic similarity patterns
    MAX_SEARCH_RESULTS = 20  # ADAPTIVE: user interaction optimization
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # LEARNABLE: domain confidence distributions

    # Multi-modal search weights - must sum to 1.0
    MULTI_MODAL_WEIGHT_VECTOR = 0.4  # ADAPTIVE: based on vector search effectiveness
    MULTI_MODAL_WEIGHT_GRAPH = 0.3  # ADAPTIVE: based on graph search effectiveness
    MULTI_MODAL_WEIGHT_GNN = 0.3  # ADAPTIVE: based on GNN search effectiveness

    # Query Complexity Analysis - Could be learned from query patterns
    QUERY_LENGTH_SIMPLE_THRESHOLD = 5  # LEARNABLE: query complexity classification
    QUERY_LENGTH_COMPLEX_THRESHOLD = 10  # LEARNABLE: query complexity classification
    MAX_GRAPH_HOP_COUNT = 5  # Maximum graph traversal depth
    QUERY_COMPLEXITY_SIMPLE_MULTIPLIER = 0.8  # ADAPTIVE: complexity-based optimization
    QUERY_COMPLEXITY_MEDIUM_MULTIPLIER = 1.2  # ADAPTIVE: complexity-based optimization
    QUERY_COMPLEXITY_COMPLEX_MULTIPLIER = 1.3  # ADAPTIVE: complexity-based optimization

    # Fallback constants for backward compatibility
    FALLBACK_VECTOR_SIMILARITY_THRESHOLD = 0.7  # Domain semantic similarity patterns
    FALLBACK_VECTOR_TOP_K = 10  # Optimal result count for quality
    FALLBACK_GRAPH_HOP_COUNT = 2  # Based on relationship depth analysis
    FALLBACK_GRAPH_MIN_RELATIONSHIP_STRENGTH = 0.5  # Domain-specific threshold
    FALLBACK_GNN_PREDICTION_CONFIDENCE = 0.6  # Based on model performance for domain
    FALLBACK_GNN_NODE_EMBEDDINGS = 128  # Optimized for domain complexity
    FALLBACK_RESULT_SYNTHESIS_THRESHOLD = 0.6  # Quality threshold for domain


# =============================================================================
# WORKFLOW LEGACY CONSTANTS
# =============================================================================
# Workflow constants maintained for backward compatibility.


class WorkflowConstants:
    """Workflow orchestration constants - many could be moved to WorkflowCoordinationConstants"""

    # File Management - STATIC conventions

    # Confidence Levels - Could be LEARNABLE from workflow success patterns
    DISCOVERY_CONFIDENCE = 0.9  # LEARNABLE: workflow step confidence
    ANALYSIS_CONFIDENCE = 0.85  # LEARNABLE: analysis quality patterns
    PATTERN_CONFIDENCE = 0.8  # LEARNABLE: pattern recognition quality
    CONFIG_CONFIDENCE = 0.9  # LEARNABLE: configuration generation quality
    EXTRACTION_CONFIDENCE = 0.75  # LEARNABLE: extraction quality patterns
    QUALITY_SCORE = 0.85  # LEARNABLE: overall quality requirements
    VALIDATION_CONFIDENCE = 0.9  # LEARNABLE: validation effectiveness

    # Orchestration Limits - Could be ADAPTIVE based on system capacity

    # Node Execution - Could be ADAPTIVE based on reliability patterns

    # Utility constants for workflow operations
    BYTES_TO_MB_DIVISOR = (
        MathematicalConstants.BYTES_PER_MB
    )  # Use consolidated constant
    STORAGE_SIZE_PRECISION = 2  # Decimal precision for storage size


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
    MS_PER_SECOND = MathematicalConstants.MS_PER_SECOND  # Use consolidated constant
    ZERO_FLOAT = MathematicalConstants.CONFIDENCE_MIN  # Use consolidated constant
    MAX_METRICS_HISTORY = SystemBoundaryConstants.MAX_METRICS_HISTORY
    BYTES_TO_GB_DIVISOR = (
        MathematicalConstants.BYTES_PER_GB
    )  # Use consolidated constant
    MEMORY_PRECISION_DECIMAL = 2  # Decimal precision for memory measurements
    TIME_PRECISION_DECIMAL = 1  # Decimal precision for time measurements
    PERCENTAGE_MULTIPLIER = (
        MathematicalConstants.PERCENTAGE_MULTIPLIER
    )  # Use consolidated constant
    CACHE_TTL_SECONDS = (
        PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL
    )  # Backward compatibility
    ACCEPTABLE_PROCESSING_TIME = 3.0  # Acceptable processing time in seconds


class ProcessingConstants:
    """Backward compatibility for processing constants"""

    DEFAULT_TIMEOUT = PerformanceAdaptiveConstants.DEFAULT_TIMEOUT
    AZURE_SERVICE_TIMEOUT = PerformanceAdaptiveConstants.AZURE_SERVICE_TIMEOUT
    MAX_RETRIES = PerformanceAdaptiveConstants.MAX_RETRIES
    MAX_CONCURRENT_CHUNKS = PerformanceAdaptiveConstants.MAX_CONCURRENT_CHUNKS
    MAX_EXECUTION_TIME_LIMIT = SystemBoundaryConstants.MAX_EXECUTION_TIME_LIMIT
    MAX_EXECUTION_TIME_SECONDS = (
        SystemBoundaryConstants.MAX_EXECUTION_TIME_LIMIT
    )  # Alias for seconds
    MAX_EXECUTION_TIME_MIN = (
        SystemBoundaryConstants.MAX_EXECUTION_TIME_LIMIT / 60.0
    )  # Convert to minutes
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
# UTILITY FUNCTIONS FOR CONSTANT ACCESS
# =============================================================================


def get_constant_by_category(category: str) -> Dict[str, Any]:
    """
    Get all constants for a specific category

    Args:
        category: Category name from the organized structure above

    Returns:
        Dictionary of constants for the category
    """
    category_mapping = {
        # New organized categories
        "system_boundary": SystemBoundaryConstants,
        "infrastructure": InfrastructureConstants,
        "filesystem": FileSystemConstants,
        "domain_adaptive": DomainAdaptiveConstants,
        "content_adaptive": ContentAnalysisAdaptiveConstants,
        "performance_adaptive": PerformanceAdaptiveConstants,
        "search_performance_adaptive": SearchPerformanceAdaptiveConstants,
        "ml_model_static": MLModelStaticConstants,
        "statistical": StatisticalConstants,
        "workflow_coordination": WorkflowCoordinationConstants,
        "error_handling_coordinated": ErrorHandlingCoordinatedConstants,
        "security": SecurityConstants,
        "extraction_algorithm": ExtractionAlgorithmConstants,
        "data_models": DataModelConstants,
        # Legacy categories (for backward compatibility)
        "domain": DomainIntelligenceConstants,
        "extraction": KnowledgeExtractionConstants,
        "search": UniversalSearchConstants,
        "workflow": WorkflowConstants,
    }

    if category not in category_mapping:
        raise ValueError(
            f"Unknown category: {category}. Available: {list(category_mapping.keys())}"
        )

    const_class = category_mapping[category]
    return {
        attr: getattr(const_class, attr)
        for attr in dir(const_class)
        if not attr.startswith("_")
    }


def get_automation_potential_summary() -> Dict[str, str]:
    """
    Get summary of automation potential for each constant category

    Returns:
        Dictionary mapping category to automation potential description
    """
    return {
        "system_boundary": "STATIC - Never auto-generate (system limits)",
        "infrastructure": "HIGH - Can discover from Azure deployment",
        "filesystem": "LOW - Conventional paths, rarely change",
        "domain_adaptive": "VERY HIGH - Should be learned by Domain Intelligence Agent",
        "content_adaptive": "HIGH - Learn from corpus analysis",
        "performance_adaptive": "HIGH - Optimize from performance metrics",
        "search_performance_adaptive": "HIGH - Optimize from search quality metrics",
        "ml_model_static": "MEDIUM - Some hyperparameters learnable",
        "statistical": "MEDIUM - Mix of standards and learnable thresholds",
        "workflow_coordination": "MEDIUM-HIGH - Optimize interdependent groups",
        "error_handling_coordinated": "MEDIUM - Coordinate for resilience",
        "security": "LOW - Keep static for consistency",
        "extraction_algorithm": "MEDIUM-HIGH - Learn from extraction performance",
        "data_models": "LOW - Keep static for API consistency",
    }


def get_interdependency_groups() -> Dict[str, List[str]]:
    """
    Get groups of constants that should be optimized together

    Returns:
        Dictionary mapping group names to lists of interdependent constants
    """
    return {
        "entity_extraction_quality": [
            "ENTITY_CONFIDENCE_THRESHOLD",
            "RELATIONSHIP_CONFIDENCE_THRESHOLD",
            "MIN_RELATIONSHIP_STRENGTH",
            "MAX_ENTITIES_PER_CHUNK",
        ],
        "document_processing_parameters": [
            "DEFAULT_CHUNK_SIZE",
            "DEFAULT_CHUNK_OVERLAP",
            "MIN_ENTITY_LENGTH",
            "MAX_ENTITY_LENGTH",
        ],
        "search_quality_thresholds": [
            "RESULT_RELEVANCE_THRESHOLD",
            "MIN_CONFIDENCE_THRESHOLD",
        ],
        "tri_modal_search_weights": [
            "MULTI_MODAL_WEIGHT_VECTOR",
            "MULTI_MODAL_WEIGHT_GRAPH",
            "MULTI_MODAL_WEIGHT_GNN",
        ],
        "timeout_retry_strategy": [
            "DEFAULT_TIMEOUT",
            "MAX_RETRIES",
            "RETRY_DELAY",
            "EXPONENTIAL_BACKOFF_MULTIPLIER",
        ],
        "batch_processing_optimization": [
            "DEFAULT_BATCH_SIZE",
            "PARALLEL_WORKERS",
            "MAX_CONCURRENT_CHUNKS",
            "MAX_BATCH_SIZE",
        ],
        "cache_performance_tuning": [
            "DEFAULT_CACHE_TTL",
            "TARGET_CACHE_HIT_RATE",
            "SHORT_CACHE_TTL",
            "LONG_CACHE_TTL",
        ],
        "synthesis_weights": [
            "CONFIDENCE_WEIGHT",
            "AGREEMENT_WEIGHT",
            "QUALITY_WEIGHT",
        ],
    }


def validate_constants() -> List[str]:
    """
    Validate that all constants are properly defined and interdependent groups are consistent

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []

    # Validate tri-modal weights sum to 1.0
    total_weight = (
        SearchPerformanceAdaptiveConstants.MULTI_MODAL_WEIGHT_VECTOR
        + SearchPerformanceAdaptiveConstants.MULTI_MODAL_WEIGHT_GRAPH
        + SearchPerformanceAdaptiveConstants.MULTI_MODAL_WEIGHT_GNN
    )
    if abs(total_weight - 1.0) > 0.001:
        errors.append(f"Tri-modal weights sum to {total_weight}, should sum to 1.0")

    # Validate synthesis weights sum to 1.0
    synthesis_total = (
        WorkflowCoordinationConstants.CONFIDENCE_WEIGHT
        + WorkflowCoordinationConstants.AGREEMENT_WEIGHT
        + WorkflowCoordinationConstants.QUALITY_WEIGHT
    )
    if abs(synthesis_total - 1.0) > 0.001:
        errors.append(f"Synthesis weights sum to {synthesis_total}, should sum to 1.0")

    # Validate threshold relationships
    if (
        DomainAdaptiveConstants.RELATIONSHIP_CONFIDENCE_THRESHOLD
        > DomainAdaptiveConstants.ENTITY_CONFIDENCE_THRESHOLD
    ):
        errors.append(
            "Relationship confidence threshold should not exceed entity confidence threshold"
        )

    # Validate chunk overlap is less than chunk size
    if (
        DomainAdaptiveConstants.DEFAULT_CHUNK_OVERLAP
        >= DomainAdaptiveConstants.DEFAULT_CHUNK_SIZE
    ):
        errors.append("Chunk overlap should be less than chunk size")

    # Validate performance thresholds are in order
    perf_thresholds = [
        WorkflowCoordinationConstants.EXCELLENT_PERFORMANCE_THRESHOLD,
        WorkflowCoordinationConstants.GOOD_PERFORMANCE_THRESHOLD,
        WorkflowCoordinationConstants.ACCEPTABLE_PERFORMANCE_THRESHOLD,
    ]
    if perf_thresholds != sorted(perf_thresholds):
        errors.append("Performance thresholds should be in ascending order")

    return errors


# Export organized constant classes for easy importing
__all__ = [
    # Base Constants (new)
    "MathematicalConstants",
    "BaseScalingFactors",
    # Organized Categories
    "SystemBoundaryConstants",
    "InfrastructureConstants",
    "FileSystemConstants",
    "DomainAdaptiveConstants",
    "ContentAnalysisAdaptiveConstants",
    "PerformanceAdaptiveConstants",
    "SearchPerformanceAdaptiveConstants",
    "MLModelStaticConstants",
    "StatisticalConstants",
    "WorkflowCoordinationConstants",
    "ErrorHandlingCoordinatedConstants",
    "SecurityConstants",
    "ExtractionAlgorithmConstants",
    "DataModelConstants",
    # Legacy Categories (backward compatibility)
    "DomainIntelligenceConstants",
    "KnowledgeExtractionConstants",
    "UniversalSearchConstants",
    "WorkflowConstants",
    # Backward Compatibility Aliases
    "CacheConstants",
    "ProcessingConstants",
    "AzureServiceConstants",
    "ContentAnalysisConstants",
    "MLModelConstants",
    # Utility Functions
    "get_constant_by_category",
    "get_automation_potential_summary",
    "get_interdependency_groups",
    "validate_constants",
]
