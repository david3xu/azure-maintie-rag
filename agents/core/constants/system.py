"""
System Boundary and Infrastructure Constants
============================================

This module contains system boundary constants (fundamental limits that should never be
auto-generated) and infrastructure constants (which can potentially be auto-configured
from Azure resource deployment or environment scanning).

System Boundary Constants:
- Physical/API limits set by external systems
- System resource limits (hardware constraints)
- Data structure limits to prevent system overload
- Core system behavior and fallbacks
- Error handling and security constants

Infrastructure Constants:
- Azure service configuration (can be discovered)
- Model deployments (learnable from deployment scanning)
- Service limits and performance (learnable from monitoring)
- File system paths and patterns

AUTO-GENERATION POTENTIAL:
- SystemBoundaryConstants: STATIC - Never auto-generate (system limits)
- InfrastructureConstants: HIGH - Can discover from Azure deployment
- FileSystemConstants: LOW - Conventional paths, rarely change
"""

from .base import BaseScalingFactors, MathematicalConstants


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
    BYTES_TO_GB_DIVISOR = MathematicalConstants.BYTES_PER_GB
    BYTES_TO_MB_DIVISOR = MathematicalConstants.BYTES_PER_MB

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
    MAX_CONFIDENCE = MathematicalConstants.CONFIDENCE_MAX
    ZERO_FLOAT = MathematicalConstants.CONFIDENCE_MIN
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
    MAX_TOKENS_GPT4 = 8192  # OpenAI GPT-4 token limit
    MAX_SEARCH_RESULTS = int(
        MathematicalConstants.BASE_RESULT_COUNT
        * BaseScalingFactors.ENTITY_RESULT_FACTOR
    )  # 50
    METRICS_COLLECTION_INTERVAL = 60  # Metrics collection interval (seconds)
    HEALTH_CHECK_INTERVAL = 30  # Health check interval (seconds)

    # Cosmos DB configuration
    MIN_COSMOS_THROUGHPUT_RU = 400  # Minimum Cosmos DB throughput


class FileSystemConstants:
    """File system paths and patterns - STATIC or CONFIGURABLE"""

    # AUTO-GENERATION POTENTIAL: LOW (conventional paths)

    # Configuration File Patterns - Conventional naming
    EXTRACTION_CONFIG_SUFFIX = "_extraction_config.yaml"
    SEARCH_CONFIG_SUFFIX = "_search_config.yaml"
    GENERAL_CONFIG_SUFFIX = "_config.yaml"


class SystemPerformanceConstants:
    """System performance and operational constants"""

    # Timeout values
    DEFAULT_TIMEOUT_SECONDS = 30
    HEALTH_CHECK_INTERVAL_SECONDS = 60

    # Retry and reliability
    DEFAULT_MAX_RETRIES = 3

    # Performance thresholds
    MAX_RESPONSE_TIME_MS = 3000.0
    MAX_ERROR_RATE = 0.05
    DEFAULT_SLA_AVAILABILITY_PERCENT = 99.9
    METRICS_WINDOW_MINUTES = 5

    # Concurrency and batching
    DEFAULT_CONCURRENT_REQUESTS = 10
    ML_BATCH_SIZE = 32
    SLOW_OPERATION_THRESHOLD_SECONDS = 5.0


# Export all constants
__all__ = [
    "SystemBoundaryConstants",
    "InfrastructureConstants",
    "FileSystemConstants",
    "SystemPerformanceConstants",
]
