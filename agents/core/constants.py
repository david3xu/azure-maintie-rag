"""
Core Constants - Simplified
===========================

Essential constants for the PydanticAI agent system.
Only includes the minimal constants needed for operation.
"""


# Azure Service Constants
class AzureConstants:
    """Azure service configuration constants"""

    DEFAULT_API_VERSION = "2024-10-21"
    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3


# Processing Constants
class ProcessingConstants:
    """Core processing parameters"""

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8


# Model Configuration Constants
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4000
DEFAULT_REQUESTS_PER_MINUTE = 50
DEFAULT_CHUNK_SIZE = 1000


# Stub Constants for infrastructure compatibility
class StubConstants:
    """Compatibility constants for infrastructure imports"""

    DEFAULT_TIMEOUT = 30
    MAX_RESULTS = 100
    DEFAULT_THRESHOLD = 0.7

    # GNN Model Parameters
    GNN_HIDDEN_DIM_STANDARD = 128
    GNN_NUM_LAYERS_STANDARD = 2
    GNN_DROPOUT_RATE_STANDARD = 0.1
    DEFAULT_DROPOUT_RATE = 0.5
    DEFAULT_LEARNING_RATE = 0.001

    # Search Parameters
    VECTOR_SEARCH_TOP_K = 10
    GRAPH_TRAVERSAL_DEPTH = 3
    VOCABULARY_HIGH_RICHNESS = 0.6
    STAT_INITIAL_ZERO = 0.0

    # Processing Parameters
    BATCH_SIZE_STANDARD = 32
    CHUNK_SIZE_OPTIMAL = 1000
    COMPLIANCE_PERFECT_SCORE = 100
    DEFAULT_QUERY_LIMIT = 1000
    MAX_CONCURRENT_REQUESTS = 10


# Confidence Thresholds
class ConfidenceConstants:
    """Confidence scoring thresholds"""

    MIN_CONFIDENCE = 0.0
    DEFAULT_THRESHOLD = 0.8
    HIGH_CONFIDENCE = 0.9
    MAX_CONFIDENCE = 1.0


# Complexity Thresholds for Adaptive Processing
class ComplexityConstants:
    """Content complexity measurement thresholds"""
    
    COMPLEXITY_LOW_THRESHOLD = 0.4
    COMPLEXITY_MEDIUM_THRESHOLD = 0.7
    COMPLEXITY_HIGH_THRESHOLD = 0.8


# Export all constants
__all__ = ["AzureConstants", "ProcessingConstants", "ConfidenceConstants", "ComplexityConstants"]
