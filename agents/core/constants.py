"""
Centralized Constants for Azure Universal RAG Agents

This file centralizes all hardcoded values found across the agents directory
to support the zero-hardcoded-values philosophy and improve maintainability.

Categories:
- Azure Service Configurations
- ML Model Parameters
- Processing Configuration
- File and Path Constants
- Performance Metrics
- Cache and Concurrency
- Statistical Analysis
"""

from typing import Dict, List, Tuple, Any

# =============================================================================
# AZURE SERVICE CONFIGURATIONS
# =============================================================================

class AzureServiceConstants:
    """Azure service related constants"""

    # API Versions
    OPENAI_API_VERSION = "2024-08-01-preview"
    SEARCH_API_VERSION = "2023-11-01"
    COSMOS_API_VERSION = "2020-04-01"

    # Model Deployments
    DEFAULT_CHAT_MODEL = "gpt-4o"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
    FALLBACK_MODEL = "gpt-4.1-mini"

    # Service Endpoints
    OPENAI_ENDPOINT_SUFFIX = ".openai.azure.com"
    SEARCH_ENDPOINT_SUFFIX = ".search.windows.net"
    COSMOS_ENDPOINT_SUFFIX = ".gremlin.cosmos.azure.com"
    COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

    # Request Limits
    MAX_TOKENS_GPT4 = 8192
    MAX_TOKENS_GPT35 = 4096
    MAX_EMBEDDING_BATCH_SIZE = 100
    MAX_SEARCH_RESULTS = 50


# =============================================================================
# ML MODEL PARAMETERS
# =============================================================================

class MLModelConstants:
    """Machine learning model configuration constants"""

    # Embedding Dimensions
    EMBEDDING_DIMENSION = 1536
    OPENAI_EMBEDDING_DIM = 1536
    CUSTOM_EMBEDDING_DIM = 768

    # Vector Search Parameters
    VECTOR_SEARCH_TOP_K = 10
    SIMILARITY_THRESHOLD = 0.7
    VECTOR_INDEX_ALGORITHM = "cosine"

    # GNN Model Architecture
    GNN_HIDDEN_LAYERS = [256, 128, 64]
    GNN_OUTPUT_DIM = 32
    GNN_DROPOUT_RATE = 0.1
    GNN_LEARNING_RATE = 0.001

    # Training Parameters
    TRAINING_EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10


# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

class ProcessingConstants:
    """Data processing and workflow constants"""

    # Timeouts (in seconds)
    DEFAULT_TIMEOUT = 30
    LONG_OPERATION_TIMEOUT = 300
    AZURE_SERVICE_TIMEOUT = 60
    EXTRACTION_TIMEOUT = 120

    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    EXPONENTIAL_BACKOFF_MULTIPLIER = 2.0

    # Batch Processing
    DEFAULT_BATCH_SIZE = 10
    MAX_BATCH_SIZE = 100
    MIN_BATCH_SIZE = 1
    PARALLEL_WORKERS = 4

    # Text Processing
    MAX_TEXT_LENGTH = 8000
    MIN_TEXT_LENGTH = 50
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200


# =============================================================================
# FILE AND PATH CONSTANTS
# =============================================================================

class PathConstants:
    """File system and path related constants"""

    # Directory Names
    CACHE_DIR = "cache"
    LOGS_DIR = "logs"
    DATA_DIR = "data"
    MODELS_DIR = "models"
    TEMP_DIR = "temp"

    # File Extensions
    JSON_EXT = ".json"
    YAML_EXT = ".yaml"
    TXT_EXT = ".txt"
    CSV_EXT = ".csv"
    PKL_EXT = ".pkl"

    # Config File Patterns
    EXTRACTION_CONFIG_SUFFIX = "_extraction_config.yaml"
    SEARCH_CONFIG_SUFFIX = "_search_config.yaml"
    GENERAL_CONFIG_SUFFIX = "_config.yaml"
    JSON_GLOB_PATTERN = "*.json"

    # Configuration Files
    AGENT_CONFIG_FILE = "agent_config.yaml"
    DOMAIN_CONFIG_FILE = "domain_config.json"
    PERFORMANCE_LOG_FILE = "performance.log"
    ERROR_LOG_FILE = "error.log"

    # Cache Subdirectories
    PATTERN_CACHE_DIR = "patterns"
    EMBEDDING_CACHE_DIR = "embeddings"
    MODEL_CACHE_DIR = "models"
    WORKFLOW_CACHE_DIR = "workflow_states"


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

class PerformanceConstants:
    """Performance monitoring and SLA constants"""

    # SLA Targets (in seconds)
    QUERY_PROCESSING_SLA = 3.0
    KNOWLEDGE_EXTRACTION_SLA = 5.0
    VECTOR_SEARCH_SLA = 1.0
    GRAPH_TRAVERSAL_SLA = 2.0

    # Accuracy Thresholds
    MIN_EXTRACTION_ACCURACY = 0.85
    MIN_SEARCH_RELEVANCE = 0.7
    MIN_CONFIDENCE_THRESHOLD = 0.8

    # Cache Performance
    TARGET_CACHE_HIT_RATE = 0.6
    CACHE_EXPIRY_HOURS = 24
    MAX_CACHE_SIZE_MB = 500

    # Monitoring Intervals
    HEALTH_CHECK_INTERVAL = 60
    METRICS_COLLECTION_INTERVAL = 300
    LOG_ROTATION_DAYS = 7


# =============================================================================
# CACHE AND CONCURRENCY
# =============================================================================

class CacheConstants:
    """Cache and concurrency management constants"""

    # Cache TTL (Time To Live) in seconds
    DEFAULT_CACHE_TTL = 3600  # 1 hour
    SHORT_CACHE_TTL = 300     # 5 minutes
    LONG_CACHE_TTL = 86400    # 24 hours

    # Config freshness validation
    CONFIG_FRESHNESS_THRESHOLD_SECONDS = 86400  # 24 hours
    CONFIG_DEFAULT_TIMESTAMP = "2000-01-01"  # Default timestamp for missing config

    # Performance Tracking
    SUB_MILLISECOND_THRESHOLD = 0.001  # 1 millisecond for performance tracking

    # Cache Performance Metrics
    CACHE_AVERAGE_WEIGHT_OLD = 0.9  # Weight for old average in moving average calculation
    CACHE_AVERAGE_WEIGHT_NEW = 0.1  # Weight for new value in moving average calculation
    CACHE_PATTERN_SCORE_THRESHOLD = 0.5  # Minimum score for pattern matching
    CACHE_OPTIMIZATION_MAX_SIZE = 2000  # Maximum cache size before optimization
    CACHE_OPTIMIZATION_REDUCE_SIZE = 3600  # Target size after optimization
    CACHE_CLEANUP_INTERVAL = 1800  # Seconds between cache cleanup operations
    CACHE_HEALTH_CHECK_THRESHOLD = 0.3  # Minimum health score threshold
    CACHE_PATTERN_BUFFER_SIZE = 1000  # Buffer size for pattern operations
    CACHE_HIGH_PERFORMANCE_THRESHOLD = 70  # High performance percentile
    CACHE_EXCELLENT_PERFORMANCE_THRESHOLD = 90  # Excellent performance percentile
    CACHE_LOW_PERFORMANCE_THRESHOLD = 50  # Low performance threshold
    CACHE_VERY_HIGH_PERFORMANCE_THRESHOLD = 95  # Very high performance threshold

    # Cache Keys
    EMBEDDING_CACHE_PREFIX = "emb:"
    SEARCH_CACHE_PREFIX = "search:"
    KNOWLEDGE_CACHE_PREFIX = "knowledge:"
    CONFIG_CACHE_PREFIX = "config:"

    # Concurrency Limits
    MAX_CONCURRENT_REQUESTS = 10
    MAX_CONCURRENT_EXTRACTIONS = 5
    MAX_CONCURRENT_SEARCHES = 8
    SEMAPHORE_TIMEOUT = 30

    # Queue Sizes
    DEFAULT_QUEUE_SIZE = 100
    PRIORITY_QUEUE_SIZE = 50
    BULK_QUEUE_SIZE = 500


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

class StatisticalConstants:
    """Statistical analysis and data quality constants"""

    # Statistical Thresholds
    OUTLIER_THRESHOLD = 2.0  # Standard deviations
    CORRELATION_THRESHOLD = 0.5
    P_VALUE_THRESHOLD = 0.05
    CONFIDENCE_INTERVAL = 0.95

    # Data Quality
    MIN_SAMPLE_SIZE = 10
    MAX_MISSING_DATA_RATIO = 0.1
    MIN_DATA_QUALITY_SCORE = 0.8

    # Pattern Recognition
    MIN_PATTERN_FREQUENCY = 3
    PATTERN_SIMILARITY_THRESHOLD = 0.8
    MAX_PATTERN_COMPLEXITY = 5

    # Domain Analysis
    MIN_DOMAIN_CONFIDENCE = 0.7
    DOMAIN_CLASSIFICATION_THRESHOLD = 0.6
    MAX_DOMAIN_CATEGORIES = 10


# =============================================================================
# ERROR HANDLING AND STATUS CODES
# =============================================================================

class ErrorConstants:
    """Error handling and status code constants"""

    # Custom Error Codes
    AZURE_SERVICE_ERROR = "AZ001"
    CONFIGURATION_ERROR = "CFG001"
    PROCESSING_ERROR = "PROC001"
    VALIDATION_ERROR = "VAL001"
    TIMEOUT_ERROR = "TO001"

    # Error Messages
    SERVICE_UNAVAILABLE_MSG = "Azure service temporarily unavailable"
    INVALID_CONFIG_MSG = "Invalid configuration provided"
    PROCESSING_FAILED_MSG = "Data processing failed"
    TIMEOUT_MSG = "Operation timed out"
    VALIDATION_FAILED_MSG = "Data validation failed"

    # HTTP Status Codes
    SUCCESS_STATUS = 200
    BAD_REQUEST_STATUS = 400
    UNAUTHORIZED_STATUS = 401
    NOT_FOUND_STATUS = 404
    INTERNAL_ERROR_STATUS = 500
    SERVICE_UNAVAILABLE_STATUS = 503


# =============================================================================
# AGENT-SPECIFIC CONFIGURATIONS
# =============================================================================

class DomainIntelligenceConstants:
    """Constants specific to Domain Intelligence Agent"""

    # Analysis Parameters
    MIN_DOCUMENT_LENGTH = 100
    MAX_DOCUMENTS_PER_BATCH = 50
    DOMAIN_DETECTION_THRESHOLD = 0.75

    # Statistical Analysis
    STATISTICAL_WINDOW_SIZE = 100
    MOVING_AVERAGE_PERIOD = 10
    TREND_ANALYSIS_DAYS = 7


class KnowledgeExtractionConstants:
    """Constants specific to Knowledge Extraction Agent"""

    # Extraction Parameters
    MAX_ENTITIES_PER_DOCUMENT = 100
    MAX_RELATIONSHIPS_PER_DOCUMENT = 50
    ENTITY_CONFIDENCE_THRESHOLD = 0.8
    RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.7

    # Knowledge Graph Builder Parameters
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_MAX_ENTITIES = 50
    DEFAULT_MAX_RELATIONS = 30
    LLM_EXTRACTION_CONFIDENCE = 0.8
    MIN_ENTITY_TEXT_LENGTH = 2
    MIN_RELATION_TEXT_LENGTH = 2

    # Dynamic Configuration Fallback Values
    FALLBACK_ENTITY_CONFIDENCE_THRESHOLD = 0.85
    FALLBACK_RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.75
    FALLBACK_CHUNK_SIZE = 1000
    FALLBACK_CHUNK_OVERLAP = 200
    FALLBACK_BATCH_SIZE = 10
    FALLBACK_MAX_ENTITIES_PER_CHUNK = 20
    FALLBACK_MIN_RELATIONSHIP_STRENGTH = 0.6
    FALLBACK_QUALITY_VALIDATION_THRESHOLD = 0.8

    # Agent Performance Constants
    ENTITY_PRECISION_MULTIPLIER = 1.0
    ENTITY_RECALL_MULTIPLIER = 1.0
    RELATIONSHIP_PRECISION_MULTIPLIER = 1.0
    RELATIONSHIP_RECALL_MULTIPLIER = 1.0
    MAX_DOCUMENTS_DIVISOR = 1
    DEFAULT_MEMORY_USAGE_MB = 256
    DEFAULT_CPU_UTILIZATION_PERCENT = 50
    DEFAULT_CACHE_HIT_RATE = 0.6
    DISABLED_CACHE_HIT_RATE = 0.0

    # Validation
    MIN_ENTITY_LENGTH = 2
    MAX_ENTITY_LENGTH = 100
    VALIDATION_SAMPLE_SIZE = 10


class UniversalSearchConstants:
    """Constants specific to Universal Search Agent"""

    # Search Parameters
    MULTI_MODAL_WEIGHT_VECTOR = 0.4
    MULTI_MODAL_WEIGHT_GRAPH = 0.3
    MULTI_MODAL_WEIGHT_GNN = 0.3

    # Result Processing
    MAX_SEARCH_RESULTS = 20
    RESULT_RELEVANCE_THRESHOLD = 0.6
    CONTEXT_WINDOW_SIZE = 5

    # Tri-Modal Search Processing Delays (in seconds)
    VECTOR_PROCESSING_DELAY = 0.1
    GRAPH_PROCESSING_DELAY = 0.15
    GNN_PROCESSING_DELAY = 0.2

    # Search Configuration Defaults
    DEFAULT_VECTOR_TOP_K = 10
    DEFAULT_GRAPH_HOP_COUNT = 2
    DEFAULT_GNN_NODE_EMBEDDINGS = 128
    DEFAULT_MAX_RESULTS_PER_MODALITY = 10

    # Search Quality Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    VECTOR_SIMILARITY_THRESHOLD = 0.7
    GRAPH_MIN_RELATIONSHIP_STRENGTH = 0.5
    GNN_MIN_PREDICTION_CONFIDENCE = 0.6

    # Synthesis Weights
    CONFIDENCE_WEIGHT = 0.4
    AGREEMENT_WEIGHT = 0.3
    QUALITY_WEIGHT = 0.3

    # Graph Search Parameters
    DEFAULT_MAX_ENTITIES = 10
    DEFAULT_RELATIONSHIP_THRESHOLD = 0.6
    DEFAULT_MAX_DEPTH = 3

    # GNN Search Parameters
    DEFAULT_MAX_PREDICTIONS = 20
    DEFAULT_PATTERN_THRESHOLD = 0.7
    DEFAULT_MIN_TRAINING_EXAMPLES = 100

    # Dynamic Search Configuration Fallback Values
    FALLBACK_VECTOR_SIMILARITY_THRESHOLD = 0.75
    FALLBACK_VECTOR_TOP_K = 15
    FALLBACK_GRAPH_HOP_COUNT = 3
    FALLBACK_GRAPH_MIN_RELATIONSHIP_STRENGTH = 0.65
    FALLBACK_GNN_PREDICTION_CONFIDENCE = 0.7
    FALLBACK_GNN_NODE_EMBEDDINGS = 128
    FALLBACK_RESULT_SYNTHESIS_THRESHOLD = 0.8

    # Query Complexity Analysis
    QUERY_LENGTH_SIMPLE_THRESHOLD = 5
    QUERY_LENGTH_COMPLEX_THRESHOLD = 10
    QUERY_COMPLEXITY_SIMPLE_MULTIPLIER = 0.8
    QUERY_COMPLEXITY_MEDIUM_MULTIPLIER = 1.2
    QUERY_COMPLEXITY_COMPLEX_MULTIPLIER = 1.3
    MAX_GRAPH_HOP_COUNT = 5

    # Search Result Confidence Decrements
    VECTOR_CONFIDENCE_DECREMENT = 0.1
    VECTOR_SIMILARITY_DECREMENT = 0.05
    GRAPH_CONFIDENCE_DECREMENT = 0.08
    GNN_CONFIDENCE_DECREMENT = 0.06


# =============================================================================
# WORKFLOW AND ORCHESTRATION
# =============================================================================

class WorkflowConstants:
    """Workflow orchestration and state management constants"""

    # State Management
    MAX_WORKFLOW_STATES = 1000
    STATE_PERSISTENCE_INTERVAL = 60
    STATE_CLEANUP_DAYS = 30
    DEFAULT_STORAGE_DIR = "/tmp/workflow_states"
    WORKFLOW_FILE_PREFIX = "workflow_"
    WORKFLOW_FILE_EXTENSION = ".json"
    TEMP_FILE_EXTENSION = ".tmp"

    # Workflow Timeouts
    WORKFLOW_STEP_TIMEOUT = 300
    WORKFLOW_TOTAL_TIMEOUT = 1800
    WORKFLOW_IDLE_TIMEOUT = 600

    # Orchestration
    MAX_PARALLEL_WORKFLOWS = 5
    WORKFLOW_QUEUE_SIZE = 100
    PRIORITY_LEVELS = 3
    
    # Search Workflow Node Execution
    MAX_NODE_RETRIES = 3
    NODE_RETRY_BASE_DELAY = 2
    CONTEXT_WINDOW_SIZE = 50
    
    # Performance Grading Thresholds (seconds)
    EXCELLENT_PERFORMANCE_THRESHOLD = 1.0
    GOOD_PERFORMANCE_THRESHOLD = 2.0
    ACCEPTABLE_PERFORMANCE_THRESHOLD = 3.0
    
    # Query Complexity Analysis
    QUERY_COMPLEXITY_SIMPLE_WORD_THRESHOLD = 5
    QUERY_COMPLEXITY_MEDIUM_WORD_THRESHOLD = 10
    QUERY_COMPLEXITY_COMPLEX_WORD_THRESHOLD = 20
    QUERY_COMPLEXITY_EXPERT_WORD_THRESHOLD = 50
    
    # Search Strategy Defaults
    DEFAULT_SEARCH_MODALITIES = ["vector", "graph", "gnn"]
    DEFAULT_SEARCH_WEIGHTS = {"vector": 0.4, "graph": 0.3, "gnn": 0.3}
    OPTIMIZATION_STRATEGY_PARALLEL = "parallel_execution"
    DEFAULT_MAX_RESULTS = 10
    
    # Tri-Modal Search Execution
    SEARCH_CONFIDENCE_DEFAULT = 0.8
    SYNTHESIS_METHOD_WEIGHTED = "weighted_ranking"
    SYNTHESIS_CONFIDENCE_DEFAULT = 0.85
    RESPONSE_CONFIDENCE_DEFAULT = 0.9
    GENERATION_METHOD_TEMPLATE = "template_based"


# =============================================================================
# MODEL SELECTION AND PERFORMANCE
# =============================================================================

class ModelSelectionConstants:
    """Dynamic model selection and performance constants"""
    
    # Model Performance Tracking
    CACHE_EXPIRY_HOURS = 1
    PERFORMANCE_LEARNING_RATE = 0.1
    COST_EFFICIENCY_MIN_COST = 0.001  # Prevent division by zero
    
    # Bootstrap Models (used only during initialization)
    BOOTSTRAP_MODELS = {
        "gpt-4o": "gpt-4o-deployment",
        "gpt-4o-mini": "gpt-4o-mini-deployment",
        "gpt-35-turbo": "gpt-35-turbo-deployment"
    }
    
    # Query Complexity Temperature Adjustments
    SIMPLE_QUERY_TEMPERATURE = 0.1
    SIMPLE_QUERY_MAX_TOKENS = 500
    MODERATE_QUERY_TEMPERATURE = 0.3
    MODERATE_QUERY_MAX_TOKENS = 1000
    COMPLEX_QUERY_TEMPERATURE = 0.7
    COMPLEX_QUERY_MAX_TOKENS = 2000
    EXPERT_QUERY_TEMPERATURE = 0.9
    EXPERT_QUERY_MAX_TOKENS = 4000
    
    # Domain-Specific Model Selection
    PROGRAMMING_DOMAIN_TEMPERATURE = 0.3
    PROGRAMMING_DOMAIN_ACCURACY = 0.85
    PROGRAMMING_DOMAIN_RESPONSE_TIME = 2.5
    
    MEDICAL_LEGAL_DOMAIN_TEMPERATURE = 0.1
    MEDICAL_LEGAL_DOMAIN_ACCURACY = 0.90
    MEDICAL_LEGAL_DOMAIN_RESPONSE_TIME = 3.0
    
    COST_EFFICIENT_TEMPERATURE = 0.5
    COST_EFFICIENT_ACCURACY = 0.75
    COST_EFFICIENT_RESPONSE_TIME = 1.5
    
    BALANCED_TEMPERATURE = 0.7
    BALANCED_ACCURACY = 0.80
    BALANCED_RESPONSE_TIME = 2.0
    
    # Complexity Analysis Indicators
    EXPERT_QUERY_INDICATORS = [
        "analyze", "compare", "evaluate", "synthesize", "derive", "prove", 
        "algorithm", "architecture", "framework", "methodology"
    ]
    
    COMPLEX_QUERY_INDICATORS = [
        "explain", "describe", "how", "why", "relationship", "impact",
        "cause", "effect", "pattern", "trend"
    ]
    
    # Performance Data Paths
    MODEL_PERFORMANCE_DIR = "agents/domain_intelligence/generated_configs/model_performance"
    GENERAL_PERFORMANCE_FILE = "general_model_performance.json"
    DOMAIN_PERFORMANCE_SUFFIX = "_model_performance.json"
    
    # Estimated Default Values (for simulation)
    DEFAULT_COST_ESTIMATE = 0.02
    DEFAULT_COST_EFFICIENCY = 0.75
    TEMPERATURE_CAP = 1.0
    TEMPERATURE_MULTIPLIER_SIMPLE = 0.5
    TEMPERATURE_MULTIPLIER_COMPLEX = 1.2
    TEMPERATURE_MULTIPLIER_EXPERT = 1.5


# =============================================================================
# SHARED TOOLSET CONSTANTS
# =============================================================================

class SharedToolsetConstants:
    """Constants for shared toolsets and common operations"""
    
    # Memory Monitoring Thresholds
    MEMORY_WARNING_THRESHOLD = 80
    MEMORY_CRITICAL_THRESHOLD = 90
    
    # Health Status Classifications
    HEALTH_STATUS_HEALTHY = "healthy"
    HEALTH_STATUS_PARTIAL = "partial"
    HEALTH_STATUS_DEGRADED = "degraded"
    HEALTH_STATUS_ERROR = "error"
    HEALTH_STATUS_NOT_INITIALIZED = "not_initialized"
    HEALTH_STATUS_UNKNOWN = "unknown"
    
    # Access Levels
    ACCESS_LEVEL_VERIFIED = "verified"
    ACCESS_LEVEL_NONE = "none"
    
    # Performance Monitor Status
    MONITOR_STATUS_NOT_INITIALIZED = "not_initialized"
    MONITOR_STATUS_ERROR = "error"
    
    # Cache Optimization Results
    OPTIMIZATION_COMPLETED = True
    OPTIMIZATION_FAILED = False
    
    # Memory Status Classifications
    MEMORY_STATUS_HEALTHY = "healthy"
    MEMORY_STATUS_WARNING = "warning"
    MEMORY_STATUS_CRITICAL = "critical"
    MEMORY_STATUS_UNKNOWN = "unknown"
    MEMORY_STATUS_ERROR = "error"
    
    # System Resource Conversion
    BYTES_TO_GB_DIVISOR = 1024**3
    DECIMAL_PLACES_MEMORY = 2


# =============================================================================
# UNIVERSAL SEARCH TOOLSET CONSTANTS
# =============================================================================

class UniversalSearchToolsetConstants:
    """Constants specific to Universal Search toolset operations"""
    
    # Domain Detection Fallbacks
    FALLBACK_DOMAIN = "general"
    FALLBACK_CONFIDENCE_HIGH = 0.3
    FALLBACK_CONFIDENCE_ERROR = 0.1
    FALLBACK_REASONING_TEMPLATE = "Fallback domain detection [NEEDS DOMAIN CONFIG]"
    ERROR_REASONING_TEMPLATE = "Error in domain detection: {error} [NEEDS DOMAIN CONFIG]"
    
    # Mock Search Results (temporary - to be replaced with real results)
    MOCK_VECTOR_CONFIDENCE_1 = 0.95
    MOCK_VECTOR_CONFIDENCE_2 = 0.87
    MOCK_GRAPH_CONFIDENCE_1 = 0.92
    MOCK_GRAPH_CONFIDENCE_2 = 0.84
    MOCK_GNN_CONFIDENCE_1 = 0.89
    MOCK_GNN_CONFIDENCE_2 = 0.81
    MOCK_SYNTHESIS_SCORE = 0.91
    
    # Result Source Identifiers
    VECTOR_DB_SOURCE = "vector_db"
    GRAPH_DB_SOURCE = "graph_db"
    GNN_MODEL_SOURCE = "gnn_model"
    HARDCODED_PLACEHOLDER_CONFIG = "HARDCODED_PLACEHOLDER"
    
    # Search Configuration Placeholders
    PLACEHOLDER_SEARCH_TYPES = ["vector", "graph", "gnn"]
    PLACEHOLDER_MAX_RESULTS = 10
    
    # Mock Content Templates
    MOCK_VECTOR_RESULT_1 = "Vector search result 1 [NEEDS DOMAIN CONFIG]"
    MOCK_VECTOR_RESULT_2 = "Vector search result 2 [NEEDS DOMAIN CONFIG]"
    MOCK_GRAPH_RESULT_1 = "Graph relationship result 1 [NEEDS DOMAIN CONFIG]"
    MOCK_GRAPH_RESULT_2 = "Graph relationship result 2 [NEEDS DOMAIN CONFIG]"
    MOCK_GNN_RESULT_1 = "GNN prediction result 1 [NEEDS DOMAIN CONFIG]"
    MOCK_GNN_RESULT_2 = "GNN prediction result 2 [NEEDS DOMAIN CONFIG]"


# =============================================================================
# EXTRACTION PROCESSOR CONSTANTS
# =============================================================================

class ExtractionProcessorConstants:
    """Constants for unified extraction processor operations"""
    
    # Pattern Matching Limits
    MAX_TECHNICAL_VOCABULARY_TERMS = 20
    CAPS_MIN_LENGTH = 3
    CONTEXT_WINDOW_SMALL = 50
    
    # Confidence Calculation Factors
    LENGTH_NORMALIZATION_DIVISOR = 20.0
    EARLY_POSITION_FACTOR = 0.8
    LATE_POSITION_FACTOR = 0.6
    CONTEXT_BOOST_DEFAULT = 1.1
    
    # Entity Classification Thresholds
    MIN_ENTITY_LENGTH_FOR_BONUS = 3
    LONG_ENTITY_THRESHOLD = 10
    SINGLE_FREQUENCY = 1
    LOW_FREQUENCY_THRESHOLD = 3
    
    # Confidence Distribution Categories
    CONFIDENCE_VERY_HIGH_THRESHOLD = 0.9
    CONFIDENCE_HIGH_THRESHOLD = 0.8
    CONFIDENCE_MEDIUM_THRESHOLD = 0.6
    CONFIDENCE_LOW_THRESHOLD = 0.0
    
    # Relationship Confidence Factors
    BASE_SYNTACTIC_CONFIDENCE = 0.6
    BASE_SEMANTIC_CONFIDENCE = 0.7
    HIGH_SEMANTIC_CONFIDENCE = 0.8
    MIN_DISTANCE_FACTOR = 0.3
    MAX_DISTANCE_FACTOR = 1.0
    DISTANCE_DIVISOR = 100.0
    CONTEXT_WINDOW_RELATIONSHIP = 100
    CONTEXT_FACTOR_DEFAULT = 0.8
    CONTEXT_FACTOR_HIGH = 1.1
    
    # Semantic Analysis Parameters
    MAX_SENTENCE_LENGTH_DIVISOR = 50
    MIN_LENGTH_FACTOR = 0.5
    MAX_PROMINENCE_DIVISOR = 4
    
    # Confidence Enhancement
    CONFIDENCE_BOOST_FACTOR = 1.3
    MULTI_METHOD_CONFIDENCE_BOOST = 1.3
    MAX_CONFIDENCE_VALUE = 1.0
    
    # Weight Factors for Confidence Calculation
    SYNTACTIC_BASE_WEIGHT = 0.4
    SYNTACTIC_DISTANCE_WEIGHT = 0.3
    SYNTACTIC_CONTEXT_WEIGHT = 0.3
    SEMANTIC_BASE_WEIGHT = 0.5
    SEMANTIC_LENGTH_WEIGHT = 0.2
    SEMANTIC_PROMINENCE_WEIGHT = 0.3
    
    # Entity Pattern Weights
    LENGTH_WEIGHT = 0.3
    POSITION_WEIGHT = 0.2
    CONTEXT_WEIGHT = 0.3
    CASE_WEIGHT = 0.2
    
    # Case Factor Values
    CASE_FACTOR_DEFAULT = 0.8
    CASE_FACTOR_HIGH = 1.0
    
    # NLP Confidence Base Values
    BASE_NLP_CONFIDENCE = 0.5
    LENGTH_BONUS_SMALL = 0.1
    LENGTH_BONUS_LARGE = 0.2
    FREQUENCY_BONUS = 0.15
    FREQUENCY_BONUS_SMALL = 0.05
    
    # Graph Metrics
    GRAPH_DENSITY_DIVISOR = 2  # For undirected graph max edges calculation
    
    # Performance Statistics Method Names
    METHOD_PATTERN_BASED = "pattern_based"
    METHOD_NLP_BASED = "nlp_based"
    METHOD_HYBRID = "hybrid"
    METHOD_SEMANTIC = "semantic"
    METHOD_HYBRID_MULTI = "hybrid_multi_method"
    
    # Entity Type Classifications
    ENTITY_TYPE_IDENTIFIER = "identifier"
    ENTITY_TYPE_CONCEPT = "concept"
    ENTITY_TYPE_TECHNICAL_TERM = "technical_term"
    ENTITY_TYPE_CODE_ELEMENT = "code_element"
    ENTITY_TYPE_API_INTERFACE = "api_interface"
    ENTITY_TYPE_SYSTEM_COMPONENT = "system_component"
    
    # Relationship Types
    RELATION_TYPE_INTERACTS_WITH = "interacts_with"
    RELATION_TYPE_HAS_RELATIONSHIP = "has_relationship"
    RELATION_TYPE_ASSOCIATED_WITH = "associated_with"
    RELATION_TYPE_PART_OF = "part_of"
    RELATION_TYPE_CONTAINED_IN = "contained_in"
    RELATION_TYPE_CONNECTED_TO = "connected_to"
    RELATION_TYPE_DERIVED_FROM = "derived_from"
    RELATION_TYPE_RELATED_TO = "related_to"
    RELATION_TYPE_IMPLEMENTS = "implements"
    RELATION_TYPE_USES = "uses"
    RELATION_TYPE_CONTAINS = "contains"
    RELATION_TYPE_PROCESSES = "processes"
    RELATION_TYPE_CONNECTS_TO = "connects_to"
    RELATION_TYPE_DEPENDS_ON = "depends_on"
    RELATION_TYPE_CONFIGURES = "configures"
    RELATION_TYPE_MONITORS = "monitors"
    RELATION_TYPE_TRIGGERS = "triggers"
    RELATION_TYPE_VALIDATES = "validates"
    
    # Strong Relationship Indicators
    STRONG_INDICATORS = ["implements", "contains", "processes", "depends on"]
    
    # Relationship Support Words
    RELATIONSHIP_SUPPORT_WORDS = [
        "relationship", "connection", "interaction",
        "dependency", "association"
    ]
    
    # Coverage Calculation
    MAX_COVERAGE_PERCENTAGE = 100.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_constant_by_category(category: str) -> Dict[str, Any]:
    """
    Get all constants for a specific category

    Args:
        category: Category name (e.g., 'azure', 'ml', 'processing')

    Returns:
        Dictionary of constants for the category
    """
    category_mapping = {
        'azure': AzureServiceConstants,
        'ml': MLModelConstants,
        'processing': ProcessingConstants,
        'paths': PathConstants,
        'performance': PerformanceConstants,
        'cache': CacheConstants,
        'statistics': StatisticalConstants,
        'errors': ErrorConstants,
        'domain': DomainIntelligenceConstants,
        'extraction': KnowledgeExtractionConstants,
        'search': UniversalSearchConstants,
        'workflow': WorkflowConstants,
        'model_selection': ModelSelectionConstants,
        'shared_toolsets': SharedToolsetConstants,
        'search_toolsets': UniversalSearchToolsetConstants,
        'extraction_processor': ExtractionProcessorConstants
    }

    if category not in category_mapping:
        raise ValueError(f"Unknown category: {category}")

    const_class = category_mapping[category]
    return {
        attr: getattr(const_class, attr)
        for attr in dir(const_class)
        if not attr.startswith('_')
    }


def validate_constants() -> List[str]:
    """
    Validate that all constants are properly defined

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []

    # Validate Azure constants
    if not hasattr(AzureServiceConstants, 'OPENAI_API_VERSION'):
        errors.append("Missing OPENAI_API_VERSION")

    # Validate ML constants
    if MLModelConstants.EMBEDDING_DIMENSION <= 0:
        errors.append("Invalid EMBEDDING_DIMENSION")

    # Validate processing constants
    if ProcessingConstants.DEFAULT_TIMEOUT <= 0:
        errors.append("Invalid DEFAULT_TIMEOUT")

    return errors


# Export all constant classes for easy importing
__all__ = [
    'AzureServiceConstants',
    'MLModelConstants',
    'ProcessingConstants',
    'PathConstants',
    'PerformanceConstants',
    'CacheConstants',
    'StatisticalConstants',
    'ErrorConstants',
    'DomainIntelligenceConstants',
    'KnowledgeExtractionConstants',
    'UniversalSearchConstants',
    'WorkflowConstants',
    'ModelSelectionConstants',
    'SharedToolsetConstants',
    'UniversalSearchToolsetConstants',
    'ExtractionProcessorConstants',
    'get_constant_by_category',
    'validate_constants'
]
