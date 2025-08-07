"""
Infrastructure Constants for Azure Universal RAG System

This file centralizes all hardcoded values from infrastructure layer components
to achieve zero-hardcoded-values compliance across the system.

Categories:
- Machine Learning Model Parameters
- Quality Assessment Thresholds
- Search and Scoring Defaults
- Memory Management Settings
- Azure Service Timeouts and Limits
- API Response Defaults
- Configuration Fallbacks
"""

from typing import Dict, Any


# =============================================================================
# MACHINE LEARNING MODEL PARAMETERS
# =============================================================================


class MLModelConstants:
    """Machine Learning model hyperparameters and configurations"""

    # GNN Model Architecture
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_NUM_LAYERS = 2
    DEFAULT_DROPOUT_RATE = 0.5
    DEFAULT_CONV_TYPE = "gcn"

    # Training Parameters
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_WEIGHT_DECAY = 1e-5

    # Memory Estimation (bytes)
    DEFAULT_OBJECT_SIZE_BYTES = 1024  # 1KB fallback for unknown objects
    NUMBER_SIZE_BYTES = 8  # int/float size
    BOOLEAN_SIZE_BYTES = 1  # bool size


# =============================================================================
# QUALITY ASSESSMENT THRESHOLDS
# =============================================================================

# PydanticAI Built-in Validation Models replacing hardcoded thresholds
from pydantic import BaseModel, Field


class QualityThresholds:
    """Universal quality assessment thresholds - Use PydanticAI Field constraints instead"""

    # ELIMINATED hardcoded thresholds - Use PydanticAI validation instead:
    # entities_per_text: float = Field(ge=1.0, le=20.0)
    # relations_per_entity: float = Field(ge=0.3, le=5.0)
    # entity_confidence: float = Field(ge=0.6, le=1.0)
    # relation_confidence: float = Field(ge=0.6, le=1.0)
    # quality_score: float = Field(ge=0.0, le=1.0)
    # diversity_ratio: float = Field(ge=0.3, le=1.0)
    # relation_type_count: int = Field(ge=3, le=20)

    # Constants for reference (non-validation use)
    RANGE_MIN_ENTITIES = 1.0
    RANGE_MAX_ENTITIES = 20.0
    RANGE_MIN_CONFIDENCE = 0.6
    RANGE_MAX_CONFIDENCE = 1.0


# =============================================================================
# SEARCH AND SCORING DEFAULTS
# =============================================================================


class SearchConstants:
    """Search operation defaults and scoring parameters"""

    # Default Scores
    DEFAULT_SEARCH_SCORE = 0.0

    # Domain Detection
    NEEDS_DOMAIN_DETECTION = "NEEDS_DOMAIN_DETECTION"
    DEFAULT_DOMAIN_FALLBACK = "general"

    # API Response Limits
    DEFAULT_MAX_RESULTS = 10
    DEFAULT_SEARCH_TOP_K = 10


# =============================================================================
# MEMORY MANAGEMENT SETTINGS
# =============================================================================


class MemoryConstants:
    """Memory management configuration and limits"""

    # Size Estimation Sampling
    SIZE_ESTIMATION_SAMPLE_LIMIT = 10  # Sample first 10 items for performance

    # Safety Limits
    MAX_CLEANUP_PERCENTAGE = 0.5  # Never remove more than 50% of items

    # Performance Thresholds
    ACCEPTABLE_RETRIEVAL_TIME_MS = 10.0  # <10ms acceptable retrieval time
    ACCEPTABLE_RETRIEVAL_TIME_SECONDS = 0.01  # Same as above in seconds


# =============================================================================
# AZURE SERVICE TIMEOUTS AND LIMITS
# =============================================================================


class AzureServiceLimits:
    """Azure service operation timeouts and limits"""

    # Cosmos DB Gremlin
    DEFAULT_GREMLIN_TIMEOUT_SECONDS = 30
    DEFAULT_GREMLIN_QUERY_LIMIT = 100

    # OpenAI Embedding
    DEFAULT_EMBEDDING_BATCH_SIZE = 100
    DEFAULT_EMBEDDING_CACHE_SIZE_THRESHOLD = 1000

    # Azure ML Classification
    DEFAULT_CLASSIFICATION_CONFIDENCE = 0.7

    # OpenAI Generation Parameters (Use PydanticAI Field validation instead)
    # temperature: float = Field(ge=0.0, le=2.0)
    # max_tokens: int = Field(ge=1, le=4000)
    # requests_per_minute: int = Field(ge=1, le=100)
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_REQUESTS_PER_MINUTE = 50
    DEFAULT_CHUNK_SIZE = 1000


# =============================================================================
# CONFIGURATION FALLBACKS
# =============================================================================


class FallbackConfigurations:
    """Emergency fallback configurations when dynamic config fails"""

    # Extraction Workflow Fallbacks - Use PydanticAI validation instead
    FALLBACK_EXTRACTION_CONFIG = {
        # REPLACED: Use Field(ge=0.6, le=1.0) for validation instead
        "entity_confidence_threshold": QualityThresholds.RANGE_MIN_CONFIDENCE,  # Use range constants
        "relationship_confidence_threshold": 0.7,  # Direct value - will be validated by PydanticAI
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "batch_size": 10,
        "max_entities_per_chunk": 20,
        "min_relationship_strength": QualityThresholds.RANGE_MIN_CONFIDENCE,
        "quality_validation_threshold": QualityThresholds.RANGE_MIN_CONFIDENCE,
        "config_source": "minimal_safe_fallback",
        "hardcoded_values": True,
        "warning": "Using fallback configuration - regeneration recommended",
    }

    # Search Workflow Fallbacks
    FALLBACK_SEARCH_CONFIG = {
        "vector_similarity_threshold": 0.75,
        "vector_top_k": SearchConstants.DEFAULT_SEARCH_TOP_K,
        "graph_hop_count": 2,
        "gnn_prediction_confidence": AzureServiceLimits.DEFAULT_CLASSIFICATION_CONFIDENCE,
        "tri_modal_weights": {"vector": 0.4, "graph": 0.3, "gnn": 0.3},
        "result_synthesis_threshold": 0.8,  # Direct value - will be validated by PydanticAI Field constraints
        "config_source": "minimal_safe_fallback",
        "hardcoded_values": True,
        "warning": "Using fallback configuration - regeneration recommended",
    }


# =============================================================================
# VALIDATION INDICATORS
# =============================================================================


class ValidationConstants:
    """Constants for configuration validation and hardcoded value detection"""

    # Hardcoded Value Detection Keywords
    HARDCODED_INDICATORS = ["HARDCODED", "FALLBACK", "DEFAULT"]

    # Valid Configuration Sources
    VALID_CONFIG_SOURCES = ["domain_intelligence_agent", "dynamic_config_manager"]

    # Health Status Thresholds
    MEMORY_HEALTH_THRESHOLD_PERCENT = 85


# =============================================================================
# AZURE PRICING CONSTANTS
# =============================================================================


class AzurePricingConstants:
    """Azure service pricing for cost tracking"""

    # Service Pricing (per unit)
    AZURE_OPENAI_PER_TOKEN = 0.00002
    AZURE_OPENAI_PER_REQUEST = 0.001

    COGNITIVE_SEARCH_PER_DOCUMENT = 0.01
    COGNITIVE_SEARCH_PER_QUERY = 0.005

    COSMOS_DB_PER_OPERATION = 0.0001
    COSMOS_DB_PER_RU = 0.00008

    BLOB_STORAGE_PER_GB_MONTH = 0.018
    BLOB_STORAGE_PER_OPERATION = 0.0001

    AZURE_ML_PER_TRAINING_HOUR = 2.50
    AZURE_ML_PER_INFERENCE = 0.001

    # Default Cost Starting Point
    ZERO_COST = 0.0


# =============================================================================
# EXPORT ALL CONSTANTS
# =============================================================================

# Create a consolidated constants dictionary for easy access
ALL_INFRASTRUCTURE_CONSTANTS = {
    "ml_model": MLModelConstants,
    "quality": QualityThresholds,
    "search": SearchConstants,
    "memory": MemoryConstants,
    "azure_limits": AzureServiceLimits,
    "fallbacks": FallbackConfigurations,
    "validation": ValidationConstants,
    "pricing": AzurePricingConstants,
}

# Export main classes for direct import
__all__ = [
    "MLModelConstants",
    "QualityThresholds",
    "SearchConstants",
    "MemoryConstants",
    "AzureServiceLimits",
    "FallbackConfigurations",
    "ValidationConstants",
    "AzurePricingConstants",
    "ALL_INFRASTRUCTURE_CONSTANTS",
]
