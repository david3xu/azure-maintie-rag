"""
Agent Base Architecture Constants

This module centralizes all hardcoded values from the Agent Base Architecture
to ensure maintainability and provide clear documentation for all thresholds,
timeouts, limits, and configuration values.

All values are based on performance analysis and empirical testing,
and can be overridden via configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any
import time


@dataclass(frozen=True)
class ReasoningEngineConstants:
    """Constants for the reasoning engine component."""
    
    # Timeout and performance limits
    DEFAULT_TIMEOUT_SECONDS: float = 30.0
    DEFAULT_RETRY_ATTEMPTS: int = 2
    MAX_TOTAL_TIME: float = 60.0
    
    # Confidence thresholds
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    LOW_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Processing limits
    MAX_PARALLEL_STEPS: int = 5
    MAX_REASONING_CYCLES: int = 10
    
    # Text analysis thresholds
    CONCEPT_MIN_WORD_LENGTH: int = 3
    MIN_CONCEPTS_FOR_ANALYSIS: int = 2


@dataclass(frozen=True)
class MemoryManagerConstants:
    """Constants for the memory management system."""
    
    # Default values
    DEFAULT_ACCESS_COUNT: int = 0
    DEFAULT_CONFIDENCE: float = 1.0
    DEFAULT_SOURCE: str = "agent"
    DEFAULT_COHERENCE_SCORE: float = 0.0
    
    # Search and retrieval limits
    DEFAULT_SEARCH_LIMIT: int = 10
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    MAX_SEARCH_RESULTS: int = 50
    
    # Memory consolidation thresholds
    CONSOLIDATION_TAG_LIMIT: int = 3  # Use first 3 tags as theme
    MIN_MEMORIES_FOR_CONSOLIDATION: int = 5
    
    # Statistics and analysis
    MIN_MEMORIES_FOR_STATS: int = 1
    CONFIDENCE_ANALYSIS_MIN_SAMPLES: int = 3
    
    # Performance limits
    MAX_MEMORIES_PER_TYPE: int = 1000
    BATCH_PROCESSING_SIZE: int = 100


@dataclass(frozen=True)
class ContextManagerConstants:
    """Constants for the context management system."""
    
    # Default values
    DEFAULT_ACCESS_COUNT: int = 0
    DEFAULT_CONFIDENCE: float = 0.0
    DEFAULT_PRIORITY: float = 1.0
    
    # Performance tracking
    INITIAL_RESPONSE_TIME: float = 0.0
    INITIAL_COUNTER: int = 0
    INITIAL_DISTRIBUTION: int = 0
    
    # Cache and cleanup
    CACHE_HIT_INCREMENT: int = 1
    CACHE_MISS_INCREMENT: int = 1
    CLEANUP_INCREMENT: int = 1
    
    # Context limits
    MAX_CONTEXTS_PER_SESSION: int = 100
    MAX_CACHE_SIZE: int = 500
    CONTEXT_CLEANUP_BATCH_SIZE: int = 50


@dataclass(frozen=True)
class ReactEngineConstants:
    """Constants for the ReAct (Reason-Act-Observe) engine."""
    
    # Cycle and reasoning limits
    DEFAULT_MAX_CYCLES: int = 5
    DEFAULT_CURRENT_CYCLE: int = 0
    MAX_REASONING_CYCLES: int = 10
    
    # Confidence and performance thresholds
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    DEFAULT_TIMEOUT_SECONDS: float = 30.0
    DEFAULT_RETRY_ATTEMPTS: int = 2
    INITIAL_ACCUMULATED_CONFIDENCE: float = 0.0
    
    # Observation and decision thresholds
    MIN_OBSERVATIONS_FOR_SYNTHESIS: int = 3
    HIGH_CONFIDENCE_THRESHOLD: float = 0.6
    CONFIDENCE_WEIGHT_THRESHOLD: float = 0.0
    
    # Cycle-specific logic
    INITIAL_CYCLE: int = 1
    NO_OBSERVATIONS: int = 0
    
    # Action planning
    MAX_PLANNED_ACTIONS: int = 10
    ACTION_TIMEOUT_SECONDS: float = 15.0


@dataclass(frozen=True)
class TemporalPatternTrackerConstants:
    """Constants for temporal pattern tracking and analysis."""
    
    # Default values
    DEFAULT_CONFIDENCE: float = 1.0
    DEFAULT_SOURCE: str = "system"
    INITIAL_PATTERN_ANALYSIS_TIME: float = 0.0
    
    # Time windows and decay
    DEFAULT_DECAY_HOURS: float = 24.0
    RECENT_EVENTS_WINDOW_SECONDS: float = 3600.0  # 1 hour
    CHANGE_MAGNITUDE_THRESHOLD: float = 0.1
    
    # Pattern analysis thresholds
    RECENT_CONFIDENCE_SAMPLE_SIZE: int = 5
    MIN_EVENTS_FOR_PREDICTION: int = 3
    DEFAULT_PREDICTION_HORIZON_HOURS: float = 24.0
    
    # Frequency and correlation analysis
    DAY_BUCKET_SIZE_SECONDS: int = 3600  # 1 hour buckets
    MIN_FREQUENCY_THRESHOLD: int = 5
    MIN_SEQUENCE_LENGTH: int = 3
    
    # Correlation detection
    CORRELATION_TIME_WINDOW_SECONDS: int = 300  # 5 minutes
    MIN_CORRELATION_THRESHOLD: int = 2
    
    # Trend analysis
    MIN_CONFIDENCE_HISTORY_SIZE: int = 5
    MIN_TREND_SAMPLES: int = 3
    TREND_THRESHOLD: float = 0.0
    
    # Stability analysis
    MIN_STABLE_ENTITIES: int = 3
    STABILITY_THRESHOLD: float = 0.05


@dataclass(frozen=True)
class PlanExecuteEngineConstants:
    """Constants for the Plan-Execute engine."""
    
    # Planning limits
    MAX_PLAN_STEPS: int = 10
    MAX_PLANNING_TIME_SECONDS: float = 30.0
    DEFAULT_STEP_TIMEOUT_SECONDS: float = 15.0
    
    # Execution parameters
    MAX_PARALLEL_STEPS: int = 5
    MAX_RETRIES_PER_STEP: int = 3
    STEP_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Progress tracking
    INITIAL_PROGRESS: float = 0.0
    PROGRESS_COMPLETE: float = 1.0
    PROGRESS_INCREMENT: float = 0.1
    
    # Quality thresholds
    MIN_PLAN_QUALITY_SCORE: float = 0.7
    MAX_PLANNING_ITERATIONS: int = 3


@dataclass(frozen=True)
class PerformanceConstants:
    """Performance-related constants across all components."""
    
    # Response time targets (seconds)
    TARGET_REASONING_TIME: float = 5.0
    TARGET_MEMORY_RETRIEVAL_TIME: float = 1.0
    TARGET_CONTEXT_ENHANCEMENT_TIME: float = 2.0
    MAX_ACCEPTABLE_RESPONSE_TIME: float = 10.0
    
    # Concurrency limits
    MAX_CONCURRENT_OPERATIONS: int = 10
    BATCH_PROCESSING_SIZE: int = 50
    
    # Memory and resource limits
    MAX_MEMORY_USAGE_MB: int = 512
    MAX_CACHE_ENTRIES: int = 1000
    
    # Monitoring intervals
    METRICS_COLLECTION_INTERVAL_SECONDS: int = 60
    HEALTH_CHECK_INTERVAL_SECONDS: int = 30


@dataclass(frozen=True)
class QualityThresholds:
    """Quality assurance thresholds and validation criteria."""
    
    # Confidence quality levels
    VERY_HIGH_CONFIDENCE: float = 0.9
    HIGH_CONFIDENCE: float = 0.7
    MEDIUM_CONFIDENCE: float = 0.5
    LOW_CONFIDENCE: float = 0.3
    
    # Success rate thresholds
    EXCELLENT_SUCCESS_RATE: float = 0.95
    GOOD_SUCCESS_RATE: float = 0.85
    ACCEPTABLE_SUCCESS_RATE: float = 0.75
    POOR_SUCCESS_RATE: float = 0.6
    
    # Data quality thresholds
    MIN_DATA_COMPLETENESS: float = 0.8
    MIN_DATA_ACCURACY: float = 0.9
    MAX_ERROR_RATE: float = 0.05
    
    # Performance quality
    MAX_ACCEPTABLE_LATENCY_MS: float = 3000.0
    MIN_THROUGHPUT_OPS_PER_SECOND: float = 10.0
    MAX_MEMORY_GROWTH_RATE: float = 0.1


@dataclass(frozen=True)
class RetryAndBackoffConstants:
    """Constants for retry logic and backoff strategies."""
    
    # Basic retry parameters
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_BACKOFF_FACTOR: float = 2.0
    DEFAULT_INITIAL_DELAY_SECONDS: float = 1.0
    MAX_DELAY_SECONDS: float = 60.0
    
    # Exponential backoff
    EXPONENTIAL_BASE: float = 2.0
    JITTER_FACTOR: float = 0.1
    
    # Circuit breaker thresholds
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: float = 60.0
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = 30.0


@dataclass(frozen=True)
class ValidationConstants:
    """Constants for validation and testing."""
    
    # Test data limits
    MAX_TEST_ITERATIONS: int = 100
    TEST_TIMEOUT_SECONDS: float = 30.0
    VALIDATION_SAMPLE_SIZE: int = 50
    
    # Assertion thresholds
    FLOAT_COMPARISON_TOLERANCE: float = 1e-6
    PERCENTAGE_TOLERANCE: float = 0.05
    
    # Performance benchmarks
    BENCHMARK_ITERATIONS: int = 10
    BENCHMARK_WARMUP_ITERATIONS: int = 3
    MAX_BENCHMARK_TIME_SECONDS: float = 60.0


# Singleton instances for easy import and use
REASONING = ReasoningEngineConstants()
MEMORY = MemoryManagerConstants()
CONTEXT = ContextManagerConstants()
REACT = ReactEngineConstants()
TEMPORAL = TemporalPatternTrackerConstants()
PLAN_EXECUTE = PlanExecuteEngineConstants()
PERFORMANCE = PerformanceConstants()
QUALITY = QualityThresholds()
RETRY = RetryAndBackoffConstants()
VALIDATION = ValidationConstants()


def get_agent_base_config() -> Dict[str, Any]:
    """Get default configuration dictionary for agent base components.
    
    Returns:
        Complete configuration dictionary with all default values
    """
    return {
        # Reasoning Engine
        "reasoning_timeout_seconds": REASONING.DEFAULT_TIMEOUT_SECONDS,
        "reasoning_retry_attempts": REASONING.DEFAULT_RETRY_ATTEMPTS,
        "reasoning_confidence_threshold": REASONING.DEFAULT_CONFIDENCE_THRESHOLD,
        "max_reasoning_cycles": REASONING.MAX_REASONING_CYCLES,
        
        # Memory Manager
        "memory_search_limit": MEMORY.DEFAULT_SEARCH_LIMIT,
        "memory_similarity_threshold": MEMORY.DEFAULT_SIMILARITY_THRESHOLD,
        "max_memories_per_type": MEMORY.MAX_MEMORIES_PER_TYPE,
        
        # Context Manager
        "max_contexts_per_session": CONTEXT.MAX_CONTEXTS_PER_SESSION,
        "max_cache_size": CONTEXT.MAX_CACHE_SIZE,
        "context_cleanup_batch_size": CONTEXT.CONTEXT_CLEANUP_BATCH_SIZE,
        
        # ReAct Engine
        "react_max_cycles": REACT.DEFAULT_MAX_CYCLES,
        "react_confidence_threshold": REACT.DEFAULT_CONFIDENCE_THRESHOLD,
        "react_timeout_seconds": REACT.DEFAULT_TIMEOUT_SECONDS,
        
        # Temporal Pattern Tracker
        "temporal_decay_hours": TEMPORAL.DEFAULT_DECAY_HOURS,
        "temporal_prediction_horizon_hours": TEMPORAL.DEFAULT_PREDICTION_HORIZON_HOURS,
        "min_frequency_threshold": TEMPORAL.MIN_FREQUENCY_THRESHOLD,
        
        # Plan-Execute Engine
        "plan_max_steps": PLAN_EXECUTE.MAX_PLAN_STEPS,
        "plan_timeout_seconds": PLAN_EXECUTE.MAX_PLANNING_TIME_SECONDS,
        "step_confidence_threshold": PLAN_EXECUTE.STEP_CONFIDENCE_THRESHOLD,
        
        # Performance
        "target_response_time": PERFORMANCE.TARGET_REASONING_TIME,
        "max_concurrent_operations": PERFORMANCE.MAX_CONCURRENT_OPERATIONS,
        "batch_processing_size": PERFORMANCE.BATCH_PROCESSING_SIZE,
        
        # Quality
        "min_success_rate": QUALITY.ACCEPTABLE_SUCCESS_RATE,
        "max_error_rate": QUALITY.MAX_ERROR_RATE,
        "max_latency_ms": QUALITY.MAX_ACCEPTABLE_LATENCY_MS,
        
        # Retry and Backoff
        "max_retries": RETRY.DEFAULT_MAX_RETRIES,
        "backoff_factor": RETRY.DEFAULT_BACKOFF_FACTOR,
        "initial_delay_seconds": RETRY.DEFAULT_INITIAL_DELAY_SECONDS
    }


def get_performance_targets() -> Dict[str, float]:
    """Get performance targets for monitoring and validation.
    
    Returns:
        Dictionary of performance targets and thresholds
    """
    return {
        "reasoning_time_seconds": PERFORMANCE.TARGET_REASONING_TIME,
        "memory_retrieval_time_seconds": PERFORMANCE.TARGET_MEMORY_RETRIEVAL_TIME,
        "context_enhancement_time_seconds": PERFORMANCE.TARGET_CONTEXT_ENHANCEMENT_TIME,
        "max_response_time_seconds": PERFORMANCE.MAX_ACCEPTABLE_RESPONSE_TIME,
        "min_success_rate": QUALITY.ACCEPTABLE_SUCCESS_RATE,
        "max_error_rate": QUALITY.MAX_ERROR_RATE,
        "max_latency_ms": QUALITY.MAX_ACCEPTABLE_LATENCY_MS,
        "min_throughput_ops_per_second": QUALITY.MIN_THROUGHPUT_OPS_PER_SECOND
    }


def get_quality_thresholds() -> Dict[str, float]:
    """Get quality thresholds for validation and assessment.
    
    Returns:
        Dictionary of quality thresholds and criteria
    """
    return {
        "very_high_confidence": QUALITY.VERY_HIGH_CONFIDENCE,
        "high_confidence": QUALITY.HIGH_CONFIDENCE,
        "medium_confidence": QUALITY.MEDIUM_CONFIDENCE,
        "low_confidence": QUALITY.LOW_CONFIDENCE,
        "excellent_success_rate": QUALITY.EXCELLENT_SUCCESS_RATE,
        "good_success_rate": QUALITY.GOOD_SUCCESS_RATE,
        "acceptable_success_rate": QUALITY.ACCEPTABLE_SUCCESS_RATE,
        "min_data_completeness": QUALITY.MIN_DATA_COMPLETENESS,
        "min_data_accuracy": QUALITY.MIN_DATA_ACCURACY
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values against acceptable ranges.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    validations = [
        # Timeout validations
        0 < config.get("reasoning_timeout_seconds", REASONING.DEFAULT_TIMEOUT_SECONDS) <= 300,
        0 < config.get("react_timeout_seconds", REACT.DEFAULT_TIMEOUT_SECONDS) <= 300,
        
        # Confidence validations
        0 <= config.get("reasoning_confidence_threshold", REASONING.DEFAULT_CONFIDENCE_THRESHOLD) <= 1,
        0 <= config.get("react_confidence_threshold", REACT.DEFAULT_CONFIDENCE_THRESHOLD) <= 1,
        
        # Limit validations
        1 <= config.get("max_reasoning_cycles", REASONING.MAX_REASONING_CYCLES) <= 50,
        1 <= config.get("react_max_cycles", REACT.DEFAULT_MAX_CYCLES) <= 20,
        1 <= config.get("memory_search_limit", MEMORY.DEFAULT_SEARCH_LIMIT) <= 1000,
        
        # Performance validations
        1 <= config.get("max_concurrent_operations", PERFORMANCE.MAX_CONCURRENT_OPERATIONS) <= 100,
        10 <= config.get("batch_processing_size", PERFORMANCE.BATCH_PROCESSING_SIZE) <= 1000,
    ]
    
    return all(validations)


# Export all constants for easy access
__all__ = [
    # Constants classes
    'REASONING',
    'MEMORY', 
    'CONTEXT',
    'REACT',
    'TEMPORAL',
    'PLAN_EXECUTE',
    'PERFORMANCE',
    'QUALITY',
    'RETRY',
    'VALIDATION',
    
    # Configuration functions
    'get_agent_base_config',
    'get_performance_targets',
    'get_quality_thresholds',
    'validate_config'
]