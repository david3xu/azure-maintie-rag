"""
Performance-Adaptive Constants
==============================

This module contains constants that should adapt based on performance metrics and
system feedback. These have medium-high automation potential and should be continuously
optimized based on actual performance data.

Key Interdependent Groups:
1. Timeout and Retry Strategy - resilience behavior optimization
2. Batch Processing Optimization - throughput vs latency tradeoffs
3. Cache Performance Tuning - memory usage vs hit rates

AUTO-GENERATION POTENTIAL: HIGH
These should be continuously optimized based on actual performance data and
coordinated as interdependent groups.
"""

from .base import BaseScalingFactors, MathematicalConstants


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


# Export all constants
__all__ = [
    "PerformanceAdaptiveConstants",
    "SearchPerformanceAdaptiveConstants",
]
