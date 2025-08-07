"""
Cache and Performance Models
===========================

Data models for caching, performance monitoring, and system health tracking.
These models support the performance optimization and monitoring systems
throughout the multi-agent architecture.

This module provides:
- Cache entry and metrics models
- Performance feedback and monitoring
- Memory management and utilization tracking
- Service health monitoring
- System performance analytics
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import ErrorCategory, HealthStatus

# =============================================================================
# CACHE MODELS
# =============================================================================


@dataclass
class CacheEntry:
    """Unified cache entry with comprehensive metadata"""

    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: int
    cache_type: str = "general"

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access statistics for LRU and performance tracking"""
        self.accessed_at = time.time()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at

    def get_time_to_expiry(self) -> float:
        """Get time until expiry in seconds"""
        age = self.get_age_seconds()
        return max(0.0, self.ttl - age)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding data)"""
        return {
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "cache_type": self.cache_type,
            "age_seconds": self.get_age_seconds(),
            "time_to_expiry": self.get_time_to_expiry(),
            "is_expired": self.is_expired(),
        }


@dataclass
class CacheMetrics:
    """Cache performance metrics"""

    hit_rate: float
    miss_rate: float
    total_requests: int
    cache_size: int
    memory_usage_mb: float
    eviction_count: int
    last_updated: datetime

    def get_hit_rate_percentage(self) -> float:
        """Get hit rate as percentage"""
        return self.hit_rate * 100.0

    def get_efficiency_score(self) -> float:
        """Calculate cache efficiency score"""
        # Combine hit rate with low eviction rate
        hit_factor = self.hit_rate
        eviction_factor = max(
            0.0, 1.0 - (self.eviction_count / max(1, self.total_requests))
        )
        return (hit_factor + eviction_factor) / 2.0

    def is_performing_well(self, min_hit_rate: float = 0.6) -> bool:
        """Check if cache is performing within acceptable parameters"""
        return self.hit_rate >= min_hit_rate and self.memory_usage_mb < 1000.0


@dataclass
class CachePerformanceMetrics:
    """Comprehensive performance metrics for monitoring"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fast_lookups: int = 0
    pattern_index_hits: int = 0
    query_cache_hits: int = 0
    domain_signature_hits: int = 0
    evictions: int = 0
    average_lookup_time: float = 0.0

    @property
    def hit_rate_percent(self) -> float:
        """Calculate cache hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100.0

    @property
    def fast_lookup_percent(self) -> float:
        """Calculate percentage of fast lookups"""
        if self.total_requests == 0:
            return 0.0
        return (self.fast_lookups / self.total_requests) * 100.0

    def update_metrics(self, is_hit: bool, lookup_time: float, is_fast: bool = False):
        """Update metrics with new request data"""
        self.total_requests += 1

        if is_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if is_fast:
            self.fast_lookups += 1

        # Update average lookup time
        if self.total_requests == 1:
            self.average_lookup_time = lookup_time
        else:
            self.average_lookup_time = (
                self.average_lookup_time * (self.total_requests - 1) + lookup_time
            ) / self.total_requests

    def reset_metrics(self):
        """Reset all metrics to zero"""
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fast_lookups = 0
        self.pattern_index_hits = 0
        self.query_cache_hits = 0
        self.domain_signature_hits = 0
        self.evictions = 0
        self.average_lookup_time = 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "total_requests": self.total_requests,
            "hit_rate_percent": self.hit_rate_percent,
            "fast_lookup_percent": self.fast_lookup_percent,
            "average_lookup_time_ms": self.average_lookup_time * 1000,
            "cache_efficiency": self.hit_rate_percent / 100.0,
            "eviction_rate": self.evictions / max(1, self.total_requests),
            "specialization": {
                "pattern_index_hits": self.pattern_index_hits,
                "query_cache_hits": self.query_cache_hits,
                "domain_signature_hits": self.domain_signature_hits,
            },
        }


# =============================================================================
# MEMORY MANAGEMENT MODELS
# =============================================================================


@dataclass
class MemoryStatus:
    """Memory management status and metrics"""

    total_items: int = 0
    memory_limit_mb: float = 200.0
    estimated_usage_mb: float = 0.0
    evictions: int = 0
    last_cleanup: float = 0.0
    health_status: str = "healthy"

    @property
    def utilization_percent(self) -> float:
        """Calculate memory utilization percentage"""
        if self.memory_limit_mb == 0:
            return 0.0
        return (self.estimated_usage_mb / self.memory_limit_mb) * 100.0

    def update_health_status(self):
        """Update health status based on utilization"""
        utilization = self.utilization_percent
        if utilization > 90:
            self.health_status = "critical"
        elif utilization > 75:
            self.health_status = "warning"
        elif utilization > 50:
            self.health_status = "moderate"
        else:
            self.health_status = "healthy"

    def needs_cleanup(self, threshold_percent: float = 80.0) -> bool:
        """Check if memory cleanup is needed"""
        return self.utilization_percent > threshold_percent

    def estimate_item_size_mb(self) -> float:
        """Estimate average item size in MB"""
        if self.total_items == 0:
            return 0.0
        return self.estimated_usage_mb / self.total_items

    def can_add_items(self, num_items: int) -> bool:
        """Check if we can add more items without exceeding limit"""
        if self.total_items == 0:
            return True  # Can't estimate without existing items

        avg_item_size = self.estimate_item_size_mb()
        projected_usage = self.estimated_usage_mb + (num_items * avg_item_size)
        return projected_usage <= self.memory_limit_mb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_items": self.total_items,
            "memory_limit_mb": self.memory_limit_mb,
            "estimated_usage_mb": self.estimated_usage_mb,
            "utilization_percent": self.utilization_percent,
            "evictions": self.evictions,
            "last_cleanup": self.last_cleanup,
            "health_status": self.health_status,
            "needs_cleanup": self.needs_cleanup(),
            "avg_item_size_mb": self.estimate_item_size_mb(),
        }


# =============================================================================
# SERVICE HEALTH MODELS
# =============================================================================


@dataclass
class ServiceHealth:
    """Service health status"""

    service_name: str
    status: HealthStatus
    response_time_ms: float
    error_rate: float
    last_check: datetime
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.PARTIAL]

    def is_degraded(self) -> bool:
        """Check if service is degraded"""
        return self.status == HealthStatus.DEGRADED

    def needs_attention(self) -> bool:
        """Check if service needs attention"""
        return self.status in [HealthStatus.DEGRADED, HealthStatus.ERROR]

    def get_health_score(self) -> float:
        """Get numerical health score (0-1)"""
        status_scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.PARTIAL: 0.8,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.ERROR: 0.2,
            HealthStatus.NOT_INITIALIZED: 0.0,
            HealthStatus.UNKNOWN: 0.1,
        }
        return status_scores.get(self.status, 0.0)

    def update_status(
        self, new_status: HealthStatus, response_time: float, error_rate: float
    ):
        """Update service health status"""
        self.status = new_status
        self.response_time_ms = response_time
        self.error_rate = error_rate
        self.last_check = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "service_name": self.service_name,
            "status": (
                self.status.value if hasattr(self.status, "value") else str(self.status)
            ),
            "response_time_ms": self.response_time_ms,
            "error_rate": self.error_rate,
            "last_check": self.last_check.isoformat(),
            "health_score": self.get_health_score(),
            "is_healthy": self.is_healthy(),
            "needs_attention": self.needs_attention(),
            "details": self.details,
        }


# =============================================================================
# PERFORMANCE MONITORING MODELS
# =============================================================================


class PerformanceFeedbackPoint(BaseModel):
    """Individual performance feedback data point for learning and optimization"""

    # Context identification
    agent_type: str = Field(description="Agent that generated this performance data")
    domain_name: str = Field(description="Domain context for the operation")
    operation_type: str = Field(
        description="Type of operation (extraction, search, analysis)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this performance was recorded"
    )

    # Configuration used
    configuration_used: Dict[str, Any] = Field(
        description="Configuration parameters that were used"
    )
    configuration_source: str = Field(
        description="Source of the configuration (dynamic, static, fallback)"
    )

    # Performance metrics
    execution_time_seconds: float = Field(ge=0.0, description="Total execution time")
    success: bool = Field(description="Whether the operation succeeded")
    quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Quality assessment score"
    )

    # Input characteristics
    input_size: Optional[int] = Field(
        default=None, ge=0, description="Size of input data"
    )
    input_complexity: Optional[str] = Field(
        default=None, description="Assessed complexity of input"
    )

    # Output characteristics
    output_size: Optional[int] = Field(
        default=None, ge=0, description="Size of output data"
    )
    output_quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Detailed quality metrics"
    )

    # Resource utilization
    memory_usage_mb: Optional[float] = Field(
        default=None, ge=0.0, description="Memory usage during operation"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="CPU usage during operation"
    )

    # Error information (if applicable)
    error_message: Optional[str] = Field(
        default=None, description="Error message if operation failed"
    )
    error_category: Optional[ErrorCategory] = Field(
        default=None, description="Error category classification"
    )

    # Context for learning
    user_satisfaction: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="User satisfaction score"
    )
    downstream_impact: Optional[Dict[str, float]] = Field(
        default=None, description="Impact on downstream operations"
    )

    def calculate_performance_score(self) -> float:
        """Calculate overall performance score"""
        # Base score from execution time (faster is better)
        time_score = min(1.0, 10.0 / max(0.1, self.execution_time_seconds))

        # Success factor
        success_factor = 1.0 if self.success else 0.3

        # Quality factor
        quality_factor = self.quality_score if self.quality_score is not None else 0.7

        # Resource efficiency
        memory_factor = 1.0
        if self.memory_usage_mb is not None:
            # Penalize high memory usage
            memory_factor = max(0.1, 1.0 - (self.memory_usage_mb / 1000.0))

        # Weighted combination
        return (
            time_score * 0.3
            + success_factor * 0.4
            + quality_factor * 0.2
            + memory_factor * 0.1
        )

    def should_trigger_optimization(self, threshold: float = 0.6) -> bool:
        """Check if this feedback should trigger optimization"""
        return self.calculate_performance_score() < threshold

    def get_optimization_targets(self) -> List[str]:
        """Get list of optimization targets based on feedback"""
        targets = []

        if self.execution_time_seconds > 5.0:
            targets.append("execution_time")

        if not self.success:
            targets.append("error_handling")

        if self.quality_score and self.quality_score < 0.7:
            targets.append("quality_improvement")

        if self.memory_usage_mb and self.memory_usage_mb > 500.0:
            targets.append("memory_optimization")

        return targets


class SystemPerformanceSnapshot(BaseModel):
    """System-wide performance snapshot"""

    timestamp: datetime = Field(default_factory=datetime.now)

    # Cache performance
    cache_metrics: CachePerformanceMetrics = Field(
        description="Cache performance metrics"
    )
    memory_status: MemoryStatus = Field(description="Memory utilization status")

    # Service health
    service_health_summary: Dict[str, ServiceHealth] = Field(
        default_factory=dict, description="Health status of all services"
    )

    # Performance indicators
    average_response_time_ms: float = Field(ge=0.0, description="Average response time")
    request_throughput: float = Field(ge=0.0, description="Requests per second")
    error_rate_percent: float = Field(
        ge=0.0, le=100.0, description="Error rate percentage"
    )

    # Resource utilization
    cpu_utilization_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="CPU utilization"
    )
    memory_utilization_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Memory utilization"
    )

    # Quality metrics
    overall_quality_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Overall system quality"
    )
    user_satisfaction_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="User satisfaction"
    )

    def calculate_system_health_score(self) -> float:
        """Calculate overall system health score"""
        # Weight different factors
        performance_score = min(1.0, 5000.0 / max(100.0, self.average_response_time_ms))
        error_score = max(0.0, 1.0 - (self.error_rate_percent / 100.0))
        resource_score = max(0.0, 1.0 - (self.cpu_utilization_percent / 100.0))
        cache_score = self.cache_metrics.hit_rate_percent / 100.0

        # Weighted average
        return (
            performance_score * 0.3
            + error_score * 0.25
            + resource_score * 0.2
            + cache_score * 0.15
            + self.overall_quality_score * 0.1
        )

    def get_performance_grade(self) -> str:
        """Get performance grade (A-F)"""
        score = self.calculate_system_health_score()
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []

        if self.average_response_time_ms > 3000:
            bottlenecks.append("high_response_time")

        if self.error_rate_percent > 5.0:
            bottlenecks.append("high_error_rate")

        if self.cpu_utilization_percent > 80:
            bottlenecks.append("high_cpu_usage")

        if self.memory_utilization_percent > 80:
            bottlenecks.append("high_memory_usage")

        if self.cache_metrics.hit_rate_percent < 60:
            bottlenecks.append("low_cache_hit_rate")

        return bottlenecks

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "health_score": self.calculate_system_health_score(),
            "performance_grade": self.get_performance_grade(),
            "key_metrics": {
                "response_time_ms": self.average_response_time_ms,
                "throughput": self.request_throughput,
                "error_rate_percent": self.error_rate_percent,
                "cache_hit_rate_percent": self.cache_metrics.hit_rate_percent,
                "cpu_utilization_percent": self.cpu_utilization_percent,
                "memory_utilization_percent": self.memory_utilization_percent,
            },
            "quality_indicators": {
                "overall_quality": self.overall_quality_score,
                "user_satisfaction": self.user_satisfaction_score,
            },
            "bottlenecks": self.identify_bottlenecks(),
            "cache_performance": self.cache_metrics.get_performance_summary(),
            "memory_status": self.memory_status.to_dict(),
        }
