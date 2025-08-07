"""
Performance Feedback Loops for Adaptive Constants
=================================================

This module implements performance feedback loops that continuously learn and adapt
constants based on real system performance metrics and user interaction patterns.

Phase 3 Implementation: Real-time performance optimization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import json


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics tracked"""

    RESPONSE_TIME = "response_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    EXTRACTION_ACCURACY = "extraction_accuracy"
    SEARCH_RELEVANCE = "search_relevance"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""

    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    domain_name: Optional[str] = None
    session_id: Optional[str] = None
    constant_configuration: Optional[Dict[str, Any]] = None


@dataclass
class AdaptationRule:
    """Rule for adapting constants based on performance metrics"""

    name: str
    trigger_condition: Callable[[List[PerformanceMetric]], bool]
    adaptation_function: Callable[[List[PerformanceMetric]], Dict[str, Any]]
    target_constants: List[str]
    min_samples: int = 10
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None


class PerformanceFeedbackOrchestrator:
    """Orchestrates performance feedback loops for constant adaptation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._metrics_buffer: List[PerformanceMetric] = []
        self._adaptation_rules: List[AdaptationRule] = []
        self._feedback_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_buffer_size = 10000
        self._feedback_lock = asyncio.Lock()

        # Initialize standard adaptation rules
        self._initialize_adaptation_rules()

    def _initialize_adaptation_rules(self):
        """Initialize standard performance adaptation rules"""

        # Response time optimization rules
        self._adaptation_rules.append(
            AdaptationRule(
                name="response_time_timeout_adaptation",
                trigger_condition=self._response_time_degradation_detected,
                adaptation_function=self._adapt_timeout_constants,
                target_constants=[
                    "SystemPerformanceConstants",
                    "PerformanceAdaptiveConstants",
                ],
                min_samples=20,
                cooldown_minutes=15,
            )
        )

        # Cache performance optimization
        self._adaptation_rules.append(
            AdaptationRule(
                name="cache_hit_rate_optimization",
                trigger_condition=self._cache_performance_suboptimal,
                adaptation_function=self._adapt_cache_constants,
                target_constants=[
                    "PerformanceAdaptiveConstants",
                    "CachePerformanceConstants",
                ],
                min_samples=50,
                cooldown_minutes=30,
            )
        )

        # Extraction accuracy optimization
        self._adaptation_rules.append(
            AdaptationRule(
                name="extraction_accuracy_optimization",
                trigger_condition=self._extraction_accuracy_degraded,
                adaptation_function=self._adapt_extraction_constants,
                target_constants=[
                    "KnowledgeExtractionConstants",
                    "DomainAdaptiveConstants",
                ],
                min_samples=25,
                cooldown_minutes=60,
            )
        )

        # Search relevance optimization
        self._adaptation_rules.append(
            AdaptationRule(
                name="search_relevance_optimization",
                trigger_condition=self._search_relevance_suboptimal,
                adaptation_function=self._adapt_search_constants,
                target_constants=[
                    "UniversalSearchConstants",
                    "DomainAdaptiveConstants",
                ],
                min_samples=30,
                cooldown_minutes=45,
            )
        )

        # Memory usage optimization
        self._adaptation_rules.append(
            AdaptationRule(
                name="memory_usage_optimization",
                trigger_condition=self._memory_usage_high,
                adaptation_function=self._adapt_memory_constants,
                target_constants=[
                    "SystemPerformanceConstants",
                    "PerformanceAdaptiveConstants",
                ],
                min_samples=15,
                cooldown_minutes=20,
            )
        )

    async def record_performance_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric for feedback analysis"""

        async with self._feedback_lock:
            self._metrics_buffer.append(metric)

            # Maintain buffer size
            if len(self._metrics_buffer) > self._max_buffer_size:
                # Remove oldest metrics, keep recent ones
                self._metrics_buffer = self._metrics_buffer[-self._max_buffer_size :]

            # Update feedback history
            metric_key = f"{metric.metric_type.value}_{metric.domain_name or 'global'}"
            if metric_key not in self._feedback_history:
                self._feedback_history[metric_key] = []

            self._feedback_history[metric_key].append(
                {
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "context": metric.context,
                    "session_id": metric.session_id,
                }
            )

            # Keep history manageable (last 1000 entries per metric type)
            if len(self._feedback_history[metric_key]) > 1000:
                self._feedback_history[metric_key] = self._feedback_history[metric_key][
                    -1000:
                ]

            self.logger.debug(
                f"Recorded {metric.metric_type.value} metric: {metric.value} "
                f"(domain: {metric.domain_name}, session: {metric.session_id})"
            )

    async def process_feedback_loop(self) -> Dict[str, Any]:
        """Process performance feedback loop and trigger adaptations"""

        async with self._feedback_lock:
            results = {
                "processed_at": datetime.now().isoformat(),
                "total_metrics": len(self._metrics_buffer),
                "triggered_adaptations": [],
                "skipped_adaptations": [],
                "errors": [],
            }

            # Process each adaptation rule
            for rule in self._adaptation_rules:
                try:
                    await self._process_adaptation_rule(rule, results)
                except Exception as e:
                    error_msg = f"Failed to process rule {rule.name}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

            return results

    async def _process_adaptation_rule(
        self, rule: AdaptationRule, results: Dict[str, Any]
    ) -> None:
        """Process a single adaptation rule"""

        # Check cooldown period
        if rule.last_triggered:
            cooldown_expired = (
                datetime.now() - rule.last_triggered
            ).total_seconds() / 60 > rule.cooldown_minutes

            if not cooldown_expired:
                results["skipped_adaptations"].append(
                    {
                        "rule": rule.name,
                        "reason": "cooldown_active",
                        "minutes_remaining": rule.cooldown_minutes
                        - int(
                            (datetime.now() - rule.last_triggered).total_seconds() / 60
                        ),
                    }
                )
                return

        # Get relevant metrics for this rule
        relevant_metrics = self._get_relevant_metrics_for_rule(rule)

        # Check if we have enough samples
        if len(relevant_metrics) < rule.min_samples:
            results["skipped_adaptations"].append(
                {
                    "rule": rule.name,
                    "reason": "insufficient_samples",
                    "samples_available": len(relevant_metrics),
                    "samples_required": rule.min_samples,
                }
            )
            return

        # Check trigger condition
        try:
            should_trigger = rule.trigger_condition(relevant_metrics)
        except Exception as e:
            self.logger.error(f"Trigger condition failed for rule {rule.name}: {e}")
            results["errors"].append(f"Trigger condition error for {rule.name}: {e}")
            return

        if not should_trigger:
            results["skipped_adaptations"].append(
                {
                    "rule": rule.name,
                    "reason": "trigger_condition_not_met",
                    "samples_evaluated": len(relevant_metrics),
                }
            )
            return

        # Execute adaptation
        try:
            adaptations = rule.adaptation_function(relevant_metrics)

            # Record successful adaptation
            rule.last_triggered = datetime.now()

            adaptation_result = {
                "rule": rule.name,
                "target_constants": rule.target_constants,
                "adaptations": adaptations,
                "samples_used": len(relevant_metrics),
                "triggered_at": datetime.now().isoformat(),
            }

            results["triggered_adaptations"].append(adaptation_result)

            self.logger.info(
                f"Triggered adaptation rule '{rule.name}' with {len(adaptations)} constants adapted"
            )

        except Exception as e:
            error_msg = f"Adaptation function failed for rule {rule.name}: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

    def _get_relevant_metrics_for_rule(
        self, rule: AdaptationRule
    ) -> List[PerformanceMetric]:
        """Get metrics relevant to a specific adaptation rule"""

        # Get recent metrics (last 24 hours by default)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_metrics = [
            metric for metric in self._metrics_buffer if metric.timestamp >= cutoff_time
        ]

        # Rule-specific metric filtering
        if "response_time" in rule.name:
            return [
                m for m in recent_metrics if m.metric_type == MetricType.RESPONSE_TIME
            ]
        elif "cache" in rule.name:
            return [
                m for m in recent_metrics if m.metric_type == MetricType.CACHE_HIT_RATE
            ]
        elif "extraction" in rule.name:
            return [
                m
                for m in recent_metrics
                if m.metric_type == MetricType.EXTRACTION_ACCURACY
            ]
        elif "search" in rule.name:
            return [
                m
                for m in recent_metrics
                if m.metric_type == MetricType.SEARCH_RELEVANCE
            ]
        elif "memory" in rule.name:
            return [
                m for m in recent_metrics if m.metric_type == MetricType.MEMORY_USAGE
            ]
        else:
            return recent_metrics

    # === Trigger Condition Functions ===

    def _response_time_degradation_detected(
        self, metrics: List[PerformanceMetric]
    ) -> bool:
        """Detect if response times are degrading"""

        if len(metrics) < 10:
            return False

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)

        # Compare recent vs older metrics
        recent_count = len(sorted_metrics) // 3
        recent_avg = statistics.mean([m.value for m in sorted_metrics[-recent_count:]])
        older_avg = statistics.mean([m.value for m in sorted_metrics[:recent_count]])

        # Trigger if recent average is 50% worse than older average
        degradation_threshold = 1.5
        return recent_avg > older_avg * degradation_threshold

    def _cache_performance_suboptimal(self, metrics: List[PerformanceMetric]) -> bool:
        """Detect suboptimal cache performance"""

        if len(metrics) < 20:
            return False

        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-20:]
        avg_hit_rate = statistics.mean([m.value for m in recent_metrics])

        # Trigger if cache hit rate is below 50%
        return avg_hit_rate < 0.5

    def _extraction_accuracy_degraded(self, metrics: List[PerformanceMetric]) -> bool:
        """Detect degraded extraction accuracy"""

        if len(metrics) < 15:
            return False

        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-15:]
        avg_accuracy = statistics.mean([m.value for m in recent_metrics])

        # Trigger if accuracy is below 80%
        return avg_accuracy < 0.8

    def _search_relevance_suboptimal(self, metrics: List[PerformanceMetric]) -> bool:
        """Detect suboptimal search relevance"""

        if len(metrics) < 20:
            return False

        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-20:]
        avg_relevance = statistics.mean([m.value for m in recent_metrics])

        # Trigger if relevance is below 70%
        return avg_relevance < 0.7

    def _memory_usage_high(self, metrics: List[PerformanceMetric]) -> bool:
        """Detect high memory usage"""

        if len(metrics) < 10:
            return False

        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-10:]
        avg_memory = statistics.mean([m.value for m in recent_metrics])

        # Trigger if average memory usage > 80% (assuming values are 0-1)
        return avg_memory > 0.8

    # === Adaptation Functions ===

    def _adapt_timeout_constants(
        self, metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Adapt timeout constants based on response time metrics"""

        # Calculate adaptive timeout based on recent performance
        recent_times = [
            m.value for m in sorted(metrics, key=lambda x: x.timestamp)[-20:]
        ]
        avg_time = statistics.mean(recent_times)
        std_dev = statistics.stdev(recent_times) if len(recent_times) > 1 else 0

        # Set timeout to average + 2 standard deviations for 95% coverage
        adaptive_timeout = max(5.0, min(60.0, avg_time + (2 * std_dev)))

        return {
            "DEFAULT_TIMEOUT_SECONDS": int(adaptive_timeout),
            "SLOW_OPERATION_THRESHOLD_SECONDS": avg_time * 1.5,
            "ADAPTIVE_TIMEOUT_MULTIPLIER": 1.2 if std_dev > avg_time * 0.3 else 1.0,
        }

    def _adapt_cache_constants(
        self, metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Adapt cache constants based on hit rate metrics"""

        recent_hit_rates = [
            m.value for m in sorted(metrics, key=lambda x: x.timestamp)[-30:]
        ]
        avg_hit_rate = statistics.mean(recent_hit_rates)

        adaptations = {}

        if avg_hit_rate < 0.4:  # Very low hit rate
            # Increase cache size and TTL
            adaptations.update(
                {
                    "OPTIMAL_CACHE_SIZE_MULTIPLIER": 1.5,
                    "DEFAULT_CACHE_TTL_MULTIPLIER": 1.3,
                    "CACHE_PRELOAD_ENABLED": True,
                }
            )
        elif avg_hit_rate < 0.6:  # Low hit rate
            # Moderate increase
            adaptations.update(
                {
                    "OPTIMAL_CACHE_SIZE_MULTIPLIER": 1.2,
                    "DEFAULT_CACHE_TTL_MULTIPLIER": 1.1,
                }
            )
        elif avg_hit_rate > 0.85:  # Very high hit rate
            # Can reduce cache size to save memory
            adaptations.update(
                {
                    "OPTIMAL_CACHE_SIZE_MULTIPLIER": 0.8,
                    "CACHE_EVICTION_FREQUENCY_MULTIPLIER": 1.2,
                }
            )

        return adaptations

    def _adapt_extraction_constants(
        self, metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Adapt extraction constants based on accuracy metrics"""

        recent_accuracies = [
            m.value for m in sorted(metrics, key=lambda x: x.timestamp)[-20:]
        ]
        avg_accuracy = statistics.mean(recent_accuracies)

        adaptations = {}

        if avg_accuracy < 0.75:  # Low accuracy
            # Increase confidence thresholds for better precision
            adaptations.update(
                {
                    "ENTITY_CONFIDENCE_THRESHOLD_ADJUSTMENT": 0.05,
                    "RELATIONSHIP_CONFIDENCE_THRESHOLD_ADJUSTMENT": 0.05,
                    "QUALITY_VALIDATION_THRESHOLD_ADJUSTMENT": 0.03,
                }
            )
        elif (
            avg_accuracy > 0.9 and statistics.stdev(recent_accuracies) < 0.02
        ):  # Very high and stable
            # Can reduce thresholds slightly for better recall
            adaptations.update(
                {
                    "ENTITY_CONFIDENCE_THRESHOLD_ADJUSTMENT": -0.02,
                    "RELATIONSHIP_CONFIDENCE_THRESHOLD_ADJUSTMENT": -0.02,
                }
            )

        return adaptations

    def _adapt_search_constants(
        self, metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Adapt search constants based on relevance metrics"""

        recent_relevance = [
            m.value for m in sorted(metrics, key=lambda x: x.timestamp)[-25:]
        ]
        avg_relevance = statistics.mean(recent_relevance)

        adaptations = {}

        if avg_relevance < 0.65:  # Low relevance
            # Adjust search parameters for better quality
            adaptations.update(
                {
                    "VECTOR_SIMILARITY_THRESHOLD_ADJUSTMENT": 0.05,
                    "RESULT_SYNTHESIS_THRESHOLD_ADJUSTMENT": 0.03,
                    "VECTOR_TOP_K_ADJUSTMENT": -2,  # Reduce to focus on top results
                }
            )
        elif avg_relevance > 0.85:  # High relevance
            # Can relax thresholds for better recall
            adaptations.update(
                {
                    "VECTOR_SIMILARITY_THRESHOLD_ADJUSTMENT": -0.03,
                    "VECTOR_TOP_K_ADJUSTMENT": 3,  # Include more results
                }
            )

        return adaptations

    def _adapt_memory_constants(
        self, metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Adapt memory-related constants based on usage metrics"""

        recent_usage = [
            m.value for m in sorted(metrics, key=lambda x: x.timestamp)[-15:]
        ]
        avg_usage = statistics.mean(recent_usage)

        adaptations = {}

        if avg_usage > 0.8:  # High memory usage
            # Reduce memory-intensive operations
            adaptations.update(
                {
                    "DEFAULT_BATCH_SIZE_ADJUSTMENT": -5,
                    "MAX_CONCURRENT_REQUESTS_ADJUSTMENT": -2,
                    "CACHE_SIZE_REDUCTION_FACTOR": 0.8,
                }
            )
        elif avg_usage < 0.5:  # Low memory usage
            # Can increase batch sizes for better throughput
            adaptations.update(
                {
                    "DEFAULT_BATCH_SIZE_ADJUSTMENT": 5,
                    "MAX_CONCURRENT_REQUESTS_ADJUSTMENT": 2,
                }
            )

        return adaptations

    # === Management Methods ===

    def add_custom_adaptation_rule(self, rule: AdaptationRule) -> None:
        """Add a custom adaptation rule"""
        self._adaptation_rules.append(rule)
        self.logger.info(f"Added custom adaptation rule: {rule.name}")

    def remove_adaptation_rule(self, rule_name: str) -> bool:
        """Remove an adaptation rule by name"""
        original_count = len(self._adaptation_rules)
        self._adaptation_rules = [
            r for r in self._adaptation_rules if r.name != rule_name
        ]
        removed = len(self._adaptation_rules) < original_count

        if removed:
            self.logger.info(f"Removed adaptation rule: {rule_name}")

        return removed

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics and feedback status"""

        # Calculate metrics by type
        metrics_by_type = {}
        for metric in self._metrics_buffer:
            metric_type = metric.metric_type.value
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric.value)

        # Calculate summaries
        metric_summaries = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                metric_summaries[metric_type] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

        return {
            "total_metrics": len(self._metrics_buffer),
            "metrics_by_type": metric_summaries,
            "adaptation_rules_count": len(self._adaptation_rules),
            "feedback_history_keys": len(self._feedback_history),
            "buffer_utilization": len(self._metrics_buffer) / self._max_buffer_size,
            "oldest_metric": (
                min(
                    self._metrics_buffer, key=lambda x: x.timestamp
                ).timestamp.isoformat()
                if self._metrics_buffer
                else None
            ),
            "newest_metric": (
                max(
                    self._metrics_buffer, key=lambda x: x.timestamp
                ).timestamp.isoformat()
                if self._metrics_buffer
                else None
            ),
        }

    def clear_metrics_buffer(self, keep_recent_hours: int = 1) -> int:
        """Clear metrics buffer, optionally keeping recent metrics"""

        if keep_recent_hours > 0:
            cutoff_time = datetime.now() - timedelta(hours=keep_recent_hours)
            old_count = len(self._metrics_buffer)
            self._metrics_buffer = [
                m for m in self._metrics_buffer if m.timestamp >= cutoff_time
            ]
            cleared_count = old_count - len(self._metrics_buffer)
        else:
            cleared_count = len(self._metrics_buffer)
            self._metrics_buffer.clear()

        self.logger.info(f"Cleared {cleared_count} metrics from buffer")
        return cleared_count


# Global orchestrator instance
performance_feedback_orchestrator = PerformanceFeedbackOrchestrator()


# Convenience functions for common metric recording
async def record_response_time(
    response_time_seconds: float,
    domain_name: str = None,
    session_id: str = None,
    context: Dict[str, Any] = None,
) -> None:
    """Record response time metric"""

    metric = PerformanceMetric(
        metric_type=MetricType.RESPONSE_TIME,
        value=response_time_seconds,
        timestamp=datetime.now(),
        context=context or {},
        domain_name=domain_name,
        session_id=session_id,
    )

    await performance_feedback_orchestrator.record_performance_metric(metric)


async def record_cache_hit_rate(
    hit_rate: float,
    domain_name: str = None,
    session_id: str = None,
    context: Dict[str, Any] = None,
) -> None:
    """Record cache hit rate metric"""

    metric = PerformanceMetric(
        metric_type=MetricType.CACHE_HIT_RATE,
        value=hit_rate,
        timestamp=datetime.now(),
        context=context or {},
        domain_name=domain_name,
        session_id=session_id,
    )

    await performance_feedback_orchestrator.record_performance_metric(metric)


async def record_extraction_accuracy(
    accuracy: float,
    domain_name: str = None,
    session_id: str = None,
    context: Dict[str, Any] = None,
) -> None:
    """Record extraction accuracy metric"""

    metric = PerformanceMetric(
        metric_type=MetricType.EXTRACTION_ACCURACY,
        value=accuracy,
        timestamp=datetime.now(),
        context=context or {},
        domain_name=domain_name,
        session_id=session_id,
    )

    await performance_feedback_orchestrator.record_performance_metric(metric)


async def record_search_relevance(
    relevance: float,
    domain_name: str = None,
    session_id: str = None,
    context: Dict[str, Any] = None,
) -> None:
    """Record search relevance metric"""

    metric = PerformanceMetric(
        metric_type=MetricType.SEARCH_RELEVANCE,
        value=relevance,
        timestamp=datetime.now(),
        context=context or {},
        domain_name=domain_name,
        session_id=session_id,
    )

    await performance_feedback_orchestrator.record_performance_metric(metric)


# Export all classes and functions
__all__ = [
    "MetricType",
    "PerformanceMetric",
    "AdaptationRule",
    "PerformanceFeedbackOrchestrator",
    "performance_feedback_orchestrator",
    "record_response_time",
    "record_cache_hit_rate",
    "record_extraction_accuracy",
    "record_search_relevance",
]
