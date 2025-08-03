"""
Shared Capability Patterns - Cross-Agent Feature Utilization

This module implements shared capability patterns that allow agents to leverage
common functionalities without tight coupling. Each pattern follows dependency
injection and interface segregation principles.

Key Design Principles:
- Interface-based design with clear contracts
- Dependency injection for loose coupling
- Azure service integration for all capabilities
- Performance optimization with caching and monitoring
- Error handling and recovery for resilience
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

from pydantic import BaseModel, Field

from ..interfaces.agent_contracts import (
    AzureServiceMetrics,
    CacheContract,
    ErrorHandlingContract,
    MonitoringContract,
    StatisticalPattern,
)

# =============================================================================
# CAPABILITY INTERFACE PROTOCOLS
# =============================================================================


class CacheCapability(Protocol):
    """Protocol for shared caching capabilities"""

    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Retrieve cached value"""
        ...

    async def set(
        self, key: str, value: Any, ttl: int = 3600, namespace: str = "default"
    ) -> bool:
        """Store value in cache"""
        ...

    async def invalidate(self, pattern: str, namespace: str = "default") -> int:
        """Invalidate cache entries matching pattern"""
        ...

    async def get_stats(self, namespace: str = "default") -> Dict[str, Any]:
        """Get cache performance statistics"""
        ...


class StatisticalAnalysisCapability(Protocol):
    """Protocol for shared statistical analysis capabilities"""

    async def calculate_confidence_interval(
        self, data: List[float], confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate confidence interval for data"""
        ...

    async def perform_significance_test(
        self, sample1: List[float], sample2: List[float]
    ) -> Dict[str, float]:
        """Perform statistical significance test"""
        ...

    async def learn_patterns_from_data(
        self, data: List[str], pattern_type: str
    ) -> List[StatisticalPattern]:
        """Learn statistical patterns from data"""
        ...

    async def validate_pattern_significance(
        self, patterns: List[StatisticalPattern]
    ) -> List[StatisticalPattern]:
        """Validate statistical significance of patterns"""
        ...


class AzureServiceOrchestrationCapability(Protocol):
    """Protocol for shared Azure service orchestration"""

    async def execute_azure_ml_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Azure ML job with monitoring"""
        ...

    async def query_azure_search(self, query_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Azure Search query with optimization"""
        ...

    async def execute_cosmos_query(self, query: str, database: str) -> Dict[str, Any]:
        """Execute Azure Cosmos query with performance tracking"""
        ...

    async def optimize_azure_costs(
        self, service_usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize Azure service costs based on usage patterns"""
        ...


class ErrorRecoveryCapability(Protocol):
    """Protocol for shared error recovery capabilities"""

    async def handle_azure_service_error(
        self, error: Exception, service_name: str, operation: str
    ) -> Optional[Any]:
        """Handle Azure service errors with recovery strategies"""
        ...

    async def handle_agent_error(
        self, error: Exception, agent_name: str, operation: str
    ) -> Optional[Any]:
        """Handle agent-specific errors with fallback strategies"""
        ...

    async def get_recovery_strategy(
        self, error_type: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """Get appropriate recovery strategy for error type"""
        ...


class PerformanceMonitoringCapability(Protocol):
    """Protocol for shared performance monitoring capabilities"""

    async def track_operation_performance(
        self, operation: str, component: str, metrics: Dict[str, float]
    ) -> None:
        """Track operation performance metrics"""
        ...

    async def track_azure_service_usage(
        self, service_name: str, metrics: AzureServiceMetrics
    ) -> None:
        """Track Azure service usage and costs"""
        ...

    async def get_performance_insights(
        self, component: str, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        ...


# =============================================================================
# SHARED CAPABILITY IMPLEMENTATIONS
# =============================================================================


@dataclass
class CapabilityContext:
    """Context for capability execution with Azure service integration"""

    azure_services: Any  # AzureServiceContainer
    performance_tracker: Any
    error_handler: Any
    cache_manager: Any
    request_id: str
    operation_start_time: float = None

    def __post_init__(self):
        if self.operation_start_time is None:
            self.operation_start_time = time.time()


class SharedCacheCapability:
    """
    Shared caching capability with Azure Redis and local cache tiers

    Provides high-performance caching for all agents with:
    - Multi-tier caching (local + Azure Redis)
    - Namespace isolation for different agents
    - Automatic cache warming and invalidation
    - Cost optimization for Azure Redis usage
    """

    def __init__(self, context: CapabilityContext):
        self.context = context
        self.local_cache = {}  # In-memory cache for hot data
        self.cache_stats = {"hits": 0, "misses": 0, "azure_redis_calls": 0}

    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Retrieve cached value with multi-tier lookup"""
        namespaced_key = f"{namespace}:{key}"

        # Check local cache first (fastest)
        if namespaced_key in self.local_cache:
            entry = self.local_cache[namespaced_key]
            if not self._is_expired(entry):
                self.cache_stats["hits"] += 1
                return entry["value"]
            else:
                del self.local_cache[namespaced_key]

        # Check Azure Redis cache
        try:
            azure_redis = self.context.azure_services.redis_client
            redis_value = await azure_redis.get(namespaced_key)

            if redis_value is not None:
                self.cache_stats["hits"] += 1
                self.cache_stats["azure_redis_calls"] += 1

                # Warm local cache
                self._store_local(
                    namespaced_key, redis_value, ttl=300
                )  # 5 min local TTL
                return redis_value

        except Exception as e:
            await self.context.error_handler.handle_error(
                error=e, operation="azure_redis_get", component="SharedCacheCapability"
            )

        self.cache_stats["misses"] += 1
        return None

    async def set(
        self, key: str, value: Any, ttl: int = 3600, namespace: str = "default"
    ) -> bool:
        """Store value in multi-tier cache"""
        namespaced_key = f"{namespace}:{key}"

        try:
            # Store in Azure Redis
            azure_redis = self.context.azure_services.redis_client
            await azure_redis.setex(namespaced_key, ttl, value)

            # Store in local cache for hot access
            local_ttl = min(300, ttl)  # Max 5 minutes local cache
            self._store_local(namespaced_key, value, local_ttl)

            self.cache_stats["azure_redis_calls"] += 1
            return True

        except Exception as e:
            await self.context.error_handler.handle_error(
                error=e, operation="azure_redis_set", component="SharedCacheCapability"
            )
            return False

    async def invalidate(self, pattern: str, namespace: str = "default") -> int:
        """Invalidate cache entries matching pattern"""
        namespaced_pattern = f"{namespace}:{pattern}"
        invalidated_count = 0

        try:
            # Invalidate Azure Redis entries
            azure_redis = self.context.azure_services.redis_client
            keys = await azure_redis.keys(namespaced_pattern)
            if keys:
                invalidated_count = await azure_redis.delete(*keys)
                self.cache_stats["azure_redis_calls"] += 1

            # Invalidate local cache entries
            local_keys_to_remove = [
                k
                for k in self.local_cache.keys()
                if k.startswith(namespaced_pattern.replace("*", ""))
            ]
            for key in local_keys_to_remove:
                del self.local_cache[key]
                invalidated_count += 1

            return invalidated_count

        except Exception as e:
            await self.context.error_handler.handle_error(
                error=e,
                operation="cache_invalidation",
                component="SharedCacheCapability",
            )
            return 0

    async def get_stats(self, namespace: str = "default") -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / max(1, total_requests)) * 100

        return {
            "namespace": namespace,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
            "azure_redis_calls": self.cache_stats["azure_redis_calls"],
            "local_cache_size": len(self.local_cache),
            "performance_tier": "high"
            if hit_rate > 80
            else "medium"
            if hit_rate > 60
            else "low",
        }

    def _store_local(self, key: str, value: Any, ttl: int):
        """Store in local cache with TTL"""
        self.local_cache[key] = {"value": value, "expires_at": time.time() + ttl}

    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired"""
        return time.time() > entry["expires_at"]


class SharedStatisticalAnalysisCapability:
    """
    Shared statistical analysis capability using Azure ML and statistical libraries

    Provides statistical analysis services for all agents with:
    - Azure ML integration for advanced analytics
    - Statistical significance testing
    - Pattern learning and validation
    - Confidence interval calculations
    """

    def __init__(self, context: CapabilityContext):
        self.context = context

    async def calculate_confidence_interval(
        self, data: List[float], confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate confidence interval using statistical methods"""
        try:
            import numpy as np
            from scipy import stats

            if len(data) < 2:
                return {"lower": 0.0, "upper": 0.0, "mean": 0.0, "std_error": 0.0}

            mean = np.mean(data)
            std_error = stats.sem(data)

            # Calculate confidence interval
            degrees_freedom = len(data) - 1
            confidence_interval = stats.t.interval(
                confidence_level, degrees_freedom, loc=mean, scale=std_error
            )

            return {
                "lower": float(confidence_interval[0]),
                "upper": float(confidence_interval[1]),
                "mean": float(mean),
                "std_error": float(std_error),
                "sample_size": len(data),
                "confidence_level": confidence_level,
            }

        except Exception as e:
            await self.context.error_handler.handle_error(
                error=e,
                operation="confidence_interval_calculation",
                component="SharedStatisticalAnalysisCapability",
            )
            return {"lower": 0.0, "upper": 0.0, "mean": 0.0, "std_error": 0.0}

    async def perform_significance_test(
        self, sample1: List[float], sample2: List[float]
    ) -> Dict[str, float]:
        """Perform statistical significance test between two samples"""
        try:
            from scipy import stats

            if len(sample1) < 2 or len(sample2) < 2:
                return {"p_value": 1.0, "statistic": 0.0, "significant": False}

            # Perform t-test
            statistic, p_value = stats.ttest_ind(sample1, sample2)

            # Perform Mann-Whitney U test (non-parametric)
            u_statistic, u_p_value = stats.mannwhitneyu(
                sample1, sample2, alternative="two-sided"
            )

            return {
                "t_test_p_value": float(p_value),
                "t_test_statistic": float(statistic),
                "mannwhitney_p_value": float(u_p_value),
                "mannwhitney_statistic": float(u_statistic),
                "significant": float(p_value) < 0.05,
                "effect_size": self._calculate_effect_size(sample1, sample2),
            }

        except Exception as e:
            await self.context.error_handler.handle_error(
                error=e,
                operation="significance_test",
                component="SharedStatisticalAnalysisCapability",
            )
            return {"p_value": 1.0, "statistic": 0.0, "significant": False}

    async def learn_patterns_from_data(
        self, data: List[str], pattern_type: str
    ) -> List[StatisticalPattern]:
        """Learn statistical patterns using Azure ML and local analysis"""
        try:
            # Use Azure ML for pattern discovery
            ml_client = self.context.azure_services.ml_client

            pattern_learning_job = {
                "algorithm": "frequent_pattern_mining",
                "data": data,
                "pattern_type": pattern_type,
                "min_support": 0.05,  # 5% minimum support
                "min_confidence": 0.7,  # 70% minimum confidence
            }

            ml_result = await ml_client.submit_pattern_mining_job(pattern_learning_job)

            # Convert Azure ML results to StatisticalPattern objects
            patterns = []
            for pattern_data in ml_result.get("patterns", []):
                pattern = StatisticalPattern(
                    pattern_id=f"{pattern_type}_{pattern_data['id']}",
                    pattern_text=pattern_data["text"],
                    pattern_type=pattern_type,
                    frequency=pattern_data["frequency"],
                    confidence=pattern_data["confidence"],
                    support=pattern_data["support"],
                    lift=pattern_data.get("lift", 1.0),
                    chi_square_p_value=pattern_data.get("chi_square_p", 0.05),
                    confidence_interval=pattern_data.get(
                        "confidence_interval", {"lower": 0.0, "upper": 1.0}
                    ),
                    source_documents=pattern_data.get("source_docs", []),
                    azure_ml_features=pattern_data.get("features", {}),
                )
                patterns.append(pattern)

            return patterns

        except Exception as e:
            await self.context.error_handler.handle_error(
                error=e,
                operation="pattern_learning",
                component="SharedStatisticalAnalysisCapability",
            )
            return []

    async def validate_pattern_significance(
        self, patterns: List[StatisticalPattern]
    ) -> List[StatisticalPattern]:
        """Validate statistical significance of learned patterns"""
        validated_patterns = []

        for pattern in patterns:
            # Apply statistical validation criteria
            if (
                pattern.chi_square_p_value < 0.05
                and pattern.frequency >= 3  # Statistically significant
                and pattern.confidence >= 0.6  # Minimum frequency
            ):  # Minimum confidence
                validated_patterns.append(pattern)

        return validated_patterns

    def _calculate_effect_size(
        self, sample1: List[float], sample2: List[float]
    ) -> float:
        """Calculate Cohen's d effect size"""
        try:
            import numpy as np

            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((len(sample1) - 1) * std1**2 + (len(sample2) - 1) * std2**2)
                / (len(sample1) + len(sample2) - 2)
            )

            # Cohen's d
            effect_size = (mean1 - mean2) / pooled_std
            return float(abs(effect_size))

        except:
            return 0.0


class SharedAzureServiceOrchestrationCapability:
    """
    Shared Azure service orchestration capability

    Provides centralized Azure service orchestration for all agents with:
    - Cost optimization and resource management
    - Performance monitoring and optimization
    - Service health monitoring and circuit breaking
    - Intelligent request routing and load balancing
    """

    def __init__(self, context: CapabilityContext):
        self.context = context
        self.service_metrics = {}

    async def execute_azure_ml_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Azure ML job with comprehensive monitoring"""
        start_time = time.time()

        try:
            ml_client = self.context.azure_services.ml_client

            # Submit job with monitoring
            job_result = await ml_client.submit_job(job_config)

            # Track performance metrics
            execution_time = time.time() - start_time
            await self._track_service_metrics(
                "azure_ml", execution_time, True, job_config
            )

            return {
                "job_id": job_result["job_id"],
                "status": job_result["status"],
                "result": job_result.get("result"),
                "execution_time": execution_time,
                "cost_estimate": await self._estimate_ml_cost(job_config),
            }

        except Exception as e:
            execution_time = time.time() - start_time
            await self._track_service_metrics(
                "azure_ml", execution_time, False, job_config
            )

            await self.context.error_handler.handle_error(
                error=e,
                operation="azure_ml_job_execution",
                component="SharedAzureServiceOrchestrationCapability",
            )
            raise

    async def query_azure_search(self, query_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Azure Search query with optimization"""
        start_time = time.time()

        try:
            search_client = self.context.azure_services.search_client

            # Optimize query based on historical performance
            optimized_config = await self._optimize_search_query(query_config)

            # Execute search
            search_result = await search_client.search(optimized_config)

            # Track performance metrics
            execution_time = time.time() - start_time
            await self._track_service_metrics(
                "azure_search", execution_time, True, optimized_config
            )

            return {
                "results": search_result["results"],
                "total_count": search_result["total_count"],
                "execution_time": execution_time,
                "optimization_applied": optimized_config != query_config,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            await self._track_service_metrics(
                "azure_search", execution_time, False, query_config
            )

            await self.context.error_handler.handle_error(
                error=e,
                operation="azure_search_query",
                component="SharedAzureServiceOrchestrationCapability",
            )
            raise

    async def execute_cosmos_query(self, query: str, database: str) -> Dict[str, Any]:
        """Execute Azure Cosmos query with performance tracking"""
        start_time = time.time()

        try:
            cosmos_client = self.context.azure_services.cosmos_client

            # Execute query
            result = await cosmos_client.execute_gremlin_query(query, database)

            # Track performance metrics
            execution_time = time.time() - start_time
            await self._track_service_metrics(
                "azure_cosmos", execution_time, True, {"query": query}
            )

            return {
                "result": result,
                "execution_time": execution_time,
                "database": database,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            await self._track_service_metrics(
                "azure_cosmos", execution_time, False, {"query": query}
            )

            await self.context.error_handler.handle_error(
                error=e,
                operation="azure_cosmos_query",
                component="SharedAzureServiceOrchestrationCapability",
            )
            raise

    async def optimize_azure_costs(
        self, service_usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize Azure service costs based on usage patterns"""
        optimizations = {}

        # Analyze Azure ML usage
        ml_usage = service_usage.get("azure_ml", {})
        if ml_usage.get("request_count", 0) > 100:
            optimizations["azure_ml"] = {
                "recommendation": "batch_processing",
                "potential_savings_percent": 25,
                "implementation": "Batch multiple ML inference requests",
            }

        # Analyze Azure Search usage
        search_usage = service_usage.get("azure_search", {})
        if search_usage.get("query_frequency") > 1000:
            optimizations["azure_search"] = {
                "recommendation": "result_caching",
                "potential_savings_percent": 30,
                "implementation": "Cache frequent search results",
            }

        # Analyze Azure Cosmos usage
        cosmos_usage = service_usage.get("azure_cosmos", {})
        if cosmos_usage.get("ru_consumption") > 10000:
            optimizations["azure_cosmos"] = {
                "recommendation": "query_optimization",
                "potential_savings_percent": 20,
                "implementation": "Optimize Gremlin queries for lower RU consumption",
            }

        return optimizations

    async def _track_service_metrics(
        self, service: str, execution_time: float, success: bool, config: Dict[str, Any]
    ):
        """Track service performance metrics"""
        if service not in self.service_metrics:
            self.service_metrics[service] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
            }

        metrics = self.service_metrics[service]
        metrics["total_requests"] += 1
        metrics["total_execution_time"] += execution_time

        if success:
            metrics["successful_requests"] += 1

        metrics["average_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_requests"]
        )

        # Track with performance monitoring capability
        await self.context.performance_tracker.track_azure_service_usage(
            service_name=service,
            metrics=AzureServiceMetrics(
                service_name=service,
                request_count=1,
                response_time_ms=execution_time * 1000,
                success_rate=1.0 if success else 0.0,
                cost_estimate_usd=await self._estimate_service_cost(service, config),
            ),
        )

    async def _optimize_search_query(
        self, query_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize search query based on historical performance"""
        # Simple optimization example - in production would use ML models
        optimized_config = query_config.copy()

        # Optimize top_k based on typical result usage
        if "top_k" in optimized_config and optimized_config["top_k"] > 50:
            optimized_config["top_k"] = 50  # Reduce to improve performance

        return optimized_config

    async def _estimate_ml_cost(self, job_config: Dict[str, Any]) -> float:
        """Estimate Azure ML job cost"""
        # Simplified cost estimation - in production would use Azure pricing APIs
        base_cost = 0.1  # Base cost per job
        compute_factor = job_config.get("compute_nodes", 1) * 0.05
        return base_cost + compute_factor

    async def _estimate_service_cost(
        self, service: str, config: Dict[str, Any]
    ) -> float:
        """Estimate service cost for request"""
        base_costs = {"azure_ml": 0.1, "azure_search": 0.01, "azure_cosmos": 0.005}
        return base_costs.get(service, 0.01)


# =============================================================================
# CAPABILITY MANAGER AND DEPENDENCY INJECTION
# =============================================================================


class CapabilityManager:
    """
    Capability manager providing dependency injection for shared capabilities

    Manages capability lifecycle and provides interface-based access to
    shared capabilities for all agents.
    """

    def __init__(self, azure_services):
        self.azure_services = azure_services
        self._capabilities = {}
        self._capability_contracts = {}

    async def get_cache_capability(self, contract: CacheContract) -> CacheCapability:
        """Get cache capability with contract specification"""
        key = f"cache_{contract.cache_key_namespace}"

        if key not in self._capabilities:
            context = CapabilityContext(
                azure_services=self.azure_services,
                performance_tracker=await self._get_performance_tracker(),
                error_handler=await self._get_error_handler(),
                cache_manager=None,  # Will be self-referential
                request_id=f"cache_capability_{int(time.time())}",
            )

            self._capabilities[key] = SharedCacheCapability(context)
            self._capability_contracts[key] = contract

        return self._capabilities[key]

    async def get_statistical_analysis_capability(
        self,
    ) -> StatisticalAnalysisCapability:
        """Get statistical analysis capability"""
        key = "statistical_analysis"

        if key not in self._capabilities:
            context = CapabilityContext(
                azure_services=self.azure_services,
                performance_tracker=await self._get_performance_tracker(),
                error_handler=await self._get_error_handler(),
                cache_manager=await self.get_cache_capability(
                    CacheContract(cache_key_namespace="stats")
                ),
                request_id=f"stats_capability_{int(time.time())}",
            )

            self._capabilities[key] = SharedStatisticalAnalysisCapability(context)

        return self._capabilities[key]

    async def get_azure_orchestration_capability(
        self,
    ) -> AzureServiceOrchestrationCapability:
        """Get Azure service orchestration capability"""
        key = "azure_orchestration"

        if key not in self._capabilities:
            context = CapabilityContext(
                azure_services=self.azure_services,
                performance_tracker=await self._get_performance_tracker(),
                error_handler=await self._get_error_handler(),
                cache_manager=await self.get_cache_capability(
                    CacheContract(cache_key_namespace="azure")
                ),
                request_id=f"azure_capability_{int(time.time())}",
            )

            self._capabilities[key] = SharedAzureServiceOrchestrationCapability(context)

        return self._capabilities[key]

    async def _get_performance_tracker(self):
        """Get performance tracking capability"""
        # Would integrate with Azure Application Insights
        return self.azure_services.application_insights_client

    async def _get_error_handler(self):
        """Get error handling capability"""
        # Would use the unified error handler
        from ..core.error_handler import get_error_handler

        return get_error_handler()

    async def get_capability_health_status(self) -> Dict[str, Any]:
        """Get health status of all managed capabilities"""
        health_status = {}

        for capability_key, capability in self._capabilities.items():
            try:
                if hasattr(capability, "get_stats"):
                    stats = await capability.get_stats()
                    health_status[capability_key] = {
                        "status": "healthy",
                        "stats": stats,
                    }
                else:
                    health_status[capability_key] = {"status": "healthy", "stats": {}}

            except Exception as e:
                health_status[capability_key] = {"status": "unhealthy", "error": str(e)}

        return health_status


# =============================================================================
# CAPABILITY DECORATORS FOR AGENT INTEGRATION
# =============================================================================


def with_cache_capability(namespace: str = "default", ttl: int = 3600):
    """Decorator to inject cache capability into agent methods"""

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Get cache capability from capability manager
            cache_contract = CacheContract(
                cache_key_namespace=namespace, ttl_seconds=ttl
            )
            cache_capability = await self.capability_manager.get_cache_capability(
                cache_contract
            )

            # Inject as first parameter
            return await func(self, cache_capability, *args, **kwargs)

        return wrapper

    return decorator


def with_statistical_analysis_capability():
    """Decorator to inject statistical analysis capability into agent methods"""

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            stats_capability = (
                await self.capability_manager.get_statistical_analysis_capability()
            )
            return await func(self, stats_capability, *args, **kwargs)

        return wrapper

    return decorator


def with_azure_orchestration_capability():
    """Decorator to inject Azure orchestration capability into agent methods"""

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            azure_capability = (
                await self.capability_manager.get_azure_orchestration_capability()
            )
            return await func(self, azure_capability, *args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# EXAMPLE USAGE IN AGENTS
# =============================================================================


class ExampleAgentWithSharedCapabilities:
    """Example agent demonstrating shared capability usage"""

    def __init__(self, azure_services, capability_manager: CapabilityManager):
        self.azure_services = azure_services
        self.capability_manager = capability_manager

    @with_cache_capability(namespace="domain_analysis", ttl=1800)
    async def analyze_domain_with_caching(
        self, cache: CacheCapability, domain_data: List[str]
    ) -> Dict[str, Any]:
        """Example method using shared cache capability"""

        # Check cache first
        cache_key = f"domain_analysis_{hash(''.join(domain_data))}"
        cached_result = await cache.get(cache_key)

        if cached_result:
            return cached_result

        # Perform analysis
        analysis_result = {"domain": "example", "confidence": 0.85}

        # Cache result
        await cache.set(cache_key, analysis_result)

        return analysis_result

    @with_statistical_analysis_capability()
    async def validate_patterns_with_statistics(
        self, stats: StatisticalAnalysisCapability, patterns: List[Dict]
    ) -> List[Dict]:
        """Example method using shared statistical analysis capability"""

        # Convert to StatisticalPattern objects and validate
        statistical_patterns = [
            StatisticalPattern(
                pattern_id=f"pattern_{i}",
                pattern_text=p["text"],
                pattern_type="entity",
                frequency=p["frequency"],
                confidence=p["confidence"],
                support=p["support"],
                lift=1.0,
                chi_square_p_value=0.01,
                confidence_interval={"lower": 0.6, "upper": 0.9},
                source_documents=[],
                azure_ml_features={},
            )
            for i, p in enumerate(patterns)
        ]

        validated_patterns = await stats.validate_pattern_significance(
            statistical_patterns
        )

        return [p.model_dump() for p in validated_patterns]

    @with_azure_orchestration_capability()
    async def execute_ml_analysis_with_orchestration(
        self, azure_orch: AzureServiceOrchestrationCapability, analysis_config: Dict
    ) -> Dict[str, Any]:
        """Example method using shared Azure orchestration capability"""

        # Execute Azure ML job with orchestration and monitoring
        ml_result = await azure_orch.execute_azure_ml_job(analysis_config)

        return ml_result
