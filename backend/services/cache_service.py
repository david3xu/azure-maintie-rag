"""
Consolidated Cache Service
Merges cache_service.py (simple caching) + cache_orchestrator.py (cache orchestration patterns)

This service provides both:
1. Simple caching functionality with Redis/memory support
2. Multi-level cache orchestration with performance optimization
3. Intelligent cache strategies and cleanup
4. Agent memory management integration

Architecture:
- Maintains backward compatibility with existing cache patterns
- Adds modern orchestration capabilities for complex caching scenarios
- Integrates with agent memory management for intelligent caching
- Provides comprehensive performance monitoring and optimization
"""

import json
import hashlib
import logging
import asyncio
import time
from typing import Any, Optional, Callable, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ===== CACHE CONFIGURATION AND TYPES =====

@dataclass
class CacheConfiguration:
    """Configuration for cache behavior"""
    use_redis: bool = False
    default_ttl: int = 300
    max_memory_entries: int = 1000
    cleanup_interval: int = 300  # seconds
    multi_level_enabled: bool = True


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    errors: int = 0
    total_requests: int = 0
    hit_ratio: float = 0.0
    

class ConsolidatedCacheService:
    """
    Consolidated cache service combining simple caching
    with multi-level orchestration patterns.
    
    Provides both:
    - Simple cache operations (backward compatibility)
    - Multi-level cache orchestration (new capabilities)
    """

    def __init__(self, use_redis: bool = False, config: Optional[CacheConfiguration] = None):
        """
        Initialize consolidated cache service
        Args:
            use_redis: Whether to use Redis (if available) or in-memory cache
            config: Cache configuration options
        """
        self.config = config or CacheConfiguration(use_redis=use_redis)
        self.use_redis = self.config.use_redis
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None
        self.memory_manager = None  # Will be initialized asynchronously
        
        # Performance metrics
        self.metrics = CacheMetrics()
        self.last_cleanup = datetime.now()
        
        # Initialize Redis if requested
        if self.use_redis:
            self._initialize_redis()
        
        logger.info(f"Consolidated Cache Service initialized (Redis: {self.use_redis}, Multi-level: {self.config.multi_level_enabled})")

    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            logger.info("Redis cache initialized")
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")
            self.use_redis = False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to memory cache")
            self.use_redis = False

    async def _ensure_memory_manager(self):
        """Ensure agent memory manager is initialized"""
        if self.memory_manager is None and self.config.multi_level_enabled:
            try:
                from agents.memory.bounded_memory_manager import get_memory_manager
                self.memory_manager = await get_memory_manager()
                logger.debug("Agent memory manager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize memory manager: {e}")

    # ===== SIMPLE CACHE METHODS (Backward Compatibility) =====

    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate consistent cache key"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _get_from_memory(self, cache_key: str) -> Optional[Any]:
        """Get from memory cache"""
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            # Check if expired
            if datetime.now() < entry['expires']:
                return entry['data']
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]
                self.metrics.evictions += 1
        return None

    async def _set_in_memory(self, cache_key: str, data: Any, ttl_seconds: int):
        """Set in memory cache"""
        # Cleanup if too many entries
        if len(self.memory_cache) >= self.config.max_memory_entries:
            await self._cleanup_memory_cache()
        
        expires = datetime.now() + timedelta(seconds=ttl_seconds)
        self.memory_cache[cache_key] = {
            'data': data,
            'expires': expires,
            'created': datetime.now()
        }

    async def _get_from_redis(self, cache_key: str) -> Optional[Any]:
        """Get from Redis cache"""
        try:
            cached_result = await self.redis_client.get(cache_key)
            return json.loads(cached_result) if cached_result else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.metrics.errors += 1
            return None

    async def _set_in_redis(self, cache_key: str, data: Any, ttl_seconds: int):
        """Set in Redis cache"""
        try:
            await self.redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.metrics.errors += 1

    async def get(self, cache_key: str) -> Optional[Any]:
        """Simple cache get - Level 1 cache only"""
        self.metrics.total_requests += 1
        
        if self.use_redis and self.redis_client:
            result = await self._get_from_redis(cache_key)
        else:
            result = await self._get_from_memory(cache_key)
        
        if result is not None:
            self.metrics.hits += 1
            logger.debug(f"Cache hit (L1): {cache_key}")
        else:
            self.metrics.misses += 1
        
        self._update_hit_ratio()
        return result

    async def set(self, cache_key: str, data: Any, ttl_seconds: int = 300):
        """Simple cache set - Level 1 cache only"""
        if self.use_redis and self.redis_client:
            await self._set_in_redis(cache_key, data, ttl_seconds)
        else:
            await self._set_in_memory(cache_key, data, ttl_seconds)
        
        self.metrics.sets += 1
        logger.debug(f"Cached (L1): {cache_key}")

    async def get_or_compute(
        self, 
        cache_key: str, 
        compute_fn: Callable,
        ttl_seconds: int = 300
    ) -> Any:
        """Get from cache or compute and cache result"""
        
        # Try cache first
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache HIT for key: {cache_key[:20]}...")
            return cached_result
        
        logger.debug(f"Cache MISS for key: {cache_key[:20]}...")
        
        # Cache miss - compute result
        result = await compute_fn()
        
        # Cache the result
        await self.set(cache_key, result, ttl_seconds)
        
        return result

    # ===== QUERY-SPECIFIC CACHE METHODS (Backward Compatibility) =====

    async def cache_query_result(
        self, 
        query: str, 
        domain: str, 
        result: Any,
        ttl_seconds: int = 300
    ):
        """Cache query result (legacy method)"""
        cache_key = self._generate_cache_key("query", query, domain)
        await self.set(cache_key, result, ttl_seconds)
        logger.debug(f"Cached query result for: {query[:50]}...")

    async def get_cached_query_result(
        self, 
        query: str, 
        domain: str
    ) -> Optional[Any]:
        """Get cached query result (legacy method)"""
        cache_key = self._generate_cache_key("query", query, domain)
        result = await self.get(cache_key)
        if result:
            logger.debug(f"Retrieved cached result for: {query[:50]}...")
        return result

    async def set_cached_query_result(
        self, 
        cache_key: str, 
        result: Any,
        ttl_seconds: int = 300
    ):
        """Set cached query result (legacy method)"""
        await self.set(cache_key, result, ttl_seconds)

    async def cache_search_result(
        self,
        search_type: str,
        query: str,
        domain: str,
        result: Any,
        ttl_seconds: int = 180  # Shorter TTL for search results
    ):
        """Cache search-specific results"""
        cache_key = self._generate_cache_key("search", search_type, query, domain)
        await self.set(cache_key, result, ttl_seconds)
        logger.debug(f"Cached {search_type} search result for: {query[:50]}...")

    async def get_cached_search_result(
        self,
        search_type: str,
        query: str,
        domain: str
    ) -> Optional[Any]:
        """Get cached search-specific result"""
        cache_key = self._generate_cache_key("search", search_type, query, domain)
        result = await self.get(cache_key)
        if result:
            logger.debug(f"Retrieved cached {search_type} result for: {query[:50]}...")
        return result

    # ===== MULTI-LEVEL ORCHESTRATION METHODS =====

    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Get cached result using multi-level cache strategy.
        
        Cache levels checked in order:
        1. In-memory cache (fastest)
        2. Agent memory (intelligent)
        3. Redis cache (if configured)
        """
        if not self.config.multi_level_enabled:
            return await self.get(cache_key)
        
        try:
            self.metrics.total_requests += 1
            
            # Level 1: Check in-memory cache first
            result = await self._get_from_memory(cache_key)
            if result is not None:
                self.metrics.hits += 1
                logger.debug(f"Cache hit (L1): {cache_key}")
                self._update_hit_ratio()
                return result
            
            # Level 2: Check agent memory management
            await self._ensure_memory_manager()
            if self.memory_manager:
                try:
                    result = await self.memory_manager.retrieve_pattern(
                        cache_key, 
                        cache_type="result"
                    )
                    if result is not None:
                        self.metrics.hits += 1
                        # Promote to L1 cache for faster future access
                        await self._set_in_memory(cache_key, result, 300)
                        logger.debug(f"Cache hit (L2): {cache_key}")
                        self._update_hit_ratio()
                        return result
                except Exception as e:
                    logger.debug(f"L2 cache error: {e}")
            
            # Level 3: Check Redis if available and not already checked
            if self.use_redis and self.redis_client:
                result = await self._get_from_redis(cache_key)
                if result is not None:
                    self.metrics.hits += 1
                    # Promote to L1 cache
                    await self._set_in_memory(cache_key, result, 300)
                    logger.debug(f"Cache hit (L3-Redis): {cache_key}")
                    self._update_hit_ratio()
                    return result
            
            # Cache miss across all levels
            self.metrics.misses += 1
            logger.debug(f"Cache miss (all levels): {cache_key}")
            self._update_hit_ratio()
            return None
            
        except Exception as e:
            logger.error(f"Multi-level cache retrieval error for {cache_key}: {e}")
            self.metrics.misses += 1
            self.metrics.errors += 1
            self._update_hit_ratio()
            return None

    async def cache_result(
        self, 
        cache_key: str, 
        data: Any, 
        ttl_seconds: int = 600,
        cache_level: str = "multi"
    ) -> bool:
        """
        Cache result using specified strategy.
        
        Args:
            cache_key: Unique identifier for cached data
            data: Data to cache
            ttl_seconds: Time to live in seconds
            cache_level: "L1", "L2", "L3", "multi" (default)
        """
        if not self.config.multi_level_enabled:
            await self.set(cache_key, data, ttl_seconds)
            return True
        
        try:
            success = False
            
            if cache_level in ["L1", "multi"]:
                # Cache in L1 (in-memory)
                await self._set_in_memory(
                    cache_key, 
                    data, 
                    min(ttl_seconds, 3600)  # L1 max 1 hour
                )
                success = True
            
            if cache_level in ["L2", "multi"]:
                # Cache in L2 (agent memory)
                await self._ensure_memory_manager()
                if self.memory_manager:
                    try:
                        await self.memory_manager.store_pattern(
                            cache_key, 
                            data, 
                            cache_type="result"
                        )
                        success = True
                    except Exception as e:
                        logger.debug(f"L2 cache storage error: {e}")
            
            if cache_level in ["L3", "multi"] and self.use_redis and self.redis_client:
                # Cache in L3 (Redis)
                await self._set_in_redis(cache_key, data, ttl_seconds)
                success = True
            
            if success:
                self.metrics.sets += 1
                logger.debug(f"Cached result ({cache_level}): {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Multi-level cache storage error for {cache_key}: {e}")
            self.metrics.errors += 1
            return False

    # ===== CACHE MANAGEMENT AND OPTIMIZATION =====

    async def invalidate_cache(self, pattern: str = None, cache_key: str = None) -> int:
        """
        Invalidate cache entries by pattern or specific key.
        
        Returns number of entries invalidated.
        """
        try:
            invalidated_count = 0
            
            if cache_key:
                # Invalidate specific key across all levels
                
                # L1 (memory)
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    invalidated_count += 1
                
                # L3 (Redis)
                if self.use_redis and self.redis_client:
                    try:
                        if await self.redis_client.delete(cache_key):
                            invalidated_count += 1
                    except Exception as e:
                        logger.error(f"Redis delete error: {e}")
                
                # L2 (Agent memory) - no direct key deletion available
                
            elif pattern:
                # Pattern-based invalidation (L1 only for now)
                keys_to_delete = [
                    key for key in self.memory_cache.keys() 
                    if pattern in key
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    invalidated_count += 1
            
            self.metrics.evictions += invalidated_count
            logger.info(f"Invalidated {invalidated_count} cache entries")
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            self.metrics.errors += 1
            return 0

    async def clear_cache(self):
        """Clear all cache levels"""
        try:
            # Clear memory cache
            cleared_memory = len(self.memory_cache)
            self.memory_cache.clear()
            
            # Clear Redis cache
            cleared_redis = 0
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.flushdb()
                    cleared_redis = 1  # Can't get exact count
                    logger.info("Redis cache cleared")
                except Exception as e:
                    logger.error(f"Error clearing Redis cache: {e}")
            
            self.metrics.evictions += cleared_memory + cleared_redis
            logger.info(f"All caches cleared (Memory: {cleared_memory}, Redis: {cleared_redis > 0})")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.metrics.errors += 1

    async def clear_domain_cache(self, domain: str):
        """Clear cache entries for specific domain"""
        pattern = f"query:{domain}" if domain else "query:"
        await self.invalidate_cache(pattern=pattern)

    async def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if now >= entry['expires']
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still too many entries, remove oldest
        if len(self.memory_cache) >= self.config.max_memory_entries:
            sorted_entries = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]['created']
            )
            
            # Remove oldest 20%
            remove_count = max(1, len(sorted_entries) // 5)
            for key, _ in sorted_entries[:remove_count]:
                del self.memory_cache[key]
                self.metrics.evictions += 1
        
        self.metrics.evictions += len(expired_keys)
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries across all levels"""
        cleaned_count = 0
        
        # Clean memory cache
        before_count = len(self.memory_cache)
        await self._cleanup_memory_cache()
        cleaned_count += before_count - len(self.memory_cache)
        
        self.last_cleanup = datetime.now()
        return cleaned_count

    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze and optimize cache performance.
        
        Returns optimization recommendations and actions taken.
        """
        try:
            optimization_results = {
                "actions_taken": [],
                "recommendations": [],
                "performance_metrics": await self.get_cache_metrics()
            }
            
            # Calculate current performance
            hit_ratio = self.metrics.hit_ratio
            
            # Optimization logic
            if hit_ratio < 0.3:  # Poor hit ratio
                optimization_results["recommendations"].append(
                    "Consider increasing cache TTL or improving cache key strategy"
                )
            
            if self.metrics.evictions > self.metrics.sets * 0.5:
                optimization_results["recommendations"].append(
                    "High eviction rate - consider increasing cache size"
                )
                
                # Auto-optimization: Clean up expired entries
                cleaned_count = await self.cleanup_expired()
                if cleaned_count > 0:
                    optimization_results["actions_taken"].append(f"Cleaned up {cleaned_count} expired entries")
            
            # Memory pressure check
            if len(self.memory_cache) > self.config.max_memory_entries * 0.8:
                optimization_results["recommendations"].append(
                    "High memory cache utilization - consider cleanup or size increase"
                )
            
            # Agent memory manager check
            await self._ensure_memory_manager()
            if self.memory_manager:
                try:
                    memory_stats = await self.memory_manager.health_check()
                    if memory_stats.get("memory_utilization", 0) > 0.8:
                        optimization_results["recommendations"].append(
                            "High agent memory utilization - consider pattern cleanup"
                        )
                except Exception:
                    pass
            
            logger.info("Cache performance optimization completed", extra={
                "hit_ratio": hit_ratio,
                "actions_taken": len(optimization_results["actions_taken"]),
                "recommendations": len(optimization_results["recommendations"])
            })
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
            self.metrics.errors += 1
            return {"error": str(e)}

    def _update_hit_ratio(self):
        """Update hit ratio metric"""
        if self.metrics.total_requests > 0:
            self.metrics.hit_ratio = self.metrics.hits / self.metrics.total_requests

    # ===== METRICS AND MONITORING =====

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (legacy method)"""
        if self.use_redis and self.redis_client:
            try:
                info = await self.redis_client.info()
                return {
                    "cache_type": "redis",
                    "keys": info.get("db0", {}).get("keys", 0),
                    "memory_usage": info.get("used_memory_human", "unknown")
                }
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
                return {"cache_type": "redis", "error": str(e)}
        else:
            # Clean expired entries for accurate count
            await self._cleanup_memory_cache()
            
            return {
                "cache_type": "memory",
                "keys": len(self.memory_cache),
                "memory_entries": len(self.memory_cache)
            }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache service metrics (legacy method)"""
        return await self.get_cache_stats()

    async def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance metrics"""
        try:
            metrics = {
                "performance_metrics": {
                    "hits": self.metrics.hits,
                    "misses": self.metrics.misses,
                    "sets": self.metrics.sets,
                    "evictions": self.metrics.evictions,
                    "errors": self.metrics.errors,
                    "total_requests": self.metrics.total_requests,
                    "hit_ratio": self.metrics.hit_ratio
                },
                "cache_stats": await self.get_cache_stats(),
                "configuration": {
                    "use_redis": self.use_redis,
                    "multi_level_enabled": self.config.multi_level_enabled,
                    "max_memory_entries": self.config.max_memory_entries,
                    "default_ttl": self.config.default_ttl
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add memory manager metrics if available
            await self._ensure_memory_manager()
            if self.memory_manager:
                try:
                    memory_metrics = await self.memory_manager.get_comprehensive_stats()
                    metrics["memory_manager_metrics"] = memory_metrics
                except Exception as e:
                    metrics["memory_manager_error"] = str(e)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting cache metrics: {e}")
            self.metrics.errors += 1
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for consolidated cache service"""
        try:
            health_status = {
                "status": "healthy",
                "cache_service": "available",
                "memory_cache": "available",
                "redis_cache": "not_configured",
                "multi_level": "disabled",
                "memory_manager": "checking"
            }
            
            # Check memory cache
            try:
                test_key = f"health_check_{int(time.time())}"
                await self._set_in_memory(test_key, "test", 1)
                test_result = await self._get_from_memory(test_key)
                if test_result == "test":
                    health_status["memory_cache"] = "healthy"
                else:
                    health_status["memory_cache"] = "degraded"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["memory_cache"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check Redis cache
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status["redis_cache"] = "healthy"
                except Exception as e:
                    health_status["redis_cache"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
            
            # Check multi-level status
            if self.config.multi_level_enabled:
                health_status["multi_level"] = "enabled"
                
                # Check memory manager health
                await self._ensure_memory_manager()
                if self.memory_manager:
                    try:
                        memory_health = await self.memory_manager.health_check()
                        health_status["memory_manager"] = memory_health.get("status", "unknown")
                    except Exception as e:
                        health_status["memory_manager"] = f"error: {str(e)}"
                else:
                    health_status["memory_manager"] = "unavailable"
            
            # Add performance metrics
            health_status["performance"] = {
                "hit_ratio": self.metrics.hit_ratio,
                "total_requests": self.metrics.total_requests,
                "errors": self.metrics.errors,
                "memory_entries": len(self.memory_cache)
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Backward compatibility aliases
SimpleCacheService = ConsolidatedCacheService
CacheOrchestrator = ConsolidatedCacheService