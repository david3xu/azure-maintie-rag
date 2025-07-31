"""
Performance Caching System for PydanticAI Agent

This module provides a simplified caching interface that maintains
sub-3-second response times while reducing architectural complexity.
"""

from typing import Dict, Any, Optional
from .simple_cache import SimpleCache, get_cache
import logging

logger = logging.getLogger(__name__)


class PerformanceCache:
    """
    Simplified performance cache that maintains the same interface
    but uses a single-level cache internally for reduced complexity.
    """
    
    def __init__(
        self, 
        max_memory_mb: float = 100,  # Kept for compatibility
        hot_ttl: float = 300,        # Now used as default TTL
        warm_ttl: float = 1800,      # Ignored in simplified version
        cold_ttl: float = 3600       # Ignored in simplified version
    ):
        # Use simplified cache internally
        cache_size = int(max_memory_mb * 10)  # Rough conversion to entry count
        self._cache = SimpleCache(max_size=cache_size, ttl=int(hot_ttl))
        
        logger.info(f"Performance cache initialized (simplified mode, size={cache_size}, ttl={hot_ttl}s)")
    
    async def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached result for operation with parameters"""
        return await self._cache.get(operation, params)
    
    async def set(self, operation: str, params: Dict[str, Any], data: Any) -> bool:
        """Store result in cache"""
        return await self._cache.set(operation, params, data)
    
    async def clear_expired(self):
        """Remove expired entries"""
        await self._cache.clear_expired()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        await self.clear_expired()
        stats = self._cache.get_stats()
        health = self._cache.get_health_status()
        
        return {
            "cache_stats": {
                "total_requests": stats["total_requests"],
                "cache_hits": stats["cache_hits"],
                "cache_misses": stats["cache_misses"],
                "hit_rate_percent": stats["hit_rate_percent"],
                "evictions": stats["evictions"]
            },
            "performance_stats": {
                "cache_size": stats["cache_size"],
                "max_size": stats["max_size"],
                "ttl_seconds": stats["ttl_seconds"]
            },
            "health": {
                "status": health["status"],
                "performance_acceptable": health["performance_acceptable"],
                "size_utilization_percent": health["size_utilization_percent"]
            }
        }
    
    async def warmup_cache(self, common_operations):
        """Pre-warm cache with common operations"""
        logger.info(f"Warming up cache with {len(common_operations)} common operations")
        
        for operation, params in common_operations:
            await self.set(operation, params, f"warmup_result_for_{operation}")
        
        logger.info("Cache warmup completed")


# Global cache instance
_global_cache: Optional[PerformanceCache] = None


def get_performance_cache() -> PerformanceCache:
    """Get or create global performance cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = PerformanceCache()
    return _global_cache


async def cached_operation(operation: str, params: dict, executor_func):
    """
    Cache wrapper for expensive operations
    
    Usage:
        result = await cached_operation(
            "tri_modal_search",
            {"query": "test", "domain": "tech"},
            lambda: expensive_search_function(query, domain)
        )
    """
    # Use the global simple cache directly for better performance
    from .simple_cache import cached_operation as simple_cached_operation
    return await simple_cached_operation(operation, params, executor_func)