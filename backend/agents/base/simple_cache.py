"""
Simplified Performance Cache for PydanticAI Agent

This module replaces the complex 3-tier cache (HOT/WARM/COLD) with a 
single, efficient cache that maintains performance while reducing complexity.
"""

import asyncio
import hashlib
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with essential metadata"""
    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    
    def is_expired(self, ttl: float) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > ttl
    
    def update_access(self):
        """Update access statistics"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Simple cache statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100


class SimpleCache:
    """
    Simplified single-level cache with LRU eviction.
    
    Replaces the complex HOT/WARM/COLD cache hierarchy with a single
    efficient cache that provides the same performance benefits.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize simple cache
        
        Args:
            max_size: Maximum number of entries to cache
            ttl: Time-to-live in seconds (default 5 minutes)
        """
        self.max_size = max_size
        self.ttl = ttl
        
        # Single cache storage with LRU ordering
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU tracking
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Simple cache initialized (max_size={max_size}, ttl={ttl}s)")
    
    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key"""
        param_str = json.dumps(params, sort_keys=True, default=str)
        combined = f"{operation}:{param_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self.stats.evictions += 1
                logger.debug(f"Evicted LRU entry: {lru_key}")
    
    async def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached result"""
        async with self._lock:
            cache_key = self._generate_cache_key(operation, params)
            self.stats.total_requests += 1
            
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check if expired
                if entry.is_expired(self.ttl):
                    del self._cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
                    self.stats.cache_misses += 1
                    return None
                
                # Update access and return data
                entry.update_access()
                self._update_access_order(cache_key)
                self.stats.cache_hits += 1
                
                logger.debug(f"Cache HIT for {operation}")
                return entry.data
            
            # Cache miss
            self.stats.cache_misses += 1
            logger.debug(f"Cache MISS for {operation}")
            return None
    
    async def set(self, operation: str, params: Dict[str, Any], data: Any) -> bool:
        """Store result in cache"""
        async with self._lock:
            cache_key = self._generate_cache_key(operation, params)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1
            )
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store entry
            self._cache[cache_key] = entry
            self._update_access_order(cache_key)
            
            logger.debug(f"Cached {operation}")
            return True
    
    async def clear_expired(self):
        """Remove expired entries"""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired(self.ttl):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired entries")
    
    async def clear_all(self):
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate_percent": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        hit_rate = self.stats.hit_rate
        size_utilization = (len(self._cache) / self.max_size) * 100
        
        if hit_rate > 70 and size_utilization < 90:
            status = "healthy"
        elif hit_rate > 50 and size_utilization < 95:
            status = "warning"
        else:
            status = "degraded"
        
        return {
            "status": status,
            "hit_rate_percent": hit_rate,
            "size_utilization_percent": size_utilization,
            "cache_size": len(self._cache),
            "performance_acceptable": hit_rate > 50
        }


# Global cache instance
_global_cache: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SimpleCache()
    return _global_cache


async def cached_operation(operation: str, params: Dict[str, Any], executor_func):
    """
    Cache wrapper for expensive operations
    
    Usage:
        result = await cached_operation(
            "tri_modal_search",
            {"query": "test", "domain": "tech"},
            lambda: expensive_search_function(query, domain)
        )
    """
    cache = get_cache()
    
    # Try cache first
    cached_result = await cache.get(operation, params)
    if cached_result is not None:
        return cached_result
    
    # Execute and cache
    start_time = time.time()
    result = await executor_func()
    execution_time = time.time() - start_time
    
    await cache.set(operation, params, result)
    
    logger.debug(f"Executed and cached {operation} ({execution_time*1000:.2f}ms)")
    return result