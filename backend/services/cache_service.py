"""
Simple Cache Service
Lightweight caching for query results and expensive operations
"""

import json
import hashlib
import logging
from typing import Any, Optional, Callable, Dict
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class SimpleCacheService:
    """Lightweight caching service for query results"""
    
    def __init__(self, use_redis: bool = False):
        """
        Initialize cache service
        Args:
            use_redis: Whether to use Redis (if available) or in-memory cache
        """
        self.use_redis = use_redis
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None
        
        if use_redis:
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
        return None
    
    async def _set_in_memory(self, cache_key: str, data: Any, ttl_seconds: int):
        """Set in memory cache"""
        expires = datetime.now() + timedelta(seconds=ttl_seconds)
        self.memory_cache[cache_key] = {
            'data': data,
            'expires': expires
        }
    
    async def _get_from_redis(self, cache_key: str) -> Optional[Any]:
        """Get from Redis cache"""
        try:
            cached_result = await self.redis_client.get(cache_key)
            return json.loads(cached_result) if cached_result else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
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
    
    async def get(self, cache_key: str) -> Optional[Any]:
        """Get item from cache"""
        if self.use_redis and self.redis_client:
            return await self._get_from_redis(cache_key)
        else:
            return await self._get_from_memory(cache_key)
    
    async def set(self, cache_key: str, data: Any, ttl_seconds: int = 300):
        """Set item in cache"""
        if self.use_redis and self.redis_client:
            await self._set_in_redis(cache_key, data, ttl_seconds)
        else:
            await self._set_in_memory(cache_key, data, ttl_seconds)
    
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
    
    async def cache_query_result(
        self, 
        query: str, 
        domain: str, 
        result: Any,
        ttl_seconds: int = 300
    ):
        """Cache query result"""
        cache_key = self._generate_cache_key("query", query, domain)
        await self.set(cache_key, result, ttl_seconds)
        logger.debug(f"Cached query result for: {query[:50]}...")
    
    async def get_cached_query_result(
        self, 
        query: str, 
        domain: str
    ) -> Optional[Any]:
        """Get cached query result"""
        cache_key = self._generate_cache_key("query", query, domain)
        result = await self.get(cache_key)
        if result:
            logger.debug(f"Retrieved cached result for: {query[:50]}...")
        return result
    
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
    
    async def clear_cache(self):
        """Clear all cache"""
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
        else:
            self.memory_cache.clear()
            logger.info("Memory cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
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
            # Clean expired entries
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if now >= entry['expires']
            ]
            for key in expired_keys:
                del self.memory_cache[key]
            
            return {
                "cache_type": "memory",
                "keys": len(self.memory_cache),
                "memory_entries": len(self.memory_cache)
            }