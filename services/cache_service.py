"""
Simple Cache Service - CODING_STANDARDS Compliant
Clean caching service without over-engineering enterprise patterns.
"""

import hashlib
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SimpleCacheService:
    """
    Simple cache service following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses simple key-value storage
    - Universal Design: Works with any data type
    - Mathematical Foundation: Simple TTL calculations
    """

    def __init__(self, default_ttl: int = 300, max_entries: int = 1000):
        """Initialize simple cache service"""
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
        
        logger.info(f"Simple cache service initialized (TTL: {default_ttl}s, max: {max_entries})")

    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.timestamps.items():
            if current_time - timestamp > self.default_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)

    def _make_room(self) -> None:
        """Make room for new entries if cache is full"""
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=self.timestamps.get)
            self.cache.pop(oldest_key, None)
            self.timestamps.pop(oldest_key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            # Clean up expired entries
            self._cleanup_expired()
            
            if key in self.cache:
                self.stats["hits"] += 1
                # Update timestamp
                self.timestamps[key] = time.time()
                return self.cache[key]
            else:
                self.stats["misses"] += 1
                return default
                
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self.stats["misses"] += 1
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # Clean up expired entries first
            self._cleanup_expired()
            
            # Make room if needed
            self._make_room()
            
            # Set value
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.stats["sets"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.cache.clear()
            self.timestamps.clear()
            return True
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(1, total_requests)) * 100
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "max_entries": self.max_entries
        }

    def hash_key(self, data: Any) -> str:
        """Create hash key from data"""
        try:
            if isinstance(data, str):
                content = data
            elif isinstance(data, dict):
                content = str(sorted(data.items()))
            else:
                content = str(data)
                
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Hash key creation failed: {e}")
            return f"key_{time.time()}"

    def cache_result(self, key: str, func, *args, **kwargs) -> Any:
        """Cache function result"""
        try:
            # Check cache first
            cached_result = self.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
            
        except Exception as e:
            logger.error(f"Cache result failed for key {key}: {e}")
            # Execute function without caching if cache fails
            return func(*args, **kwargs)

    async def cache_async_result(self, key: str, func, *args, **kwargs) -> Any:
        """Cache async function result"""
        try:
            # Check cache first
            cached_result = self.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute async function and cache result
            result = await func(*args, **kwargs)
            self.set(key, result)
            return result
            
        except Exception as e:
            logger.error(f"Async cache result failed for key {key}: {e}")
            # Execute function without caching if cache fails
            return await func(*args, **kwargs)


# Backward compatibility - Global instance
_cache_service = SimpleCacheService()

# Backward compatibility functions
def get_cache(key: str, default: Any = None) -> Any:
    """Backward compatibility function"""
    return _cache_service.get(key, default)

def set_cache(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Backward compatibility function"""
    return _cache_service.set(key, value, ttl)

def delete_cache(key: str) -> bool:
    """Backward compatibility function"""
    return _cache_service.delete(key)

def clear_cache() -> bool:
    """Backward compatibility function"""
    return _cache_service.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Backward compatibility function"""
    return _cache_service.get_stats()

def hash_cache_key(data: Any) -> str:
    """Backward compatibility function"""
    return _cache_service.hash_key(data)

# Backward compatibility aliases
CacheService = SimpleCacheService
ConsolidatedCacheService = SimpleCacheService