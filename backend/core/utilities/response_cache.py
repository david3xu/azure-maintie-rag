"""
Simple, professional response caching with Redis
Builds on existing architecture - no over-engineering
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.models.maintenance_models import RAGResponse
from config.settings import settings

logger = logging.getLogger(__name__)

class ResponseCache:
    """Simple response caching for RAG queries"""

    def __init__(self):
        """Initialize cache with Redis or fallback to memory"""
        self.redis_client = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.use_redis = False

        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_memory_cache_size = 1000

        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection with fallback"""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available, using memory cache")
            return

        try:
            # Try to connect to Redis
            self.redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_DB', 0),
                decode_responses=True,
                socket_timeout=5
            )

            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache")
            self.redis_client = None
            self.use_redis = False

    def get_cache_key(self, query: str, max_results: int = 10) -> str:
        """Generate cache key for query"""
        # Normalize query for consistent caching
        normalized_query = query.strip().lower()

        # Create hash from query and parameters
        key_data = f"{normalized_query}::{max_results}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()

        return f"rag_response:{cache_key}"

    def get_cached_response(self, query: str, max_results: int = 10) -> Optional[RAGResponse]:
        """Get cached response if available"""
        cache_key = self.get_cache_key(query, max_results)

        try:
            if self.use_redis and self.redis_client:
                # Try Redis first
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    response_data = json.loads(cached_data)
                    logger.info(f"Cache hit (Redis): {cache_key[:12]}...")
                    rag_response = self._deserialize_response(response_data)
                    # Auto-clear broken cache (enhanced_query is None)
                    if getattr(rag_response, 'enhanced_query', None) is None:
                        logger.warning(f"Auto-clearing broken cache entry: {cache_key[:12]} (Redis)")
                        self.redis_client.delete(cache_key)
                        return None
                    return rag_response

            else:
                # Fallback to memory cache
                if cache_key in self.memory_cache:
                    cache_entry = self.memory_cache[cache_key]

                    # Check expiration
                    if datetime.now() < cache_entry['expires_at']:
                        logger.info(f"Cache hit (Memory): {cache_key[:12]}...")
                        rag_response = self._deserialize_response(cache_entry['data'])
                        # Auto-clear broken cache (enhanced_query is None)
                        if getattr(rag_response, 'enhanced_query', None) is None:
                            logger.warning(f"Auto-clearing broken cache entry: {cache_key[:12]} (Memory)")
                            del self.memory_cache[cache_key]
                            return None
                        return rag_response
                    else:
                        # Remove expired entry
                        del self.memory_cache[cache_key]

        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")

        return None

    def cache_response(self, query: str, response: RAGResponse,
                      max_results: int = 10, ttl: Optional[int] = None) -> bool:
        """Cache response for future use"""
        cache_key = self.get_cache_key(query, max_results)
        ttl = ttl or self.default_ttl

        try:
            # Serialize response
            response_data = self._serialize_response(response)

            if self.use_redis and self.redis_client:
                # Cache in Redis
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(response_data)
                )
                logger.debug(f"Response cached (Redis): {cache_key[:12]}...")

            else:
                # Cache in memory with size limit
                if len(self.memory_cache) >= self.max_memory_cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.memory_cache.keys(),
                                   key=lambda k: self.memory_cache[k]['cached_at'])
                    del self.memory_cache[oldest_key]

                self.memory_cache[cache_key] = {
                    'data': response_data,
                    'cached_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(seconds=ttl)
                }
                logger.debug(f"Response cached (Memory): {cache_key[:12]}...")

            return True

        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False

    def _serialize_response(self, response: RAGResponse) -> Dict[str, Any]:
        """Convert RAGResponse to cacheable format"""
        return {
            'query': response.query,
            'generated_response': response.generated_response,
            'confidence_score': response.confidence_score,
            'processing_time': response.processing_time,
            'sources': response.sources or [],  # sources is List[str], not List[Dict]
            'safety_warnings': response.safety_warnings or [],
            'citations': response.citations or [],
            'cached_at': datetime.now().isoformat()
        }

    def _deserialize_response(self, data: Dict[str, Any]) -> RAGResponse:
        """Convert cached data back to RAGResponse"""
        # Note: This creates a simplified RAGResponse from cache
        # Some fields may not be fully restored
        from core.models.maintenance_models import RAGResponse

        return RAGResponse(
            query=data['query'],
            enhanced_query=None,  # Not cached for simplicity
            search_results=[],    # Not cached for simplicity
            generated_response=data['generated_response'],
            confidence_score=data['confidence_score'],
            processing_time=data['processing_time'],
            sources=data.get('sources', []),  # sources is List[str]
            safety_warnings=data.get('safety_warnings', []),
            citations=data.get('citations', [])
        )

    def clear_cache(self) -> bool:
        """Clear all cached responses"""
        try:
            if self.use_redis and self.redis_client:
                # Clear Redis cache
                keys = self.redis_client.keys("rag_response:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached responses from Redis")

            # Clear memory cache
            cache_size = len(self.memory_cache)
            self.memory_cache.clear()
            logger.info(f"Cleared {cache_size} cached responses from memory")

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cache_type': 'redis' if self.use_redis else 'memory',
            'redis_available': REDIS_AVAILABLE,
            'redis_connected': self.use_redis
        }

        try:
            if self.use_redis and self.redis_client:
                # Redis stats
                keys = self.redis_client.keys("rag_response:*")
                stats['cached_responses'] = len(keys)

            else:
                # Memory cache stats
                stats['cached_responses'] = len(self.memory_cache)
                stats['max_cache_size'] = self.max_memory_cache_size

        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            stats['error'] = str(e)

        return stats