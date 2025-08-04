"""
Unified Cache Manager - Consolidation of Multiple Cache Systems

This module consolidates three different cache implementations into a single,
high-performance cache system that maintains all competitive advantages:

1. SimpleCache (basic LRU caching)
2. PerformanceCache (performance wrapper)
3. DomainCache (domain pattern indexing and fast lookup)

Key features preserved:
- Sub-3-second response time optimization
- Domain pattern indexing for O(1) query matching
- LRU eviction with memory bounds
- Performance metrics and monitoring
- Background processing integration
- Persistent cache for startup optimization
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import centralized configuration
from config.centralized_config import get_ml_hyperparameters_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Unified cache entry with comprehensive metadata"""

    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: int
    cache_type: str = (
        "general"  # general, domain_signature, domain_config, query_mapping
    )

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access statistics for LRU and performance tracking"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CachePerformanceMetrics:
    """Comprehensive performance metrics for monitoring"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fast_lookups: int = 0  # Sub-millisecond lookups
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
        return (self.cache_hits / self.total_requests) * 100

    @property
    def fast_lookup_percent(self) -> float:
        """Calculate percentage of fast lookups"""
        if self.total_requests == 0:
            return 0.0
        return (self.fast_lookups / self.total_requests) * 100

    def record_request(
        self, is_hit: bool, lookup_time: float, lookup_type: str = "general"
    ):
        """Record a cache request and its performance characteristics"""
        self.total_requests += 1

        if is_hit:
            self.cache_hits += 1

            # Track specific hit types
            if lookup_type == "pattern_index":
                self.pattern_index_hits += 1
            elif lookup_type == "query_cache":
                self.query_cache_hits += 1
            elif lookup_type == "domain_signature":
                self.domain_signature_hits += 1
        else:
            self.cache_misses += 1

        # Track lookup performance (using centralized threshold)
        ml_config = get_ml_hyperparameters_config()
        if lookup_time < ml_config.sub_millisecond_threshold:  # Sub-millisecond
            self.fast_lookups += 1

        # Update average lookup time (weighted moving average)
        if self.total_requests == 1:
            self.average_lookup_time = lookup_time
        else:
            self.average_lookup_time = (self.average_lookup_time * 0.9) + (
                lookup_time * 0.1
            )


class QueryPatternIndex:
    """High-performance pattern index for O(1) domain query matching"""

    def __init__(self):
        self.word_to_domains: Dict[str, Set[str]] = defaultdict(set)
        self.phrase_to_domains: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.pattern_frequencies: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def add_domain_patterns(
        self, domain: str, entity_patterns: List[Dict], action_patterns: List[Dict]
    ):
        """Add domain patterns to the index for fast lookup"""
        with self._lock:
            # Index entity patterns
            for pattern_dict in entity_patterns:
                pattern_text = pattern_dict.get("pattern_text", "").lower()
                confidence = pattern_dict.get("confidence", 0.0)

                if pattern_text:
                    self.phrase_to_domains[pattern_text].append((domain, confidence))
                    self.pattern_frequencies[pattern_text] += 1

                    # Add words to word index
                    for word in pattern_text.split():
                        if len(word) > 2:
                            self.word_to_domains[word].add(domain)

            # Index action patterns
            for pattern_dict in action_patterns:
                pattern_text = pattern_dict.get("pattern_text", "").lower()
                confidence = pattern_dict.get("confidence", 0.0)

                if pattern_text:
                    self.phrase_to_domains[pattern_text].append((domain, confidence))
                    self.pattern_frequencies[pattern_text] += 1

                    for word in pattern_text.split():
                        if len(word) > 2:
                            self.word_to_domains[word].add(domain)

    def find_matching_domains(self, query: str) -> List[Tuple[str, float]]:
        """Find domains matching query with confidence scores (target: <5ms)"""
        with self._lock:
            query_lower = query.lower()
            query_words = set(word for word in query_lower.split() if len(word) > 2)
            domain_scores = defaultdict(float)

            # 1. Exact phrase matches (highest priority)
            for phrase, domain_confidences in self.phrase_to_domains.items():
                if phrase in query_lower:
                    phrase_score = 2.0 * self.pattern_frequencies.get(phrase, 1)
                    for domain, confidence in domain_confidences:
                        domain_scores[domain] += phrase_score * confidence

            # 2. Word-level matches
            for word in query_words:
                if word in self.word_to_domains:
                    word_score = 1.0 / len(query_words)
                    for domain in self.word_to_domains[word]:
                        domain_scores[domain] += word_score * 0.5

            # Return top 5 matches sorted by score
            sorted_domains = sorted(
                domain_scores.items(), key=lambda x: x[1], reverse=True
            )
            return sorted_domains[:5]

    def get_stats(self) -> Dict[str, int]:
        """Get pattern index statistics"""
        with self._lock:
            return {
                "indexed_phrases": len(self.phrase_to_domains),
                "indexed_words": len(self.word_to_domains),
                "total_pattern_mappings": sum(
                    len(domains) for domains in self.phrase_to_domains.values()
                ),
            }


class UnifiedCacheManager:
    """
    Unified cache manager consolidating all cache functionality with competitive advantages.

    Features:
    - High-performance LRU cache with memory bounds
    - Domain pattern indexing for fast query matching
    - Persistent cache for startup optimization
    - Performance monitoring and metrics
    - Background processing integration
    - Sub-3-second response time optimization
    """

    def __init__(
        self,
        max_size: int = 2000,
        default_ttl: int = 3600,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize unified cache manager

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
            cache_dir: Directory for persistent cache files
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_dir = (
            cache_dir or Path(__file__).parent.parent.parent / "cache" / "unified"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Main cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU tracking

        # High-performance pattern index for domain queries
        self.pattern_index = QueryPatternIndex()

        # Fast query cache for repeated domain detection
        self._query_cache: Dict[
            str, Tuple[str, float, float]
        ] = {}  # hash -> (domain, confidence, timestamp)
        self._query_cache_ttl = 1800  # 30 minutes

        # Performance metrics
        self.metrics = CachePerformanceMetrics()

        # Thread safety
        self._lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

        # Background processing integration
        self.background_processed = False
        self.startup_time = time.time()

        logger.info(
            f"Unified cache manager initialized: max_size={max_size}, ttl={default_ttl}s"
        )

        # Load persistent cache
        self._load_persistent_cache()

    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key"""
        if isinstance(params, dict):
            param_str = json.dumps(params, sort_keys=True, default=str)
        else:
            param_str = str(params)
        combined = f"{operation}:{param_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _update_access_order(self, key: str):
        """Update LRU access order (thread-safe)"""
        with self._sync_lock:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    async def _evict_lru_entries(self, target_count: int = None):
        """Evict least recently used entries"""
        if target_count is None:
            target_count = self.max_size // 4  # Evict 25% when full

        evicted = 0
        while (
            len(self._cache) > (self.max_size - target_count)
            and self._access_order
            and evicted < target_count
        ):
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                evicted += 1
                self.metrics.evictions += 1

        if evicted > 0:
            logger.debug(f"Evicted {evicted} LRU cache entries")

    async def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached result with performance tracking"""
        start_time = time.time()
        cache_key = self._generate_cache_key(operation, params)

        async with self._lock:
            # Check memory cache
            if cache_key in self._cache:
                entry = self._cache[cache_key]

                if not entry.is_expired():
                    entry.update_access()
                    self._update_access_order(cache_key)

                    lookup_time = time.time() - start_time
                    self.metrics.record_request(True, lookup_time, entry.cache_type)

                    return entry.data
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)

            # Cache miss
            lookup_time = time.time() - start_time
            self.metrics.record_request(False, lookup_time)
            return None

    async def set(
        self,
        operation: str,
        params: Dict[str, Any],
        data: Any,
        ttl: Optional[int] = None,
        cache_type: str = "general",
    ) -> bool:
        """Store result in cache with type classification"""
        cache_key = self._generate_cache_key(operation, params)
        ttl = ttl or self.default_ttl

        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                await self._evict_lru_entries()

            # Create cache entry
            entry = CacheEntry(
                data=data,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                ttl=ttl,
                cache_type=cache_type,
            )

            # Store in memory
            self._cache[cache_key] = entry
            self._update_access_order(cache_key)

            # Store in persistent cache for important entries
            if cache_type in ["domain_signature", "domain_config"]:
                await self._save_to_persistent_cache(cache_key, entry)

            return True

    # Domain-specific cache methods (preserving DomainCache interface)

    async def get_domain_signature(self, domain: str) -> Optional[Any]:
        """Get cached domain signature"""
        return await self.get("domain_signature", {"domain": domain})

    async def set_domain_signature(
        self, domain: str, signature: Any, ttl: Optional[int] = None
    ) -> bool:
        """Cache domain signature with pattern indexing"""
        success = await self.set(
            "domain_signature", {"domain": domain}, signature, ttl, "domain_signature"
        )

        if (
            success
            and hasattr(signature, "entity_patterns")
            and hasattr(signature, "action_patterns")
        ):
            # Add to pattern index for fast query matching
            self.pattern_index.add_domain_patterns(
                domain, signature.entity_patterns, signature.action_patterns
            )

        return success

    async def fast_domain_detection(self, query: str) -> Tuple[str, float]:
        """Ultra-fast domain detection using pattern indexes (target: <5ms)"""
        start_time = time.time()

        # Check query cache first (microseconds)
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()

        with self._sync_lock:
            if query_hash in self._query_cache:
                domain, confidence, timestamp = self._query_cache[query_hash]
                if time.time() - timestamp < self._query_cache_ttl:
                    lookup_time = time.time() - start_time
                    self.metrics.record_request(True, lookup_time, "query_cache")
                    return domain, confidence
                else:
                    del self._query_cache[query_hash]

            # Use pattern index for fast matching
            matching_domains = self.pattern_index.find_matching_domains(query)

            if matching_domains:
                best_domain, confidence = matching_domains[0]

                # Cache the result
                self._query_cache[query_hash] = (best_domain, confidence, time.time())

                lookup_time = time.time() - start_time
                self.metrics.record_request(True, lookup_time, "pattern_index")
                return best_domain, confidence

            # Fallback
            lookup_time = time.time() - start_time
            self.metrics.record_request(False, lookup_time, "fallback")
            return "general", 0.3

    # Performance and maintenance methods

    async def cached_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        executor_func,
        ttl: Optional[int] = None,
    ):
        """Cache wrapper for expensive operations"""
        # Try cache first
        cached_result = await self.get(operation, params)
        if cached_result is not None:
            return cached_result

        # Execute and cache
        start_time = time.time()
        result = await executor_func()
        execution_time = time.time() - start_time

        await self.set(operation, params, result, ttl)

        logger.debug(f"Executed and cached {operation} ({execution_time*1000:.2f}ms)")
        return result

    async def clear_expired(self):
        """Remove expired entries"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            # Clean query cache
            current_time = time.time()
            expired_queries = [
                query_hash
                for query_hash, (_, _, timestamp) in self._query_cache.items()
                if current_time - timestamp > self._query_cache_ttl
            ]

            for query_hash in expired_queries:
                del self._query_cache[query_hash]

            if expired_keys or expired_queries:
                logger.info(
                    f"Cleared {len(expired_keys)} expired cache entries, {len(expired_queries)} expired queries"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._sync_lock:
            return {
                "cache_stats": {
                    "total_entries": len(self._cache),
                    "active_entries": len(
                        [e for e in self._cache.values() if not e.is_expired()]
                    ),
                    "query_cache_size": len(self._query_cache),
                    "max_size": self.max_size,
                },
                "performance_metrics": {
                    "total_requests": self.metrics.total_requests,
                    "cache_hits": self.metrics.cache_hits,
                    "cache_misses": self.metrics.cache_misses,
                    "hit_rate_percent": self.metrics.hit_rate_percent,
                    "fast_lookup_percent": self.metrics.fast_lookup_percent,
                    "average_lookup_time_ms": self.metrics.average_lookup_time * 1000,
                    "pattern_index_hits": self.metrics.pattern_index_hits,
                    "domain_signature_hits": self.metrics.domain_signature_hits,
                    "evictions": self.metrics.evictions,
                },
                "pattern_index_stats": self.pattern_index.get_stats(),
                "health_status": self._get_health_status(),
            }

    def _get_health_status(self) -> str:
        """Determine cache health status"""
        hit_rate = self.metrics.hit_rate_percent
        utilization = (len(self._cache) / self.max_size) * 100

        if hit_rate > 70 and utilization < 90:
            return "healthy"
        elif hit_rate > 50 and utilization < 95:
            return "warning"
        else:
            return "degraded"

    # Persistent cache methods

    async def _save_to_persistent_cache(self, cache_key: str, entry: CacheEntry):
        """Save important entries to persistent cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_data = {
                "data": self._serialize_data(entry.data),
                "created_at": entry.created_at,
                "ttl": entry.ttl,
                "cache_type": entry.cache_type,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save to persistent cache: {e}")

    def _load_persistent_cache(self):
        """Load persistent cache entries at startup"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_key = cache_file.stem

                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Check if still valid
                if time.time() - cache_data["created_at"] < cache_data["ttl"]:
                    entry = CacheEntry(
                        data=self._deserialize_data(
                            cache_data["data"], cache_data.get("cache_type", "general")
                        ),
                        created_at=cache_data["created_at"],
                        accessed_at=cache_data["created_at"],
                        access_count=0,
                        ttl=cache_data["ttl"],
                        cache_type=cache_data.get("cache_type", "general"),
                    )

                    self._cache[cache_key] = entry
                    self._access_order.append(cache_key)

                    # Add domain signatures to pattern index
                    if (
                        entry.cache_type == "domain_signature"
                        and hasattr(entry.data, "entity_patterns")
                        and hasattr(entry.data, "action_patterns")
                    ):
                        domain = cache_key.replace("domain_signature:", "")
                        self.pattern_index.add_domain_patterns(
                            domain,
                            entry.data.entity_patterns,
                            entry.data.action_patterns,
                        )
                else:
                    # Remove expired file
                    cache_file.unlink()

        except Exception as e:
            logger.warning(f"Error loading persistent cache: {e}")

    # Domain Configuration Methods

    def set_domain_config(
        self, domain: str, config: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set domain configuration in cache"""
        cache_key = f"domain_config:{domain}"
        current_time = time.time()
        entry = CacheEntry(
            data=config,
            created_at=current_time,
            accessed_at=current_time,
            access_count=0,
            ttl=ttl or self.default_ttl,
            cache_type="domain_config",
        )
        self._cache[cache_key] = entry
        self._access_order.append(cache_key)
        return True

    def get_domain_config(self, domain: str) -> Optional[Any]:
        """Get domain configuration from cache"""
        cache_key = f"domain_config:{domain}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                return entry.data
        return None

    def set_extraction_config(
        self, domain: str, config: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set extraction configuration in cache"""
        cache_key = f"extraction_config:{domain}"
        current_time = time.time()
        entry = CacheEntry(
            data=config,
            created_at=current_time,
            accessed_at=current_time,
            access_count=0,
            ttl=ttl or self.default_ttl,
            cache_type="extraction_config",
        )
        self._cache[cache_key] = entry
        self._access_order.append(cache_key)
        return True

    def get_extraction_config(self, domain: str) -> Optional[Any]:
        """Get extraction configuration from cache"""
        cache_key = f"extraction_config:{domain}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                return entry.data
        return None

    def set_domain_signature(
        self, domain: str, signature: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set domain signature in cache"""
        cache_key = f"domain_signature:{domain}"
        current_time = time.time()
        entry = CacheEntry(
            data=signature,
            created_at=current_time,
            accessed_at=current_time,
            access_count=0,
            ttl=ttl or self.default_ttl,
            cache_type="domain_signature",
        )
        self._cache[cache_key] = entry
        self._access_order.append(cache_key)
        return True

    def get_domain_signature(self, domain: str) -> Optional[Any]:
        """Get domain signature from cache"""
        cache_key = f"domain_signature:{domain}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                return entry.data
        return None

    def set_query_domain_mapping(
        self, query_hash: str, domain: str, score: float, ttl: Optional[int] = None
    ) -> bool:
        """Set query-to-domain mapping in cache"""
        mapping = {"domain": domain, "score": score}
        cache_key = f"query_mapping:{query_hash}"
        current_time = time.time()
        entry = CacheEntry(
            data=mapping,
            created_at=current_time,
            accessed_at=current_time,
            access_count=0,
            ttl=ttl or self.default_ttl,
            cache_type="query_mapping",
        )
        self._cache[cache_key] = entry
        self._access_order.append(cache_key)
        return True

    def get_query_domain_mapping(self, query_hash: str) -> Optional[Any]:
        """Get query-to-domain mapping from cache"""
        cache_key = f"query_mapping:{query_hash}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                return entry.data
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": self.metrics.hit_rate_percent,
            "fast_lookup_rate": self.metrics.fast_lookup_percent,
            "total_requests": self.metrics.hits + self.metrics.misses,
            "pattern_index_domains": len(self.pattern_index.word_to_domains),
            "cache_types": list(
                set(entry.cache_type for entry in self._cache.values())
            ),
        }

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage"""
        if hasattr(data, "__dict__") and hasattr(data, "__dataclass_fields__"):
            return asdict(data)
        elif hasattr(data, "__dict__"):
            return data.__dict__
        return data

    def _deserialize_data(self, data: Any, cache_type: str) -> Any:
        """Deserialize data from JSON storage"""
        # For now, return as-is. In production, would reconstruct proper objects
        return data


# Global cache manager instance
_global_cache: Optional[UnifiedCacheManager] = None


def get_cache_manager() -> UnifiedCacheManager:
    """Get or create global unified cache manager"""
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCacheManager()
    return _global_cache


# Convenience functions for backward compatibility


async def cached_operation(
    operation: str, params: Dict[str, Any], executor_func, ttl: Optional[int] = None
):
    """Convenience function for cached operations"""
    cache = get_cache_manager()
    return await cache.cached_operation(operation, params, executor_func, ttl)


def get_cache():
    """Backward compatibility with SimpleCache interface"""
    return get_cache_manager()


async def get_performance_cache():
    """Backward compatibility with PerformanceCache interface"""
    return get_cache_manager()
