"""
Unified Memory Manager - Standardized Memory Management

This module consolidates memory management by standardizing on the
SimpleMemoryManager approach while removing complex BoundedMemoryManager
references. Maintains essential memory bounds checking with simplified architecture.

Key features:
- LRU-based memory management with configurable limits
- Essential bounds checking without complex monitoring overhead
- Performance tracking and health monitoring
- Async-compatible interface for PydanticAI integration
- Simplified maintenance and cleanup procedures
"""

import asyncio
import gc
import logging
import time
from dataclasses import dataclass

# Import constants for zero-hardcoded-values compliance
from agents.core.constants import CacheConstants
from typing import Any, Dict, List, Optional

from agents.core.constants import ProcessingConstants
from agents.core.math_expressions import EXPR
from agents.core.data_models import MemoryStatus
from infrastructure.constants import (
    MemoryConstants,
    MLModelConstants,
    ValidationConstants,
)

logger = logging.getLogger(__name__)


class UnifiedMemoryManager:
    """
    Unified memory manager providing essential memory bounds checking
    with simplified architecture and high performance.

    This replaces multiple memory management systems with a single,
    efficient implementation that maintains competitive advantages
    while reducing complexity.
    """

    def __init__(
        self,
        memory_limit_mb: float = ProcessingConstants.DEFAULT_MEMORY_LIMIT_MB,
        cleanup_threshold: float = ProcessingConstants.MEMORY_CLEANUP_THRESHOLD,
    ):
        """
        Initialize unified memory manager

        Args:
            memory_limit_mb: Maximum memory usage in MB
            cleanup_threshold: Utilization threshold to trigger cleanup (0.8 = 80%)
        """
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_target = (
            cleanup_threshold * ProcessingConstants.MEMORY_CLEANUP_THRESHOLD
        )  # Clean down to proportion of threshold

        # Memory tracking
        self._items: Dict[str, Any] = {}
        self._item_sizes: Dict[str, int] = {}
        self._item_categories: Dict[str, str] = {}
        self._access_order: List[str] = []  # LRU tracking
        self._access_times: Dict[str, float] = {}

        # Status tracking
        self.status = MemoryStatus(memory_limit_mb=memory_limit_mb)

        # Performance metrics
        self.metrics = {
            "total_stores": 0,
            "total_retrievals": 0,
            "total_evictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_store_time": CacheConstants.ZERO_FLOAT,
            "avg_retrieval_time": CacheConstants.ZERO_FLOAT,
        }

        # Thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"Unified memory manager initialized: limit={memory_limit_mb}MB, threshold={cleanup_threshold}"
        )

    def _estimate_item_size(self, item: Any) -> int:
        """Estimate item size in bytes using efficient heuristics"""
        try:
            if isinstance(item, str):
                return EXPR.estimate_string_size(item)
            elif isinstance(item, bytes):
                return len(item)
            elif isinstance(item, (int, float)):
                return MLModelConstants.NUMBER_SIZE_BYTES  # 8 bytes for numbers
            elif isinstance(item, bool):
                return MLModelConstants.BOOLEAN_SIZE_BYTES
            elif isinstance(item, (list, tuple)):
                # Sample first 10 items for performance
                sample_size = sum(self._estimate_item_size(x) for x in item[:10])
                return EXPR.estimate_sample_size(len(item), sample_size)
            elif isinstance(item, dict):
                # Sample first 10 key-value pairs
                sample_items = list(item.items())[:10]
                sample_size = sum(
                    self._estimate_item_size(k) + self._estimate_item_size(v)
                    for k, v in sample_items
                )
                return EXPR.estimate_sample_size(len(item), sample_size)
            else:
                # Default estimate for objects
                return MLModelConstants.DEFAULT_OBJECT_SIZE_BYTES  # 1KB default
        except Exception:
            return MLModelConstants.DEFAULT_OBJECT_SIZE_BYTES  # Fallback estimate

    def _update_access_order(self, key: str):
        """Update LRU access order efficiently"""
        # Remove from current position
        if key in self._access_order:
            self._access_order.remove(key)

        # Add to end (most recently used)
        self._access_order.append(key)
        self._access_times[key] = time.time()

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed based on utilization"""
        return self.status.utilization_percent >= (
            self.cleanup_threshold * CacheConstants.PERCENTAGE_MULTIPLIER
        )

    async def _perform_cleanup(self) -> int:
        """Perform LRU-based cleanup to free memory"""
        if not self._access_order:
            return 0

        current_usage = sum(self._item_sizes.values())
        target_usage = int(current_usage * self.cleanup_target)

        items_removed = 0
        safety_limit = len(self._items) // 2  # Never remove more than 50%

        while (
            sum(self._item_sizes.values()) > target_usage
            and self._access_order
            and items_removed < safety_limit
        ):
            # Remove least recently used item
            lru_key = self._access_order.pop(0)

            if lru_key in self._items:
                del self._items[lru_key]
                del self._item_sizes[lru_key]
                del self._item_categories[lru_key]
                if lru_key in self._access_times:
                    del self._access_times[lru_key]

                items_removed += 1
                self.metrics["total_evictions"] += 1

        if items_removed > 0:
            self._update_status()
            logger.debug(f"Memory cleanup: removed {items_removed} items")

        return items_removed

    def _update_status(self):
        """Update memory status and metrics"""
        self.status.total_items = len(self._items)
        self.status.estimated_usage_mb = EXPR.bytes_to_mb(
            sum(self._item_sizes.values())
        )
        self.status.evictions = self.metrics["total_evictions"]
        self.status.last_cleanup = time.time()
        self.status.update_health_status()

    async def store_item(self, key: str, item: Any, category: str = "default") -> bool:
        """
        Store item with memory bounds checking and performance tracking

        Args:
            key: Unique identifier for the item
            item: Data to store
            category: Item category for organization and prioritization

        Returns:
            True if stored successfully, False if rejected due to memory limits
        """
        start_time = time.time()

        async with self._lock:
            try:
                # Estimate item size
                item_size = self._estimate_item_size(item)

                # Check if storing would exceed memory limits
                current_usage = sum(self._item_sizes.values())
                projected_usage_mb = EXPR.calculate_projected_usage_mb(
                    current_usage, item_size
                )

                if projected_usage_mb > self.memory_limit_mb:
                    # Try cleanup first
                    if self._should_cleanup():
                        await self._perform_cleanup()

                    # Check again after cleanup
                    current_usage = sum(self._item_sizes.values())
                    projected_usage_mb = EXPR.calculate_projected_usage_mb(
                        current_usage, item_size
                    )

                    if projected_usage_mb > self.memory_limit_mb:
                        logger.warning(
                            f"Cannot store item {key}: would exceed memory limit"
                        )
                        return False

                # Store the item
                self._items[key] = item
                self._item_sizes[key] = item_size
                self._item_categories[key] = category
                self._update_access_order(key)

                # Update status
                self._update_status()

                # Perform cleanup if needed
                if self._should_cleanup():
                    await self._perform_cleanup()

                # Update metrics
                store_time = time.time() - start_time
                self.metrics["total_stores"] += 1
                self.metrics["avg_store_time"] = (
                    self.metrics["avg_store_time"] * (self.metrics["total_stores"] - 1)
                    + store_time
                ) / self.metrics["total_stores"]

                return True

            except Exception as e:
                logger.error(f"Error storing item {key}: {e}")
                return False

    async def retrieve_item(self, key: str, category: str = "default") -> Optional[Any]:
        """
        Retrieve item by key with performance tracking

        Args:
            key: Item identifier
            category: Item category (for compatibility, not used in lookup)

        Returns:
            Stored item or None if not found
        """
        start_time = time.time()

        async with self._lock:
            try:
                if key in self._items:
                    # Update access order for LRU
                    self._update_access_order(key)

                    # Update metrics
                    retrieval_time = time.time() - start_time
                    self.metrics["total_retrievals"] += 1
                    self.metrics["cache_hits"] += 1
                    self.metrics["avg_retrieval_time"] = (
                        self.metrics["avg_retrieval_time"]
                        * (self.metrics["total_retrievals"] - 1)
                        + retrieval_time
                    ) / self.metrics["total_retrievals"]

                    return self._items[key]
                else:
                    # Cache miss
                    self.metrics["total_retrievals"] += 1
                    self.metrics["cache_misses"] += 1
                    return None

            except Exception as e:
                logger.error(f"Error retrieving item {key}: {e}")
                return None

    async def remove_item(self, key: str) -> bool:
        """Remove item by key"""
        async with self._lock:
            if key in self._items:
                del self._items[key]
                del self._item_sizes[key]
                del self._item_categories[key]

                if key in self._access_order:
                    self._access_order.remove(key)
                if key in self._access_times:
                    del self._access_times[key]

                self._update_status()
                return True

            return False

    async def clear_all(self):
        """Clear all stored items and reset state"""
        async with self._lock:
            self._items.clear()
            self._item_sizes.clear()
            self._item_categories.clear()
            self._access_order.clear()
            self._access_times.clear()

            self._update_status()

            # Force garbage collection
            gc.collect()

            logger.info("All items cleared from unified memory manager")

    async def clear_category(self, category: str) -> int:
        """Clear all items in a specific category"""
        async with self._lock:
            keys_to_remove = [
                key for key, cat in self._item_categories.items() if cat == category
            ]

            for key in keys_to_remove:
                await self.remove_item(key)

            logger.info(
                f"Cleared {len(keys_to_remove)} items from category '{category}'"
            )
            return len(keys_to_remove)

    async def perform_maintenance(self):
        """Perform comprehensive maintenance tasks"""
        async with self._lock:
            initial_items = len(self._items)

            # Cleanup if needed
            if self._should_cleanup():
                removed = await self._perform_cleanup()
                logger.info(f"Maintenance cleanup: removed {removed} items")

            # Force garbage collection
            gc.collect()

            # Update status
            self._update_status()

            final_items = len(self._items)
            if initial_items != final_items:
                logger.info(
                    f"Maintenance complete: {initial_items - final_items} items cleaned"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory management statistics"""
        self._update_status()

        # Calculate cache hit rate
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = EXPR.calculate_hit_rate(self.metrics["cache_hits"], total_requests)

        return {
            "memory_status": {
                "total_items": self.status.total_items,
                "estimated_usage_mb": self.status.estimated_usage_mb,
                "memory_limit_mb": self.status.memory_limit_mb,
                "utilization_percent": self.status.utilization_percent,
                "health_status": self.status.health_status,
            },
            "performance_metrics": {
                "total_stores": self.metrics["total_stores"],
                "total_retrievals": self.metrics["total_retrievals"],
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "hit_rate_percent": hit_rate,
                "total_evictions": self.metrics["total_evictions"],
                "avg_store_time_ms": self.metrics["avg_store_time"]
                * CacheConstants.MS_PER_SECOND,
                "avg_retrieval_time_ms": self.metrics["avg_retrieval_time"]
                * CacheConstants.MS_PER_SECOND,
            },
            "category_breakdown": self._get_category_breakdown(),
            "cleanup_configuration": {
                "cleanup_threshold_percent": self.cleanup_threshold
                * CacheConstants.PERCENTAGE_MULTIPLIER,
                "cleanup_target_percent": self.cleanup_target
                * CacheConstants.PERCENTAGE_MULTIPLIER,
            },
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring systems"""
        self._update_status()

        return {
            "status": self.status.health_status,
            "memory_healthy": self.status.utilization_percent
            < ValidationConstants.MEMORY_HEALTH_THRESHOLD_PERCENT,
            "utilization_percent": self.status.utilization_percent,
            "total_items": self.status.total_items,
            "within_limits": self.status.estimated_usage_mb < self.memory_limit_mb,
            "last_cleanup": self.status.last_cleanup,
            "performance_acceptable": self.metrics["avg_retrieval_time"]
            < MemoryConstants.ACCEPTABLE_RETRIEVAL_TIME_SECONDS,  # <10ms acceptable
        }

    def _get_category_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get breakdown of items by category"""
        category_stats = {}

        for key, category in self._item_categories.items():
            if category not in category_stats:
                category_stats[category] = {
                    "item_count": 0,
                    "total_size_bytes": 0,
                    "avg_size_bytes": 0,
                }

            category_stats[category]["item_count"] += 1
            size = self._item_sizes.get(key, 0)
            category_stats[category]["total_size_bytes"] += size

        # Calculate averages
        for category, stats in category_stats.items():
            if stats["item_count"] > 0:
                stats["avg_size_bytes"] = (
                    stats["total_size_bytes"] / stats["item_count"]
                )

        return category_stats

    def get_lru_info(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get information about least recently used items"""
        lru_info = []

        for key in self._access_order[:limit]:  # First items are LRU
            if key in self._items:
                lru_info.append(
                    {
                        "key": key,
                        "category": self._item_categories.get(key, "unknown"),
                        "size_bytes": self._item_sizes.get(key, 0),
                        "last_access": self._access_times.get(key, 0),
                        "age_seconds": time.time()
                        - self._access_times.get(key, time.time()),
                    }
                )

        return lru_info


# Global memory manager instance
_global_memory_manager: Optional[UnifiedMemoryManager] = None


async def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create global unified memory manager instance"""
    global _global_memory_manager

    if _global_memory_manager is None:
        _global_memory_manager = UnifiedMemoryManager()

    return _global_memory_manager


async def shutdown_memory_manager():
    """Shutdown and cleanup global memory manager"""
    global _global_memory_manager

    if _global_memory_manager:
        await _global_memory_manager.clear_all()
        _global_memory_manager = None
        logger.info("Unified memory manager shutdown complete")


# Convenience functions for common operations
async def store_pattern(key: str, pattern: Any) -> bool:
    """Store a pattern with unified memory management"""
    manager = await get_memory_manager()
    return await manager.store_item(f"pattern:{key}", pattern, "pattern")


async def retrieve_pattern(key: str) -> Optional[Any]:
    """Retrieve a pattern from unified memory"""
    manager = await get_memory_manager()
    return await manager.retrieve_item(f"pattern:{key}", "pattern")


async def store_result(key: str, result: Any) -> bool:
    """Store a result with unified memory management"""
    manager = await get_memory_manager()
    return await manager.store_item(f"result:{key}", result, "result")


async def retrieve_result(key: str) -> Optional[Any]:
    """Retrieve a result from unified memory"""
    manager = await get_memory_manager()
    return await manager.retrieve_item(f"result:{key}", "result")


async def store_domain_data(key: str, data: Any) -> bool:
    """Store domain-related data with unified memory management"""
    manager = await get_memory_manager()
    return await manager.store_item(f"domain:{key}", data, "domain")


async def retrieve_domain_data(key: str) -> Optional[Any]:
    """Retrieve domain-related data from unified memory"""
    manager = await get_memory_manager()
    return await manager.retrieve_item(f"domain:{key}", "domain")
