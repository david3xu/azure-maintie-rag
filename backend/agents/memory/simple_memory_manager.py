"""
Simplified Memory Management System for PydanticAI Agent

This module replaces the complex memory monitoring and multi-cache system
with a simple, efficient memory manager that maintains essential bounds checking.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimpleMemoryStats:
    """Simplified memory statistics"""
    total_items: int = 0
    memory_limit_mb: float = 200.0
    estimated_usage_mb: float = 0.0
    evictions: int = 0
    last_cleanup: float = 0.0
    
    @property
    def utilization_percent(self) -> float:
        """Calculate memory utilization percentage"""
        if self.memory_limit_mb == 0:
            return 0.0
        return min(100.0, (self.estimated_usage_mb / self.memory_limit_mb) * 100)
    
    @property
    def health_status(self) -> str:
        """Get health status based on utilization"""
        if self.utilization_percent > 90:
            return "critical"
        elif self.utilization_percent > 75:
            return "warning"
        else:
            return "healthy"


class SimpleMemoryManager:
    """
    Simplified memory manager that provides essential memory bounds checking
    without complex monitoring overhead.
    
    Replaces the complex BoundedMemoryManager with a lightweight alternative
    that maintains memory limits while reducing complexity.
    """
    
    def __init__(self, memory_limit_mb: float = 200.0):
        """
        Initialize simple memory manager
        
        Args:
            memory_limit_mb: Maximum memory usage in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.stats = SimpleMemoryStats(memory_limit_mb=memory_limit_mb)
        
        # Simple item tracking (replaces complex multi-cache system)
        self._items: Dict[str, Any] = {}
        self._item_sizes: Dict[str, int] = {}
        self._access_order: list = []  # LRU tracking
        
        # Cleanup configuration
        self.cleanup_threshold = 0.8  # Cleanup at 80% capacity
        self.cleanup_target = 0.6     # Clean down to 60% capacity
        
        logger.info(f"Simple memory manager initialized (limit: {memory_limit_mb}MB)")
    
    def _estimate_item_size(self, item: Any) -> int:
        """Estimate item size in bytes (simplified approach)"""
        try:
            # Simple size estimation without complex serialization
            if isinstance(item, str):
                return len(item.encode('utf-8'))
            elif isinstance(item, (list, tuple)):
                return sum(self._estimate_item_size(x) for x in item[:10])  # Sample first 10
            elif isinstance(item, dict):
                return sum(
                    self._estimate_item_size(k) + self._estimate_item_size(v)
                    for k, v in list(item.items())[:10]  # Sample first 10
                )
            else:
                return 1024  # Default 1KB estimate
        except:
            return 1024
    
    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _cleanup_if_needed(self):
        """Perform cleanup if memory usage exceeds threshold"""
        if self.stats.utilization_percent < (self.cleanup_threshold * 100):
            return
        
        # Calculate how many items to remove
        current_size = sum(self._item_sizes.values())
        target_size = int(current_size * self.cleanup_target)
        
        # Remove LRU items until we reach target
        items_removed = 0
        while (sum(self._item_sizes.values()) > target_size and 
               self._access_order and 
               items_removed < len(self._items) // 2):  # Safety limit
            
            lru_key = self._access_order.pop(0)
            if lru_key in self._items:
                del self._items[lru_key]
                del self._item_sizes[lru_key]
                items_removed += 1
                self.stats.evictions += 1
        
        if items_removed > 0:
            self._update_stats()
            logger.debug(f"Memory cleanup: removed {items_removed} items")
    
    def _update_stats(self):
        """Update memory statistics"""
        self.stats.total_items = len(self._items)
        self.stats.estimated_usage_mb = sum(self._item_sizes.values()) / (1024 * 1024)
        self.stats.last_cleanup = time.time()
    
    async def store_item(self, key: str, item: Any, category: str = "default") -> bool:
        """
        Store item with memory bounds checking
        
        Args:
            key: Unique identifier for the item
            item: Data to store
            category: Item category (for compatibility, ignored in simple version)
        
        Returns:
            True if stored successfully
        """
        try:
            # Estimate size
            item_size = self._estimate_item_size(item)
            
            # Check if adding this item would exceed limits
            projected_usage = (sum(self._item_sizes.values()) + item_size) / (1024 * 1024)
            if projected_usage > self.memory_limit_mb:
                # Try cleanup first
                self._cleanup_if_needed()
                
                # Check again after cleanup
                projected_usage = (sum(self._item_sizes.values()) + item_size) / (1024 * 1024)
                if projected_usage > self.memory_limit_mb:
                    logger.warning(f"Cannot store item {key}: would exceed memory limit")
                    return False
            
            # Store item
            self._items[key] = item
            self._item_sizes[key] = item_size
            self._update_access_order(key)
            
            # Update stats
            self._update_stats()
            
            # Perform cleanup if needed
            self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing item {key}: {e}")
            return False
    
    async def retrieve_item(self, key: str, category: str = "default") -> Optional[Any]:
        """
        Retrieve item by key
        
        Args:
            key: Item identifier
            category: Item category (for compatibility, ignored in simple version)
        
        Returns:
            Stored item or None if not found
        """
        if key in self._items:
            # Update access order for LRU
            self._update_access_order(key)
            return self._items[key]
        
        return None
    
    async def remove_item(self, key: str) -> bool:
        """Remove item by key"""
        if key in self._items:
            del self._items[key]
            del self._item_sizes[key]
            if key in self._access_order:
                self._access_order.remove(key)
            
            self._update_stats()
            return True
        
        return False
    
    async def clear_all(self):
        """Clear all stored items"""
        self._items.clear()
        self._item_sizes.clear() 
        self._access_order.clear()
        self._update_stats()
        logger.info("All items cleared from memory manager")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        self._update_stats()
        
        return {
            "total_items": self.stats.total_items,
            "estimated_usage_mb": self.stats.estimated_usage_mb,
            "memory_limit_mb": self.stats.memory_limit_mb,
            "utilization_percent": self.stats.utilization_percent,
            "health_status": self.stats.health_status,
            "evictions": self.stats.evictions,
            "last_cleanup": self.stats.last_cleanup
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring"""
        stats = self.get_stats()
        
        return {
            "status": stats["health_status"],
            "memory_healthy": stats["utilization_percent"] < 85,
            "utilization_percent": stats["utilization_percent"],
            "total_items": stats["total_items"],
            "within_limits": stats["estimated_usage_mb"] < self.memory_limit_mb
        }
    
    async def perform_maintenance(self):
        """Perform maintenance tasks (simplified cleanup)"""
        initial_items = len(self._items)
        self._cleanup_if_needed()
        final_items = len(self._items)
        
        if initial_items != final_items:
            logger.info(f"Maintenance: {initial_items - final_items} items cleaned up")


# Global memory manager instance  
_global_memory_manager: Optional[SimpleMemoryManager] = None


async def get_memory_manager() -> SimpleMemoryManager:
    """Get or create global memory manager instance"""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = SimpleMemoryManager()
    
    return _global_memory_manager


async def shutdown_memory_manager():
    """Shutdown global memory manager"""
    global _global_memory_manager
    
    if _global_memory_manager:
        await _global_memory_manager.clear_all()
        _global_memory_manager = None
        logger.info("Memory manager shutdown complete")


# Convenience functions for common operations
async def store_pattern(key: str, pattern: Any) -> bool:
    """Store a pattern with memory management"""
    manager = await get_memory_manager()
    return await manager.store_item(f"pattern:{key}", pattern, "pattern")


async def retrieve_pattern(key: str) -> Optional[Any]:
    """Retrieve a pattern from memory"""
    manager = await get_memory_manager()
    return await manager.retrieve_item(f"pattern:{key}", "pattern")


async def store_result(key: str, result: Any) -> bool:
    """Store a result with memory management"""
    manager = await get_memory_manager()
    return await manager.store_item(f"result:{key}", result, "result")


async def retrieve_result(key: str) -> Optional[Any]:
    """Retrieve a result from memory"""
    manager = await get_memory_manager()
    return await manager.retrieve_item(f"result:{key}", "result")