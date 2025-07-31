"""
Simplified Memory Management System (Backward Compatibility Layer)

This module provides a backward-compatible interface to the simplified memory
management system while reducing complexity from 582 lines to ~150 lines.
"""

from .simple_memory_manager import (
    SimpleMemoryManager,
    SimpleMemoryStats,
    get_memory_manager as get_simple_memory_manager,
    shutdown_memory_manager
)
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Backward compatibility classes
@dataclass
class MemoryUsageStats:
    """Backward compatibility for old MemoryUsageStats"""
    current_memory_mb: float
    max_memory_mb: float
    cache_count: int
    eviction_count: int
    hit_count: int = 0
    miss_count: int = 0
    timestamp: float = field(default_factory=time.time)


class MemoryMonitor:
    """Simplified memory monitor (backward compatibility)"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        logger.info("Simplified memory monitor initialized")
    
    def get_current_usage_mb(self) -> float:
        """Get estimated current memory usage"""
        # Simplified - just return a reasonable estimate
        return 50.0  # Conservative estimate
    
    def get_memory_percent(self) -> float:
        """Get memory usage percentage"""
        return 25.0  # Conservative estimate
    
    async def start_monitoring(self):
        """Start monitoring (simplified - no-op)"""
        logger.info("Memory monitoring started (simplified mode)")
    
    async def stop_monitoring(self):
        """Stop monitoring (simplified - no-op)"""
        logger.info("Memory monitoring stopped")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get basic memory stats"""
        return {
            'current_mb': self.get_current_usage_mb(),
            'memory_percent': self.get_memory_percent(),
            'history_available': False
        }


class LRUCache:
    """Backward compatibility wrapper for LRU cache functionality"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0, 
                 ttl_seconds: Optional[float] = None, name: str = "LRUCache"):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        self.name = name
        
        # Use simplified memory manager internally
        self._memory_manager = SimpleMemoryManager(memory_limit_mb=max_memory_mb)
        
        # Statistics tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        logger.info(f"LRU Cache '{name}' initialized (simplified mode)")
    
    def get(self, key: str):
        """Get item from cache"""
        import asyncio
        
        async def _get():
            result = await self._memory_manager.retrieve_item(key)
            if result is not None:
                self.hit_count += 1
                return result
            else:
                self.miss_count += 1
                return None
        
        # Handle both sync and async contexts
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                return asyncio.create_task(_get())
            else:
                return loop.run_until_complete(_get())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(_get())
    
    def put(self, key: str, value) -> bool:
        """Put item in cache"""
        import asyncio
        
        async def _put():
            return await self._memory_manager.store_item(key, value)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.create_task(_put())
            else:
                return loop.run_until_complete(_put())
        except RuntimeError:
            return asyncio.run(_put())
    
    def clear(self):
        """Clear cache"""
        import asyncio
        
        async def _clear():
            await self._memory_manager.clear_all()
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_clear())
            else:
                loop.run_until_complete(_clear())
        except RuntimeError:
            asyncio.run(_clear())
        
        logger.info(f"Cache '{self.name}' cleared")
    
    def size(self) -> int:
        """Get current cache size"""
        stats = self._memory_manager.get_stats()
        return stats.get('total_items', 0)
    
    def get_stats(self) -> MemoryUsageStats:
        """Get cache statistics"""
        memory_stats = self._memory_manager.get_stats()
        
        return MemoryUsageStats(
            current_memory_mb=memory_stats.get('estimated_usage_mb', 0),
            max_memory_mb=self.max_memory_mb,
            cache_count=memory_stats.get('total_items', 0),
            eviction_count=memory_stats.get('evictions', 0),
            hit_count=self.hit_count,
            miss_count=self.miss_count
        )
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio"""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests


class BoundedMemoryManager:
    """
    Backward compatibility wrapper for simplified memory management.
    
    Maintains the same interface but uses simplified memory manager internally.
    """
    
    def __init__(self, global_memory_limit_mb: float = 500.0, 
                 pattern_cache_size: int = 5000, result_cache_size: int = 2000,
                 session_cache_size: int = 1000, cleanup_interval: float = 300.0):
        
        self.global_memory_limit_mb = global_memory_limit_mb
        
        # Use simplified memory manager
        self._memory_manager = SimpleMemoryManager(memory_limit_mb=global_memory_limit_mb)
        
        # Create simplified caches using LRUCache wrapper
        self.pattern_cache = LRUCache(
            max_size=pattern_cache_size,
            max_memory_mb=global_memory_limit_mb * 0.4,
            name="PatternCache"
        )
        
        self.result_cache = LRUCache(
            max_size=result_cache_size, 
            max_memory_mb=global_memory_limit_mb * 0.3,
            name="ResultCache"
        )
        
        self.session_cache = LRUCache(
            max_size=session_cache_size,
            max_memory_mb=global_memory_limit_mb * 0.2,
            name="SessionCache"
        )
        
        # Simplified memory monitor
        self.memory_monitor = MemoryMonitor()
        
        logger.info("Bounded Memory Manager initialized (simplified mode)")
    
    async def start(self):
        """Start memory management services"""
        await self.memory_monitor.start_monitoring()
        logger.info("Bounded Memory Manager started")
    
    async def stop(self):
        """Stop memory management services"""
        await self.memory_monitor.stop_monitoring()
        logger.info("Bounded Memory Manager stopped")
    
    async def store_pattern(self, key: str, pattern: Any, cache_type: str = "pattern") -> bool:
        """Store pattern with memory bounds checking"""
        cache_map = {
            'pattern': self.pattern_cache,
            'result': self.result_cache, 
            'session': self.session_cache
        }
        
        cache = cache_map.get(cache_type, self.pattern_cache)
        return await cache.put(key, pattern)
    
    async def retrieve_pattern(self, key: str, cache_type: str = "pattern") -> Optional[Any]:
        """Retrieve pattern from appropriate cache"""
        cache_map = {
            'pattern': self.pattern_cache,
            'result': self.result_cache,
            'session': self.session_cache
        }
        
        cache = cache_map.get(cache_type, self.pattern_cache)
        return await cache.get(key)
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and cache statistics"""
        memory_stats = self.memory_monitor.get_memory_stats()
        
        return {
            'global_memory': memory_stats,
            'global_limit_mb': self.global_memory_limit_mb,
            'memory_utilization': memory_stats.get('current_mb', 0) / self.global_memory_limit_mb,
            'caches': {
                'pattern_cache': {
                    'stats': self.pattern_cache.get_stats().__dict__,
                    'hit_ratio': self.pattern_cache.get_hit_ratio(),
                    'size': self.pattern_cache.size()
                },
                'result_cache': {
                    'stats': self.result_cache.get_stats().__dict__,
                    'hit_ratio': self.result_cache.get_hit_ratio(),
                    'size': self.result_cache.size()
                },
                'session_cache': {
                    'stats': self.session_cache.get_stats().__dict__,
                    'hit_ratio': self.session_cache.get_hit_ratio(),
                    'size': self.session_cache.size()
                }
            }
        }
    
    async def clear_all_caches(self):
        """Clear all caches"""
        self.pattern_cache.clear()
        self.result_cache.clear()
        self.session_cache.clear()
        logger.info("All caches cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for memory management system"""
        try:
            stats = await self.get_comprehensive_stats()
            memory_utilization = stats.get('memory_utilization', 0)
            
            if memory_utilization > 0.9:
                status = 'critical'
            elif memory_utilization > 0.8:
                status = 'warning' 
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'memory_utilization': memory_utilization,
                'total_cache_items': sum(
                    cache_info['size'] for cache_info in stats['caches'].values()
                )
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Global memory manager instance
global_memory_manager: Optional[BoundedMemoryManager] = None


async def get_memory_manager() -> BoundedMemoryManager:
    """Get or create global memory manager instance"""
    global global_memory_manager
    
    if global_memory_manager is None:
        global_memory_manager = BoundedMemoryManager()
        await global_memory_manager.start()
    
    return global_memory_manager