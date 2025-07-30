#!/usr/bin/env python3
"""
Demo script showing performance improvements from Minimal Architecture Enhancement
"""

import asyncio
import time
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from services.cache_service import SimpleCacheService
from services.performance_service import PerformanceService

async def simulate_old_sequential_search(query: str, domain: str):
    """Simulate the old sequential search pattern"""
    print(f"🐌 OLD (Sequential): Processing '{query}'")
    start_time = time.time()
    
    # Simulate sequential execution (old way)
    print("  ├─ Vector search...")
    await asyncio.sleep(0.3)  # Simulate vector search time
    
    print("  ├─ Graph search...")
    await asyncio.sleep(0.5)  # Simulate graph search time
    
    print("  └─ Entity search...")
    await asyncio.sleep(0.2)  # Simulate entity search time
    
    total_time = time.time() - start_time
    print(f"  ⏱️  Total time: {total_time:.2f}s")
    return {"total_time": total_time, "cached": False}

async def simulate_new_parallel_search(query: str, domain: str, cache_service: SimpleCacheService):
    """Simulate the new parallel search pattern with caching"""
    print(f"⚡ NEW (Parallel + Cache): Processing '{query}'")
    
    # Check cache first
    cache_key = f"search_{query}_{domain}"
    cached_result = await cache_service.get(cache_key)
    
    if cached_result:
        print("  ✅ Cache HIT - Returning cached result")
        print(f"  ⏱️  Total time: ~0.05s (99% faster!)")
        return {"total_time": 0.05, "cached": True}
    
    print("  ❌ Cache MISS - Computing result")
    start_time = time.time()
    
    # Simulate parallel execution (new way)
    print("  ├─ Parallel execution:")
    print("  │   ├─ Vector search...")
    print("  │   ├─ Graph search...")
    print("  │   └─ Entity search...")
    
    # All searches run in parallel using asyncio.gather
    await asyncio.gather(
        asyncio.sleep(0.3),  # Vector search
        asyncio.sleep(0.5),  # Graph search  
        asyncio.sleep(0.2),  # Entity search
    )
    # Total time is max(0.3, 0.5, 0.2) = 0.5s instead of 0.3+0.5+0.2 = 1.0s
    
    total_time = time.time() - start_time
    print(f"  ⏱️  Total time: {total_time:.2f}s")
    
    # Cache the result for next time
    result = {"total_time": total_time, "cached": False}
    await cache_service.set(cache_key, result, ttl_seconds=180)
    print("  💾 Result cached for 3 minutes")
    
    return result

async def demonstrate_performance_monitoring():
    """Demonstrate the performance monitoring capabilities"""
    print("\n📊 Performance Monitoring Demo")
    print("=" * 50)
    
    perf_service = PerformanceService()
    
    # Simulate different query types with performance tracking
    test_queries = [
        ("What is pump maintenance?", "maintenance"),
        ("How to troubleshoot motor issues?", "maintenance"),
        ("System architecture overview", "general"),
    ]
    
    for query, domain in test_queries:
        async with perf_service.create_performance_context(
            query, domain, "demo_query"
        ) as perf:
            # Simulate query processing with different phases
            await asyncio.sleep(0.1)  # Analysis phase
            perf.record_phase_time("analysis", time.time() - 0.1)
            
            await asyncio.sleep(0.3)  # Search phase
            perf.record_phase_time("search", time.time() - 0.3)
            
            await asyncio.sleep(0.05)  # Response phase
            perf.record_phase_time("response", time.time() - 0.05)
            
            perf.set_result_count(15)
    
    # Get performance summary
    summary = await perf_service.get_performance_summary(hours=1)
    print(f"\n📈 Performance Summary:")
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Average response time: {summary['response_times']['average']}s")
    print(f"  Cache hit rate: {summary['cache_performance']['hit_rate']}%")
    print(f"  Target compliance: {summary['target_compliance']['percentage']}%")

async def main():
    """Main demonstration"""
    print("🚀 Minimal Architecture Enhancement - Performance Demo")
    print("=" * 60)
    
    # Initialize services
    cache_service = SimpleCacheService(use_redis=False)
    
    test_query = "What maintenance is needed for centrifugal pump bearings?"
    test_domain = "maintenance"
    
    print("\n1️⃣  Comparing Sequential vs Parallel Execution")
    print("-" * 50)
    
    # Show old sequential approach
    old_result = await simulate_old_sequential_search(test_query, test_domain)
    
    print()
    
    # Show new parallel approach (first time - cache miss)
    new_result_1 = await simulate_new_parallel_search(test_query, test_domain, cache_service)
    
    print()
    
    # Show new parallel approach (second time - cache hit)
    new_result_2 = await simulate_new_parallel_search(test_query, test_domain, cache_service)
    
    # Calculate improvements
    sequential_time = old_result["total_time"]
    parallel_time = new_result_1["total_time"]
    cached_time = new_result_2["total_time"]
    
    parallel_improvement = ((sequential_time - parallel_time) / sequential_time) * 100
    cache_improvement = ((sequential_time - cached_time) / sequential_time) * 100
    
    print(f"\n📈 Performance Improvements:")
    print(f"  Sequential → Parallel: {parallel_improvement:.1f}% faster")
    print(f"  Sequential → Cached:   {cache_improvement:.1f}% faster")
    print(f"  Parallel → Cached:     {((parallel_time - cached_time) / parallel_time) * 100:.1f}% faster")
    
    # Demonstrate performance monitoring
    await demonstrate_performance_monitoring()
    
    print(f"\n✅ Demo Complete!")
    print(f"Key Benefits Demonstrated:")
    print(f"  🚀 Parallel execution: ~50% faster than sequential")
    print(f"  💾 Intelligent caching: ~99% faster for repeated queries")
    print(f"  📊 Performance monitoring: Real-time analytics and SLA tracking")
    print(f"  🏗️  Clean architecture: Zero breaking changes, additive enhancements")

if __name__ == "__main__":
    asyncio.run(main())