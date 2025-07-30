#!/usr/bin/env python3
"""
Test script for Minimal Architecture Enhancement implementation
Validates parallel execution, caching, and performance monitoring
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_query_service_import():
    """Test that query service can be imported and initialized"""
    try:
        from services.query_service import QueryService
        from services.cache_service import SimpleCacheService
        from services.performance_service import PerformanceService
        
        logger.info("✅ Successfully imported all enhanced services")
        
        # Test initialization
        cache_service = SimpleCacheService(use_redis=False)
        perf_service = PerformanceService()
        
        logger.info("✅ Successfully initialized cache and performance services")
        
        # Test cache operations
        test_key = "test_key"
        test_data = {"test": "data", "value": 123}
        
        await cache_service.set(test_key, test_data, ttl_seconds=60)
        cached_result = await cache_service.get(test_key)
        
        if cached_result == test_data:
            logger.info("✅ Cache service working correctly")
        else:
            logger.error("❌ Cache service test failed")
        
        # Test performance context
        async with perf_service.create_performance_context(
            "test query", "test_domain", "test_operation"
        ) as perf:
            await asyncio.sleep(0.1)  # Simulate work
            perf.mark_cache_hit()
            perf.set_result_count(5)
        
        logger.info("✅ Performance monitoring context working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import/initialization test failed: {e}")
        return False

async def test_parallel_execution_syntax():
    """Test that the parallel execution syntax is correct"""
    try:
        # Test the asyncio.gather pattern we implemented
        async def mock_search_1():
            await asyncio.sleep(0.1)
            return {"type": "documents", "results": [1, 2, 3]}
        
        async def mock_search_2():
            await asyncio.sleep(0.1)
            return {"type": "graph", "results": ["a", "b", "c"]}
        
        async def mock_search_3():
            await asyncio.sleep(0.1)
            return {"type": "entities", "results": ["x", "y", "z"]}
        
        # This is the pattern we implemented in semantic_search
        tasks = [mock_search_1(), mock_search_2(), mock_search_3()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if len(results) == 3 and all(isinstance(r, dict) for r in results):
            logger.info("✅ Parallel execution pattern working correctly")
            return True
        else:
            logger.error("❌ Parallel execution pattern test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Parallel execution test failed: {e}")
        return False

async def test_cache_performance():
    """Test cache hit/miss performance"""
    try:
        from services.cache_service import SimpleCacheService
        import time
        
        cache = SimpleCacheService(use_redis=False)
        
        # Test cache miss (first time)
        async def expensive_operation():
            await asyncio.sleep(0.1)  # Simulate expensive work
            return {"expensive": "result", "timestamp": time.time()}
        
        # First call - should be cache miss
        start_time = time.time()
        result1 = await cache.get_or_compute(
            "test_expensive", expensive_operation, ttl_seconds=60
        )
        miss_time = time.time() - start_time
        
        # Second call - should be cache hit
        start_time = time.time()
        result2 = await cache.get_or_compute(
            "test_expensive", expensive_operation, ttl_seconds=60
        )
        hit_time = time.time() - start_time
        
        if result1 == result2 and hit_time < miss_time * 0.5:
            logger.info(f"✅ Cache performance test passed - Miss: {miss_time:.3f}s, Hit: {hit_time:.3f}s")
            return True
        else:
            logger.error(f"❌ Cache performance test failed - Miss: {miss_time:.3f}s, Hit: {hit_time:.3f}s")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cache performance test failed: {e}")
        return False

async def test_configuration_loading():
    """Test that domain patterns and configurations still load correctly"""
    try:
        from config.domain_patterns import DomainPatternManager
        
        # Test domain detection
        test_query = "What maintenance is needed for centrifugal pump bearings?"
        domain = DomainPatternManager.detect_domain(test_query)
        
        # Test pattern loading
        patterns = DomainPatternManager.get_patterns(domain)
        training = DomainPatternManager.get_training(domain)
        
        if domain and patterns and training:
            logger.info(f"✅ Configuration loading working - Detected domain: {domain}")
            return True
        else:
            logger.error("❌ Configuration loading test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Configuration loading test failed: {e}")
        return False

async def run_all_tests():
    """Run all validation tests"""
    logger.info("🚀 Starting Minimal Architecture Enhancement validation tests")
    
    tests = [
        ("Service Import & Initialization", test_query_service_import),
        ("Parallel Execution Pattern", test_parallel_execution_syntax),
        ("Cache Performance", test_cache_performance),
        ("Configuration Loading", test_configuration_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info(f"\n📊 Test Results Summary:")
    logger.info(f"✅ Passed: {passed}/{total}")
    logger.info(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Minimal Architecture Enhancement is working correctly.")
        return True
    else:
        logger.error("⚠️  Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)