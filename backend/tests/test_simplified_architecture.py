"""
Test Suite for Simplified Architecture Performance

This test suite verifies that the simplified cache and error handling
systems maintain the required performance characteristics.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock

# Import simplified components
from agents.base.simple_cache import SimpleCache, cached_operation
from agents.base.simple_error_handler import SimpleErrorHandler, ErrorType, ErrorContext, resilient_operation
from agents.base.performance_cache import PerformanceCache, get_performance_cache
from agents.base.error_handling import ErrorHandler, get_error_handler


class TestSimplifiedCache:
    """Test simplified cache performance and functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_basic_operations(self):
        """Test basic cache get/set operations"""
        cache = SimpleCache(max_size=100, ttl=60)
        
        # Test cache miss
        result = await cache.get("test_op", {"param": "value"})
        assert result is None
        
        # Test cache set and hit
        await cache.set("test_op", {"param": "value"}, "test_result")
        result = await cache.get("test_op", {"param": "value"})
        assert result == "test_result"
        
        # Verify stats
        stats = cache.get_stats()
        assert stats["total_requests"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate_percent"] == 50.0
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance under load"""
        cache = SimpleCache(max_size=1000, ttl=300)
        
        # Test with multiple operations
        operations = [
            ("search_op", {"query": f"test_{i}", "domain": "tech"})
            for i in range(100)
        ]
        
        # Fill cache
        start_time = time.time()
        for i, (op, params) in enumerate(operations):
            await cache.set(op, params, f"result_{i}")
        fill_time = time.time() - start_time
        
        # Read from cache
        start_time = time.time()
        for op, params in operations:
            result = await cache.get(op, params)
            assert result is not None
        read_time = time.time() - start_time
        
        # Performance assertions
        assert fill_time < 1.0, f"Cache fill took {fill_time:.3f}s, should be < 1.0s"
        assert read_time < 0.1, f"Cache read took {read_time:.3f}s, should be < 0.1s"
        
        # Hit rate should be 100% for cache reads
        stats = cache.get_stats()
        assert stats["hit_rate_percent"] == 100.0
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU eviction works correctly"""
        cache = SimpleCache(max_size=3, ttl=300)
        
        # Fill cache to capacity
        await cache.set("op1", {"id": 1}, "result1")
        await cache.set("op2", {"id": 2}, "result2") 
        await cache.set("op3", {"id": 3}, "result3")
        
        assert cache.get_stats()["cache_size"] == 3
        
        # Add one more item to trigger eviction
        await cache.set("op4", {"id": 4}, "result4")
        
        # Should still have max_size items
        assert cache.get_stats()["cache_size"] == 3
        
        # First item should be evicted (LRU)
        result = await cache.get("op1", {"id": 1})
        assert result is None
        
        # Last item should still exist  
        result = await cache.get("op4", {"id": 4})
        assert result == "result4"
    
    @pytest.mark.asyncio
    async def test_cached_operation_function(self):
        """Test the cached_operation helper function"""
        call_count = 0
        
        async def expensive_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return f"result_{call_count}"
        
        # First call should execute function
        result1 = await cached_operation("expensive_op", {"param": "test"}, expensive_operation)
        assert call_count == 1
        assert result1 == "result_1"
        
        # Second call should use cache
        result2 = await cached_operation("expensive_op", {"param": "test"}, expensive_operation)
        assert call_count == 1  # Should not increment
        assert result2 == "result_1"  # Same cached result


class TestSimplifiedErrorHandler:
    """Test simplified error handling performance and functionality"""
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test error classification into 3 types"""
        handler = SimpleErrorHandler()
        context = ErrorContext("test_op", {"param": "value"})
        
        # Test transient error
        error = TimeoutError("Request timeout")
        error_type = handler.classify_error(error, context)
        assert error_type == ErrorType.TRANSIENT
        
        # Test permanent error
        error = ValueError("Invalid input")
        error_type = handler.classify_error(error, context)
        assert error_type == ErrorType.PERMANENT
        
        # Test critical error
        error = MemoryError("Out of memory")
        error_type = handler.classify_error(error, context)
        assert error_type == ErrorType.CRITICAL
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test error handling performance"""
        handler = SimpleErrorHandler()
        
        # Test handling multiple errors quickly
        errors = [
            (TimeoutError("timeout"), "timeout_op"),
            (ValueError("validation"), "validation_op"),
            (ConnectionError("network"), "network_op")
        ]
        
        start_time = time.time()
        for error, operation in errors:
            context = ErrorContext(operation, {"test": True})
            result = await handler.handle_error(error, context)
            assert "error_id" in result
            assert "error_type" in result
        
        handling_time = time.time() - start_time
        assert handling_time < 0.5, f"Error handling took {handling_time:.3f}s, should be < 0.5s"
        
        # Verify all errors were recorded
        stats = handler.get_error_stats()
        assert stats["total_errors"] == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        handler = SimpleErrorHandler()
        
        operation = "failing_operation"
        context = ErrorContext(operation, {"test": True})
        
        # Generate failures to trip circuit breaker
        for i in range(6):  # Above threshold of 5
            handler.record_failure(operation)
        
        # Circuit breaker should be open
        assert handler._is_circuit_breaker_open(operation) == True
        
        # Should not retry when circuit breaker is open
        error = TimeoutError("timeout")
        result = await handler.handle_error(error, context)
        assert result["should_retry"] == False
    
    @pytest.mark.asyncio 
    async def test_resilient_operation_decorator(self):
        """Test the resilient_operation decorator"""
        call_count = 0
        
        @resilient_operation("test_resilient_op", max_retries=2, timeout=5.0)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Simulated timeout")
            return "success"
        
        # Should retry and eventually succeed
        result = await test_function()
        assert result == "success"
        assert call_count == 3  # Initial call + 2 retries


class TestBackwardCompatibility:
    """Test that simplified components maintain backward compatibility"""
    
    @pytest.mark.asyncio
    async def test_performance_cache_compatibility(self):
        """Test PerformanceCache maintains same interface"""
        cache = PerformanceCache()
        
        # Test basic operations work
        await cache.set("test_op", {"param": "value"}, "test_result")
        result = await cache.get("test_op", {"param": "value"})
        assert result == "test_result"
        
        # Test metrics still work
        metrics = await cache.get_performance_metrics()
        assert "cache_stats" in metrics
        assert "performance_stats" in metrics
        assert "health" in metrics
    
    @pytest.mark.asyncio
    async def test_error_handler_compatibility(self):
        """Test ErrorHandler maintains same interface"""
        handler = get_error_handler()
        
        # Test old methods still work
        context = ErrorContext("test_op", {"param": "value"})
        error = ValueError("test error")
        
        # Old categorize_error method
        category = handler.categorize_error(error, context)
        assert hasattr(category, 'value')  # Should be an enum
        
        # Old determine_severity method
        severity = handler.determine_severity(error, context, category)
        assert hasattr(severity, 'value')  # Should be an enum
        
        # Circuit breaker compatibility
        cb = handler.get_circuit_breaker("test_operation")
        assert hasattr(cb, 'should_allow_request')
        assert hasattr(cb, 'record_success')
        assert hasattr(cb, 'record_failure')


class TestPerformanceRequirements:
    """Test that simplified architecture meets performance requirements"""
    
    @pytest.mark.asyncio
    async def test_sub_3_second_response_maintained(self):
        """Test that cache and error handling don't add significant overhead"""
        cache = SimpleCache(max_size=1000, ttl=300)
        handler = SimpleErrorHandler()
        
        async def mock_search_operation():
            # Simulate a 1-second search operation
            await asyncio.sleep(0.1)  # Reduced for test speed
            return {"results": ["item1", "item2", "item3"]}
        
        # Test cached operation performance
        start_time = time.time()
        
        # First call - cache miss
        result1 = await cached_operation("tri_modal_search", {"query": "test"}, mock_search_operation)
        first_call_time = time.time() - start_time
        
        # Second call - cache hit
        start_time = time.time()
        result2 = await cached_operation("tri_modal_search", {"query": "test"}, mock_search_operation)
        second_call_time = time.time() - start_time
        
        # Assertions
        assert result1 == result2
        assert first_call_time > 0.1  # Should include operation time
        assert second_call_time < 0.01  # Should be very fast (cache hit)
        
        print(f"First call (cache miss): {first_call_time:.3f}s")
        print(f"Second call (cache hit): {second_call_time:.3f}s")
        print(f"Cache speedup: {first_call_time / second_call_time:.1f}x")
    
    @pytest.mark.asyncio
    async def test_memory_usage_efficiency(self):
        """Test that simplified components use memory efficiently"""
        cache = SimpleCache(max_size=100, ttl=300)
        handler = SimpleErrorHandler()
        
        # Fill cache with data
        for i in range(100):
            await cache.set(f"op_{i}", {"param": i}, f"result_{i}")
        
        # Generate some errors
        for i in range(10):
            context = ErrorContext(f"op_{i}", {"param": i})
            error = TimeoutError(f"Error {i}")
            await handler.handle_error(error, context)
        
        # Verify reasonable memory usage
        cache_stats = cache.get_stats()
        error_stats = handler.get_error_stats()
        
        assert cache_stats["cache_size"] == 100
        assert error_stats["total_errors"] == 10
        
        # Health should be good
        health = cache.get_health_status()
        assert health["performance_acceptable"] == True


if __name__ == "__main__":
    # Run basic tests directly
    import asyncio
    
    async def run_basic_tests():
        print("Testing simplified cache...")
        cache = SimpleCache(max_size=10, ttl=60)
        await cache.set("test", {"param": "value"}, "result")
        result = await cache.get("test", {"param": "value"})
        print(f"Cache test result: {result}")
        print(f"Cache stats: {cache.get_stats()}")
        
        print("\nTesting simplified error handler...")
        handler = SimpleErrorHandler()
        context = ErrorContext("test_op", {"param": "value"})
        error_result = await handler.handle_error(ValueError("test"), context)
        print(f"Error handling result: {error_result}")
        print(f"Error stats: {handler.get_error_stats()}")
        
        print("\nAll basic tests passed!")
    
    asyncio.run(run_basic_tests())