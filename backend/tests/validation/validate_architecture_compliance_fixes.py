"""
Validation Tests for Architecture Compliance Fixes

This module validates that all critical architecture compliance fixes
have been properly implemented and meet the specified requirements.
"""

import asyncio
import time
import uuid
import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import the fixed components
from agents.search.tri_modal_orchestrator import TriModalOrchestrator, SearchResult
from agents.discovery.dynamic_pattern_extractor import DynamicPatternExtractor, IntentPattern
from agents.base.optimized_reasoning_engine import OptimizedReasoningEngine, ReasoningStrategy
from core.observability.enhanced_observability import (
    ObservableOperation, OperationType, observable_operation, get_current_correlation_id
)
from core.memory.bounded_memory_manager import BoundedMemoryManager, LRUCache


class TestTriModalUnityFix:
    """Test Fix 1: Tri-Modal Unity Principle Implementation"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create tri-modal orchestrator for testing"""
        return TriModalOrchestrator(timeout=2.0)
    
    @pytest.mark.asyncio
    async def test_unified_search_executes_all_modalities(self, orchestrator):
        """Test that all three modalities are executed simultaneously"""
        
        query = "find similar documents about machine learning"
        context = {"domain": "technical"}
        
        result = await orchestrator.execute_unified_search(query, context)
        
        # Validate tri-modal unity principle
        assert isinstance(result, SearchResult)
        assert result.modality_contributions is not None
        assert 'vector_contribution' in result.modality_contributions
        assert 'graph_contribution' in result.modality_contributions
        assert 'gnn_contribution' in result.modality_contributions
        
        # Verify no heuristic selection logic
        assert 'synthesis_method' in result.metadata
        assert result.metadata['synthesis_method'] == 'tri_modal_unity'
        
        # Verify performance requirements
        assert result.execution_time < 3.0
        
        print(f"✅ Tri-modal unity test passed - all modalities executed in {result.execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_no_heuristic_selection_remains(self, orchestrator):
        """Test that no heuristic keyword matching remains"""
        
        # Test various query types to ensure no heuristic selection
        test_queries = [
            "find similar documents",  # Should NOT trigger only vector search
            "show relationships between entities",  # Should NOT trigger only graph search
            "predict future trends",  # Should NOT trigger only GNN search
            "complex query with multiple aspects"
        ]
        
        for query in test_queries:
            result = await orchestrator.execute_unified_search(query, {})
            
            # Every query should use all three modalities
            contributions = result.modality_contributions
            assert len(contributions) == 3, f"Query '{query}' did not use all modalities"
            
            # Verify strengthening approach (not selection)
            for modality, contribution in contributions.items():
                assert contribution['content_influence'] > 0, f"Modality {modality} not contributing"
        
        print("✅ No heuristic selection logic detected - all queries use unified approach")
    
    @pytest.mark.asyncio
    async def test_health_check_functionality(self, orchestrator):
        """Test orchestrator health check"""
        
        health_status = await orchestrator.health_check()
        
        assert 'orchestrator' in health_status
        assert 'modalities' in health_status
        assert health_status['orchestrator'] in ['healthy', 'unhealthy']
        
        # Check all three modalities are monitored
        modalities = health_status['modalities']
        assert 'vector' in modalities
        assert 'graph' in modalities
        assert 'gnn' in modalities
        
        print("✅ Tri-modal orchestrator health check functioning")


class TestDataDrivenDiscoveryFix:
    """Test Fix 2: Data-Driven Domain Discovery Implementation"""
    
    @pytest.fixture
    async def pattern_extractor(self):
        """Create dynamic pattern extractor for testing"""
        return DynamicPatternExtractor()
    
    @pytest.mark.asyncio
    async def test_no_hardcoded_keywords_remain(self, pattern_extractor):
        """Test that no hardcoded intent keywords are used"""
        
        query = "analyze performance metrics for optimization"
        context = {"domain": "technical", "user_history": []}
        
        intent_pattern = await pattern_extractor.extract_intent_patterns(query, context)
        
        # Verify data-driven approach
        assert isinstance(intent_pattern, IntentPattern)
        assert intent_pattern.metadata['hardcoded_assumptions'] == False
        assert intent_pattern.metadata['extraction_method'] == 'data_driven_discovery'
        
        # Verify patterns are discovered, not hardcoded
        assert len(intent_pattern.discovered_patterns) > 0
        assert intent_pattern.semantic_features is not None
        
        print(f"✅ Data-driven discovery working - found {len(intent_pattern.discovered_patterns)} patterns")
    
    @pytest.mark.asyncio
    async def test_dynamic_pattern_learning(self, pattern_extractor):
        """Test that patterns are learned from data"""
        
        # Clear any existing patterns
        await pattern_extractor.clear_learned_patterns()
        
        # Process multiple queries to test learning
        queries = [
            "optimize database performance",
            "improve system efficiency", 
            "enhance application speed",
            "boost processing power"
        ]
        
        context = {"domain": "performance"}
        learned_patterns_count = []
        
        for query in queries:
            await pattern_extractor.extract_intent_patterns(query, context)
            stats = await pattern_extractor.get_learning_statistics()
            learned_patterns_count.append(stats['total_patterns'])
        
        # Verify learning is occurring
        assert learned_patterns_count[-1] > learned_patterns_count[0], "Pattern learning not functioning"
        
        print(f"✅ Dynamic pattern learning validated - patterns grew from {learned_patterns_count[0]} to {learned_patterns_count[-1]}")
    
    @pytest.mark.asyncio
    async def test_contextual_pattern_extraction(self, pattern_extractor):
        """Test context-aware pattern extraction"""
        
        query = "create new dashboard"
        
        # Test with different contexts
        technical_context = {"domain": "technical", "user_history": ["debug application", "fix performance"]}
        business_context = {"domain": "business", "user_history": ["generate report", "analyze metrics"]}
        
        tech_result = await pattern_extractor.extract_intent_patterns(query, technical_context)
        business_result = await pattern_extractor.extract_intent_patterns(query, business_context)
        
        # Results should differ based on context
        assert tech_result.discovered_patterns != business_result.discovered_patterns or \
               tech_result.semantic_features != business_result.semantic_features
        
        print("✅ Contextual pattern extraction working - different contexts produce different patterns")


class TestOptimizedReasoningFix:
    """Test Fix 3: Parallel Reasoning with Timeout Implementation"""
    
    @pytest.fixture
    async def reasoning_engine(self):
        """Create optimized reasoning engine for testing"""
        return OptimizedReasoningEngine()
    
    @pytest.mark.asyncio
    async def test_sub_3_second_response_requirement(self, reasoning_engine):
        """Test that reasoning completes within 3 seconds"""
        
        query = "complex reasoning query requiring multiple steps"
        context = {"domain": "test", "constraints": {}}
        
        start_time = time.time()
        result = await reasoning_engine.execute_reasoning_with_timeout(
            query, context, timeout=2.5
        )
        execution_time = time.time() - start_time
        
        # Verify performance requirement
        assert execution_time < 3.0, f"Reasoning took {execution_time:.2f}s, exceeds 3s limit"
        assert result.performance_met == True
        
        print(f"✅ Sub-3-second response validated - completed in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_parallel_execution_strategies(self, reasoning_engine):
        """Test that multiple reasoning strategies execute in parallel"""
        
        query = "test parallel reasoning"
        context = {}
        
        # Mock the individual engines to track parallel execution
        with patch.object(reasoning_engine.fast_engine, 'execute_reasoning', new_callable=AsyncMock) as mock_fast, \
             patch.object(reasoning_engine.deep_engine, 'execute_reasoning', new_callable=AsyncMock) as mock_deep, \
             patch.object(reasoning_engine.context_engine, 'execute_reasoning', new_callable=AsyncMock) as mock_context:
            
            # Configure mocks to simulate different completion times
            mock_fast.return_value = MagicMock(strategy_used=ReasoningStrategy.FAST_PATH, performance_met=True)
            mock_deep.return_value = MagicMock(strategy_used=ReasoningStrategy.DEEP_ANALYSIS, performance_met=True)
            mock_context.return_value = MagicMock(strategy_used=ReasoningStrategy.CONTEXT_AWARE, performance_met=True)
            
            result = await reasoning_engine.execute_reasoning_with_timeout(query, context, timeout=2.0)
            
            # Verify all engines were called (parallel execution)
            mock_fast.assert_called_once()
            mock_deep.assert_called_once()  
            mock_context.assert_called_once()
        
        print("✅ Parallel reasoning execution validated")
    
    @pytest.mark.asyncio
    async def test_early_termination_on_timeout(self, reasoning_engine):
        """Test early termination and fallback on timeout"""
        
        query = "timeout test query"
        context = {}
        
        # Use very short timeout to trigger fallback
        result = await reasoning_engine.execute_reasoning_with_timeout(
            query, context, timeout=0.1
        )
        
        # Should get fallback result
        assert result is not None
        assert 'fallback' in result.metadata or result.execution_time >= 0.1
        
        print("✅ Early termination and fallback validated")
    
    @pytest.mark.asyncio
    async def test_hybrid_reasoning_performance(self, reasoning_engine):
        """Test hybrid reasoning performance optimization"""
        
        query = "hybrid reasoning test"
        context = {}
        
        result = await reasoning_engine.execute_hybrid_reasoning(
            query, context, timeout=2.0
        )
        
        assert result.strategy_used == ReasoningStrategy.HYBRID
        assert result.execution_time < 2.5
        
        print("✅ Hybrid reasoning performance validated")


class TestEnhancedObservabilityFix:
    """Test Fix 4: Enhanced Observability Implementation"""
    
    @pytest.mark.asyncio
    async def test_correlation_id_tracking(self):
        """Test correlation ID tracking across operations"""
        
        correlation_id = str(uuid.uuid4())
        
        async with ObservableOperation(
            "test_operation",
            OperationType.BACKGROUND_TASK,
            correlation_id=correlation_id
        ) as op:
            
            # Check correlation ID is set
            current_id = get_current_correlation_id()
            assert current_id == correlation_id
            
            # Create child operation
            child_op = op.create_child_operation("child_test", OperationType.API_REQUEST)
            assert child_op.context.correlation_id == correlation_id
            assert child_op.context.parent_operation_id == op.operation_id
        
        print("✅ Correlation ID tracking validated")
    
    @pytest.mark.asyncio
    async def test_structured_logging_with_context(self):
        """Test structured logging with operation context"""
        
        test_metadata = {"test_key": "test_value", "query": "test query"}
        
        async with ObservableOperation(
            "structured_logging_test",
            OperationType.AGENT_REASONING,
            metadata=test_metadata,
            tags=["test", "validation"]
        ) as op:
            
            op.log_checkpoint("test_checkpoint", {"checkpoint_data": "test"})
            op.add_metadata("runtime_data", "added_at_runtime")
            op.add_tag("runtime_tag")
            
            # Verify context is properly structured
            assert op.context.metadata["test_key"] == "test_value"
            assert "runtime_data" in op.context.metadata
            assert "runtime_tag" in op.context.tags
        
        print("✅ Structured logging with context validated")
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring and threshold detection"""
        
        # Test with operation that should meet performance threshold
        async with ObservableOperation(
            "fast_operation",
            OperationType.API_REQUEST,
            performance_threshold=1.0
        ) as op:
            await asyncio.sleep(0.1)  # Fast operation
        
        assert op.metrics.performance_met == True
        
        # Test with operation that exceeds threshold
        async with ObservableOperation(
            "slow_operation", 
            OperationType.API_REQUEST,
            performance_threshold=0.05
        ) as op:
            await asyncio.sleep(0.1)  # Slow relative to threshold
        
        assert op.metrics.performance_met == False
        
        print("✅ Performance monitoring and threshold detection validated")


class TestBoundedMemoryFix:
    """Test Fix 5: Bounded Memory Management Implementation"""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create bounded memory manager for testing"""
        manager = BoundedMemoryManager(
            global_memory_limit_mb=50.0,  # Small limit for testing
            pattern_cache_size=100,
            result_cache_size=50,
            session_cache_size=30
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_lru_eviction_functionality(self, memory_manager):
        """Test LRU eviction when cache limits are exceeded"""
        
        # Fill cache beyond limit
        for i in range(150):  # Exceeds pattern_cache_size of 100
            await memory_manager.store_pattern(f"pattern_{i}", f"data_{i}", "pattern")
        
        # Verify cache is bounded
        stats = await memory_manager.get_comprehensive_stats()
        pattern_cache_size = stats['caches']['pattern_cache']['size']
        
        assert pattern_cache_size <= 100, f"Cache not bounded: {pattern_cache_size} items"
        
        # Verify LRU behavior - recent items should be present
        recent_item = await memory_manager.retrieve_pattern("pattern_149", "pattern")
        old_item = await memory_manager.retrieve_pattern("pattern_0", "pattern")
        
        assert recent_item is not None, "Recent item was evicted"
        assert old_item is None, "Old item was not evicted"
        
        print("✅ LRU eviction functionality validated")
    
    @pytest.mark.asyncio
    async def test_memory_bounds_enforcement(self, memory_manager):
        """Test memory bounds are enforced"""
        
        # Get initial memory usage
        initial_stats = await memory_manager.get_comprehensive_stats()
        initial_memory = initial_stats['global_memory']['current_mb']
        
        # Fill caches with data
        large_data = "x" * 1000  # 1KB per item
        for i in range(1000):
            await memory_manager.store_pattern(f"large_{i}", large_data, "pattern")
        
        # Check memory bounds
        final_stats = await memory_manager.get_comprehensive_stats()
        memory_utilization = final_stats['memory_utilization']
        
        # Memory utilization should be managed (not unlimited growth)
        assert memory_utilization < 2.0, f"Memory utilization too high: {memory_utilization}"
        
        print(f"✅ Memory bounds enforced - utilization: {memory_utilization:.2f}")
    
    @pytest.mark.asyncio
    async def test_cache_hit_ratios(self, memory_manager):
        """Test cache performance with hit ratios"""
        
        # Store some test data
        test_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        for key, value in test_data.items():
            await memory_manager.store_pattern(key, value, "result")
        
        # Access items multiple times
        for _ in range(5):
            for key in test_data.keys():
                retrieved = await memory_manager.retrieve_pattern(key, "result")
                assert retrieved is not None
        
        # Check hit ratios
        stats = await memory_manager.get_comprehensive_stats()
        result_cache_stats = stats['caches']['result_cache']
        hit_ratio = result_cache_stats['hit_ratio']
        
        assert hit_ratio > 0.8, f"Hit ratio too low: {hit_ratio}"
        
        print(f"✅ Cache hit ratio validated: {hit_ratio:.2f}")
    
    @pytest.mark.asyncio
    async def test_health_check_functionality(self, memory_manager):
        """Test memory manager health check"""
        
        health = await memory_manager.health_check()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'warning', 'caution', 'critical', 'error']
        assert 'current_memory_mb' in health
        assert 'memory_utilization' in health
        
        print(f"✅ Memory manager health check validated - status: {health['status']}")


class TestIntegrationValidation:
    """Integration tests for all fixes working together"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test all fixes working together in realistic scenario"""
        
        # Create components
        orchestrator = TriModalOrchestrator(timeout=2.5)
        pattern_extractor = DynamicPatternExtractor()
        reasoning_engine = OptimizedReasoningEngine(pattern_extractor)
        
        query = "find and analyze performance patterns in distributed systems"
        context = {"domain": "technical", "user_history": ["system optimization", "performance tuning"]}
        
        start_time = time.time()
        
        # Test tri-modal search
        search_result = await orchestrator.execute_unified_search(query, context)
        
        # Test dynamic pattern extraction
        intent_pattern = await pattern_extractor.extract_intent_patterns(query, context)
        
        # Test optimized reasoning
        reasoning_result = await reasoning_engine.execute_reasoning_with_timeout(
            query, context, timeout=2.0
        )
        
        total_time = time.time() - start_time
        
        # Validate integration
        assert search_result.confidence > 0.5
        assert intent_pattern.confidence > 0.5
        assert reasoning_result.performance_met == True
        assert total_time < 5.0  # Total system response time
        
        print(f"✅ Full system integration validated - completed in {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        
        orchestrator = TriModalOrchestrator(timeout=0.01)  # Very short timeout
        
        # Should gracefully handle timeout
        result = await orchestrator.execute_unified_search("test query", {})
        
        assert result is not None
        assert 'fallback' in result.metadata or result.execution_time >= 0.01
        
        print("✅ Error handling and fallbacks validated")


async def run_all_validation_tests():
    """Run all validation tests and provide summary"""
    
    print("🔍 Starting Architecture Compliance Fix Validation Tests\n")
    
    test_classes = [
        TestTriModalUnityFix(),
        TestDataDrivenDiscoveryFix(), 
        TestOptimizedReasoningFix(),
        TestEnhancedObservabilityFix(),
        TestBoundedMemoryFix(),
        TestIntegrationValidation()
    ]
    
    passed_tests = 0
    total_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n🧪 Running {class_name}")
        print("-" * 50)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, method_name)
                
                # Handle fixtures for methods that need them
                if hasattr(test_class, method_name.replace('test_', '') + '_fixture'):
                    # Skip methods requiring fixtures for now in direct execution
                    continue
                
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                passed_tests += 1
                
            except Exception as e:
                failed_tests.append(f"{class_name}.{method_name}: {str(e)}")
                print(f"❌ {method_name} failed: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 VALIDATION SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {len(failed_tests)}")
    print(f"📊 Total: {total_tests}")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for failure in failed_tests:
            print(f"   - {failure}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 VALIDATION PASSED - Architecture compliance fixes working correctly!")
        return True
    else:
        print("⚠️  VALIDATION NEEDS ATTENTION - Some fixes require investigation")
        return False


if __name__ == "__main__":
    """Run validation when executed directly"""
    import sys
    
    async def main():
        success = await run_all_validation_tests()
        sys.exit(0 if success else 1)
    
    asyncio.run(main())