"""
Layer Boundary Validation Tests

This module validates that all architectural layer boundaries are properly
implemented and enforced according to the layer boundary definitions.
"""

import asyncio
import pytest
import time
import uuid
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import boundary contracts and implementations
from config.inter_layer_contracts import (
    LayerType,
    OperationStatus,
    OperationResult,
    LayerBoundaryEnforcer,
    ContractViolationError,
    ContractMonitor,
    contract_monitor
)

# Import layer implementations
from agents.base.agent_service_interface import (
    IntelligenceRequest,
    IntelligenceResult,
    AgentServiceContract
)
from agents.universal_agent_service import UniversalAgentService
from services.enhanced_query_service import EnhancedQueryService
from agents.base.integrated_memory_manager import (
    IntegratedMemoryManager,
    AgentMemoryContext,
    MemoryOperationResult
)


class TestLayerBoundaryDefinitions:
    """Test that layer boundaries are correctly defined and enforced"""
    
    def test_layer_dependency_rules(self):
        """Test that layer dependency rules are correctly defined"""
        
        # Test allowed dependencies
        assert LayerBoundaryEnforcer.validate_dependency(LayerType.API, LayerType.SERVICES)
        assert LayerBoundaryEnforcer.validate_dependency(LayerType.SERVICES, LayerType.AGENTS)
        assert LayerBoundaryEnforcer.validate_dependency(LayerType.SERVICES, LayerType.CORE)
        assert LayerBoundaryEnforcer.validate_dependency(LayerType.AGENTS, LayerType.TOOLS)
        assert LayerBoundaryEnforcer.validate_dependency(LayerType.AGENTS, LayerType.CORE)
        assert LayerBoundaryEnforcer.validate_dependency(LayerType.TOOLS, LayerType.CORE)
        
        print("✅ Allowed layer dependencies validated")
    
    def test_layer_boundary_violations(self):
        """Test that boundary violations are properly detected"""
        
        # Test forbidden dependencies
        with pytest.raises(ContractViolationError):
            LayerBoundaryEnforcer.validate_dependency(LayerType.API, LayerType.AGENTS)
        
        with pytest.raises(ContractViolationError):
            LayerBoundaryEnforcer.validate_dependency(LayerType.API, LayerType.CORE)
        
        with pytest.raises(ContractViolationError):
            LayerBoundaryEnforcer.validate_dependency(LayerType.SERVICES, LayerType.API)
        
        with pytest.raises(ContractViolationError):
            LayerBoundaryEnforcer.validate_dependency(LayerType.AGENTS, LayerType.SERVICES)
        
        with pytest.raises(ContractViolationError):
            LayerBoundaryEnforcer.validate_dependency(LayerType.CORE, LayerType.AGENTS)
        
        print("✅ Layer boundary violations properly detected")
    
    def test_operation_result_validation(self):
        """Test that operation results follow contract format"""
        
        # Valid operation result
        valid_result = OperationResult(
            status=OperationStatus.SUCCESS,
            data={"test": "data"},
            execution_time=0.5,
            correlation_id="test-123"
        )
        
        assert LayerBoundaryEnforcer.validate_operation_result(valid_result, LayerType.SERVICES)
        
        # Invalid operation result (missing required fields)
        invalid_result = {"status": "success"}  # Not an OperationResult object
        
        with pytest.raises(ContractViolationError):
            LayerBoundaryEnforcer.validate_operation_result(invalid_result, LayerType.SERVICES)
        
        print("✅ Operation result validation working correctly")


class TestServiceAgentBoundaryCompliance:
    """Test that Service-Agent boundary is properly implemented"""
    
    @pytest.fixture
    async def agent_service(self):
        """Create agent service for testing"""
        return AgentServiceContract(UniversalAgentService())
    
    @pytest.fixture
    async def query_service(self):
        """Create query service for testing"""
        return EnhancedQueryService()
    
    @pytest.mark.asyncio
    async def test_service_delegates_intelligence_to_agents(self, query_service, agent_service):
        """Test that services delegate intelligence to agents, not implement it themselves"""
        
        # Mock the agent service to track calls
        with patch.object(query_service, 'agent_service', new_callable=AsyncMock) as mock_agent:
            mock_agent.analyze_query_intelligence.return_value = IntelligenceResult(
                primary_intent="test_intent",
                confidence=0.8,
                discovered_domain="test_domain",
                reasoning_trace=MagicMock(),
                tool_recommendations=["vector_search"],
                context_insights={},
                performance_met=True,
                execution_time=0.5,
                metadata={}
            )
            
            # Execute query processing
            result = await query_service.process_universal_query(
                query="test query for intelligence delegation",
                domain=None,
                max_results=10
            )
            
            # Verify that service delegated intelligence to agents
            mock_agent.analyze_query_intelligence.assert_called_once()
            
            # Verify service didn't do its own intelligence analysis
            intelligence_request = mock_agent.analyze_query_intelligence.call_args[0][0]
            assert isinstance(intelligence_request, IntelligenceRequest)
            assert intelligence_request.query == "test query for intelligence delegation"
            
            # Verify result contains agent intelligence
            assert result is not None
            assert result.get('metadata', {}).get('service_layer') == 'enhanced_query_service'
        
        print("✅ Service properly delegates intelligence to agents")
    
    @pytest.mark.asyncio
    async def test_agent_service_interface_contract(self, agent_service):
        """Test that agent service follows interface contract"""
        
        # Test intelligence analysis contract
        intelligence_request = IntelligenceRequest(
            query="test contract compliance",
            domain="test",
            context={"test": "context"},
            performance_requirements={"max_response_time": 2.0}
        )
        
        result = await agent_service.analyze_query_intelligence(intelligence_request)
        
        # Verify contract compliance
        assert isinstance(result, IntelligenceResult)
        assert hasattr(result, 'primary_intent')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning_trace')
        assert hasattr(result, 'tool_recommendations')
        assert 0 <= result.confidence <= 1.0
        assert result.execution_time >= 0
        
        print("✅ Agent service interface contract validated")
    
    @pytest.mark.asyncio
    async def test_no_business_logic_in_agents(self, agent_service):
        """Test that agents don't contain business logic orchestration"""
        
        # Agents should focus on intelligence, not orchestration
        intelligence_request = IntelligenceRequest(
            query="test business logic separation",
            domain="business"
        )
        
        result = await agent_service.analyze_query_intelligence(intelligence_request)
        
        # Agent should return intelligence insights, not orchestration results
        assert result.primary_intent  # Intelligence analysis
        assert result.tool_recommendations  # Tool selection intelligence
        assert result.reasoning_trace  # Reasoning process
        
        # Agent should NOT return infrastructure results or caching info
        metadata = result.metadata
        assert 'cache_hit' not in metadata  # Business/infrastructure concern
        assert 'infrastructure_results' not in metadata  # Business concern
        
        print("✅ Agents properly separated from business logic")


class TestCoreAgentIntegrationCompliance:
    """Test that Core-Agent integration follows boundary rules"""
    
    @pytest.fixture
    async def integrated_memory_manager(self):
        """Create integrated memory manager for testing"""
        manager = IntegratedMemoryManager(max_memory_mb=50.0)
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_agent_uses_core_infrastructure(self, integrated_memory_manager):
        """Test that agents use core infrastructure, not duplicate it"""
        
        # Verify that agent memory manager uses core bounded infrastructure
        assert hasattr(integrated_memory_manager, 'bounded_manager')
        assert integrated_memory_manager.bounded_manager is not None
        
        # Verify agent intelligence is layered on top of core infrastructure
        assert hasattr(integrated_memory_manager, 'pattern_recognizer')
        assert hasattr(integrated_memory_manager, 'consolidation_engine')
        
        # Test that memory operations use both agent intelligence and core infrastructure
        from agents.base.memory_manager import MemoryEntry, MemoryType, MemoryPriority
        
        memory_entry = MemoryEntry(
            memory_id="test_integration",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.HIGH,
            content={"test": "data"},
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        context = AgentMemoryContext(
            agent_id="test_agent",
            domain="test_domain"
        )
        
        result = await integrated_memory_manager.store_agent_memory(memory_entry, context)
        
        # Verify operation result shows both agent intelligence and core infrastructure
        assert isinstance(result, MemoryOperationResult)
        assert result.success
        assert 'intelligence_insights' in result.__dict__
        assert 'performance_metrics' in result.__dict__
        
        print("✅ Agents properly use core infrastructure with intelligence layer")
    
    @pytest.mark.asyncio
    async def test_core_provides_infrastructure_not_intelligence(self, integrated_memory_manager):
        """Test that core provides infrastructure capabilities, not intelligence"""
        
        # Core should provide bounded memory management
        core_stats = await integrated_memory_manager.bounded_manager.get_comprehensive_stats()
        
        # Core provides infrastructure metrics
        assert 'global_memory' in core_stats
        assert 'caches' in core_stats
        assert 'memory_utilization' in core_stats
        
        # Agent layer provides intelligence analytics
        agent_analytics = await integrated_memory_manager.get_memory_analytics()
        
        # Agent provides intelligence insights
        assert 'intelligence_analytics' in agent_analytics
        assert 'pattern_recognition_accuracy' in agent_analytics.get('intelligence_analytics', {})
        
        print("✅ Clear separation: Core provides infrastructure, Agents provide intelligence")


class TestContractEnforcementMechanisms:
    """Test that contract enforcement mechanisms work correctly"""
    
    def test_contract_monitor_tracks_violations(self):
        """Test that contract monitor properly tracks violations"""
        
        monitor = ContractMonitor()
        
        # Record valid operation
        valid_result = OperationResult(
            status=OperationStatus.SUCCESS,
            execution_time=0.5,
            correlation_id="test-123",
            performance_met=True
        )
        
        monitor.record_operation(
            LayerType.SERVICES, LayerType.AGENTS, "intelligence_request", valid_result
        )
        
        # Record invalid operation (boundary violation)
        try:
            monitor.record_operation(
                LayerType.API, LayerType.AGENTS, "direct_call", valid_result
            )
        except ContractViolationError:
            pass  # Expected violation
        
        # Get compliance metrics
        metrics = monitor.get_compliance_metrics()
        
        assert metrics['total_operations'] >= 1
        assert metrics['contract_violations'] >= 1
        assert 'compliance_rate' in metrics
        
        print("✅ Contract monitor properly tracks violations")
    
    def test_contract_validation_helper(self):
        """Test contract validation helper utilities"""
        
        from config.inter_layer_contracts import ContractImplementationHelper
        
        # Test operation result creation
        result = ContractImplementationHelper.create_operation_result(
            status=OperationStatus.SUCCESS,
            data={"test": "data"},
            layer_source=LayerType.SERVICES,
            correlation_id="test-456"
        )
        
        assert isinstance(result, OperationResult)
        assert result.status == OperationStatus.SUCCESS
        assert result.layer_source == LayerType.SERVICES
        assert result.correlation_id == "test-456"
        
        # Test health status creation
        health = ContractImplementationHelper.create_health_status(
            layer_type=LayerType.AGENTS,
            overall_status="healthy",
            component_health={"reasoning": "operational"},
            performance_metrics={"response_time": 0.8}
        )
        
        assert health.layer_type == LayerType.AGENTS
        assert health.overall_status == "healthy"
        assert health.performance_metrics["response_time"] == 0.8
        
        print("✅ Contract validation helpers working correctly")


class TestEndToEndBoundaryCompliance:
    """Test end-to-end boundary compliance across multiple layers"""
    
    @pytest.mark.asyncio
    async def test_complete_request_flow_boundary_compliance(self):
        """Test that a complete request follows all boundary rules"""
        
        # This would test a complete flow: API -> Services -> Agents -> Core
        # For now, we'll test the key boundaries we've implemented
        
        # Create service with proper agent integration
        query_service = EnhancedQueryService()
        
        # Mock agent service to track boundary compliance
        with patch.object(query_service, 'agent_service') as mock_agent:
            mock_agent.analyze_query_intelligence.return_value = IntelligenceResult(
                primary_intent="end_to_end_test",
                confidence=0.9,
                discovered_domain="test",
                reasoning_trace=MagicMock(),
                tool_recommendations=["vector_search", "graph_traversal"],
                context_insights={"complexity": "medium"},
                performance_met=True,
                execution_time=1.2,
                metadata={"boundary_compliant": True}
            )
            
            # Execute complete workflow
            result = await query_service.process_universal_query(
                query="end-to-end boundary compliance test",
                domain=None,
                max_results=5,
                user_session="test_session",
                context={"test_context": True}
            )
            
            # Verify boundary compliance throughout flow
            assert result is not None
            
            # Verify service delegated to agents (proper boundary)
            mock_agent.analyze_query_intelligence.assert_called_once()
            
            # Verify service handled orchestration (proper responsibility)
            assert 'processing_result' in result
            assert 'performance' in result
            assert 'metadata' in result
            
            # Verify agent intelligence was used (proper integration)
            intelligence_call = mock_agent.analyze_query_intelligence.call_args[0][0]
            assert isinstance(intelligence_call, IntelligenceRequest)
            assert intelligence_call.query == "end-to-end boundary compliance test"
        
        print("✅ End-to-end request flow maintains boundary compliance")
    
    @pytest.mark.asyncio
    async def test_error_handling_preserves_boundaries(self):
        """Test that error handling preserves layer boundaries"""
        
        query_service = EnhancedQueryService()
        
        # Mock agent service to raise an exception
        with patch.object(query_service, 'agent_service') as mock_agent:
            mock_agent.analyze_query_intelligence.side_effect = Exception("Agent failure test")
            
            # Service should handle error gracefully without violating boundaries
            result = await query_service.process_universal_query(
                query="error handling test",
                domain="test"
            )
            
            # Verify service handled error at service layer (proper boundary)
            assert result is not None
            assert result.get('metadata', {}).get('fallback_used') == True
            assert 'error' in result.get('processing_result', {})
            
            # Verify agent was still called (proper delegation)
            mock_agent.analyze_query_intelligence.assert_called_once()
        
        print("✅ Error handling preserves layer boundaries")


async def run_all_boundary_validation_tests():
    """Run all boundary validation tests"""
    
    print("🔍 Starting Layer Boundary Validation Tests\n")
    
    test_classes = [
        TestLayerBoundaryDefinitions(),
        TestServiceAgentBoundaryCompliance(),
        TestCoreAgentIntegrationCompliance(),
        TestContractEnforcementMechanisms(),
        TestEndToEndBoundaryCompliance()
    ]
    
    passed_tests = 0
    total_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n🧪 Running {class_name}")
        print("-" * 60)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, method_name)
                
                # Handle async methods
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                passed_tests += 1
                
            except Exception as e:
                failed_tests.append(f"{class_name}.{method_name}: {str(e)}")
                print(f"❌ {method_name} failed: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 BOUNDARY VALIDATION SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {len(failed_tests)}")
    print(f"📊 Total: {total_tests}")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for failure in failed_tests:
            print(f"   - {failure}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    # Test specific boundary compliance metrics
    compliance_metrics = contract_monitor.get_compliance_metrics()
    print(f"\n📋 Contract Compliance Metrics:")
    print(f"   - Total Operations: {compliance_metrics['total_operations']}")
    print(f"   - Contract Violations: {compliance_metrics['contract_violations']}")
    print(f"   - Compliance Rate: {compliance_metrics['compliance_rate']:.2%}")
    print(f"   - Performance Compliance: {compliance_metrics['performance_compliance_rate']:.2%}")
    
    if success_rate >= 90 and compliance_metrics['compliance_rate'] >= 0.9:
        print("🎉 BOUNDARY VALIDATION PASSED - Layer boundaries properly implemented!")
        return True
    else:
        print("⚠️  BOUNDARY VALIDATION NEEDS ATTENTION - Some boundaries need refinement")
        return False


if __name__ == "__main__":
    """Run validation when executed directly"""
    import sys
    
    async def main():
        success = await run_all_boundary_validation_tests()
        sys.exit(0 if success else 1)
    
    asyncio.run(main())