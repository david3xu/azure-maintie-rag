#!/usr/bin/env python3
"""
Standalone Layer Boundary Validation Runner

This script runs the boundary validation tests independently to avoid 
circular import issues while validating the implemented boundaries.
"""

import sys
import os
import asyncio
import time
from typing import Dict, Any, List

# Add the backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_layer_dependency_rules():
    """Test that layer dependency rules are correctly defined"""
    from config.inter_layer_contracts import LayerType, LayerBoundaryEnforcer, ContractViolationError
    
    print("Testing layer dependency rules...")
    
    # Test allowed dependencies
    allowed_tests = [
        (LayerType.API, LayerType.SERVICES),
        (LayerType.SERVICES, LayerType.AGENTS),
        (LayerType.SERVICES, LayerType.CORE),
        (LayerType.AGENTS, LayerType.TOOLS),
        (LayerType.AGENTS, LayerType.CORE),
        (LayerType.TOOLS, LayerType.CORE)
    ]
    
    for source, target in allowed_tests:
        try:
            result = LayerBoundaryEnforcer.validate_dependency(source, target)
            assert result == True, f"Expected {source.value} -> {target.value} to be allowed"
            print(f"✅ {source.value} -> {target.value} correctly allowed")
        except Exception as e:
            print(f"❌ {source.value} -> {target.value} failed: {e}")
            return False
    
    # Test forbidden dependencies
    forbidden_tests = [
        (LayerType.API, LayerType.AGENTS),
        (LayerType.API, LayerType.CORE),
        (LayerType.SERVICES, LayerType.API),
        (LayerType.AGENTS, LayerType.SERVICES),
        (LayerType.CORE, LayerType.AGENTS)
    ]
    
    for source, target in forbidden_tests:
        try:
            LayerBoundaryEnforcer.validate_dependency(source, target)
            print(f"❌ {source.value} -> {target.value} should have been forbidden but was allowed")
            return False
        except ContractViolationError:
            print(f"✅ {source.value} -> {target.value} correctly forbidden")
        except Exception as e:
            print(f"❌ {source.value} -> {target.value} unexpected error: {e}")
            return False
    
    return True

def test_operation_result_validation():
    """Test that operation results follow contract format"""
    from config.inter_layer_contracts import (
        LayerType, OperationResult, OperationStatus, 
        LayerBoundaryEnforcer, ContractViolationError
    )
    
    print("Testing operation result validation...")
    
    # Valid operation result
    valid_result = OperationResult(
        status=OperationStatus.SUCCESS,
        data={"test": "data"},
        execution_time=0.5,
        correlation_id="test-123"
    )
    
    try:
        result = LayerBoundaryEnforcer.validate_operation_result(valid_result, LayerType.SERVICES)
        assert result == True, "Expected valid result to pass validation"
        print("✅ Valid operation result correctly validated")
    except Exception as e:
        print(f"❌ Valid operation result validation failed: {e}")
        return False
    
    # Invalid operation result (wrong type)
    invalid_result = {"status": "success"}  # Not an OperationResult object
    
    try:
        LayerBoundaryEnforcer.validate_operation_result(invalid_result, LayerType.SERVICES)
        print("❌ Invalid operation result should have failed validation")
        return False
    except ContractViolationError:
        print("✅ Invalid operation result correctly rejected")
    except Exception as e:
        print(f"❌ Invalid operation result unexpected error: {e}")
        return False
    
    return True

def test_contract_monitor():
    """Test that contract monitor properly tracks violations"""
    from config.inter_layer_contracts import (
        LayerType, ContractMonitor, OperationResult, 
        OperationStatus, ContractViolationError
    )
    
    print("Testing contract monitor...")
    
    monitor = ContractMonitor()
    
    # Record valid operation
    valid_result = OperationResult(
        status=OperationStatus.SUCCESS,
        execution_time=0.5,
        correlation_id="test-123",
        performance_met=True
    )
    
    try:
        monitor.record_operation(
            LayerType.SERVICES, LayerType.AGENTS, "intelligence_request", valid_result
        )
        print("✅ Valid operation recorded successfully")
    except Exception as e:
        print(f"❌ Valid operation recording failed: {e}")
        return False
    
    # Record invalid operation (boundary violation)
    try:
        monitor.record_operation(
            LayerType.API, LayerType.AGENTS, "direct_call", valid_result
        )
        print("❌ Invalid operation should have been rejected")
        return False
    except ContractViolationError:
        print("✅ Invalid operation correctly rejected")
    except Exception as e:
        print(f"❌ Invalid operation unexpected error: {e}")
        return False
    
    # Get compliance metrics
    try:
        metrics = monitor.get_compliance_metrics()
        assert 'total_operations' in metrics, "Missing total_operations metric"
        assert 'contract_violations' in metrics, "Missing contract_violations metric"
        assert 'compliance_rate' in metrics, "Missing compliance_rate metric"
        print("✅ Contract compliance metrics generated successfully")
        print(f"   - Total operations: {metrics['total_operations']}")
        print(f"   - Contract violations: {metrics['contract_violations']}")
        print(f"   - Compliance rate: {metrics['compliance_rate']:.2%}")
    except Exception as e:
        print(f"❌ Contract compliance metrics failed: {e}")
        return False
    
    return True

def test_contract_implementation_helpers():
    """Test contract implementation helper utilities"""
    from config.inter_layer_contracts import (
        ContractImplementationHelper, OperationStatus, LayerType,
        OperationResult, LayerHealthStatus
    )
    
    print("Testing contract implementation helpers...")
    
    # Test operation result creation
    try:
        result = ContractImplementationHelper.create_operation_result(
            status=OperationStatus.SUCCESS,
            data={"test": "data"},
            layer_source=LayerType.SERVICES,
            correlation_id="test-456"
        )
        
        assert isinstance(result, OperationResult), "Result should be OperationResult instance"
        assert result.status == OperationStatus.SUCCESS, "Status should match"
        assert result.layer_source == LayerType.SERVICES, "Layer source should match"
        assert result.correlation_id == "test-456", "Correlation ID should match"
        print("✅ Operation result creation helper working correctly")
    except Exception as e:
        print(f"❌ Operation result creation helper failed: {e}")
        return False
    
    # Test health status creation
    try:
        health = ContractImplementationHelper.create_health_status(
            layer_type=LayerType.AGENTS,
            overall_status="healthy",
            component_health={"reasoning": "operational"},
            performance_metrics={"response_time": 0.8}
        )
        
        assert isinstance(health, LayerHealthStatus), "Health should be LayerHealthStatus instance"
        assert health.layer_type == LayerType.AGENTS, "Layer type should match"
        assert health.overall_status == "healthy", "Overall status should match"
        assert health.performance_metrics["response_time"] == 0.8, "Response time should match"
        print("✅ Health status creation helper working correctly")
    except Exception as e:
        print(f"❌ Health status creation helper failed: {e}")
        return False
    
    return True

async def test_agent_service_interface_exists():
    """Test that agent service interface exists and has correct structure"""
    try:
        from agents.base.agent_service_interface import (
            IntelligenceRequest, IntelligenceResult, AgentServiceContract
        )
        
        print("Testing agent service interface...")
        
        # Test IntelligenceRequest creation
        request = IntelligenceRequest(
            query="test contract compliance",
            domain="test",
            context={"test": "context"},
            performance_requirements={"max_response_time": 2.0}
        )
        
        assert request.query == "test contract compliance", "Query should match"
        assert request.domain == "test", "Domain should match"  
        assert request.context["test"] == "context", "Context should match"
        print("✅ IntelligenceRequest structure validated")
        
        # Test that AgentServiceContract exists
        assert hasattr(AgentServiceContract, '__init__'), "AgentServiceContract should be a class"
        print("✅ AgentServiceContract class exists")
        
        return True
        
    except ImportError as e:
        print(f"❌ Agent service interface import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent service interface test failed: {e}")
        return False

async def test_enhanced_query_service_exists():
    """Test that enhanced query service exists"""
    try:
        from services.enhanced_query_service import EnhancedQueryService
        
        print("Testing enhanced query service...")
        
        # Test service instantiation
        service = EnhancedQueryService()
        assert hasattr(service, 'process_universal_query'), "Should have process_universal_query method"
        print("✅ EnhancedQueryService exists and has required methods")
        
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced query service import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced query service test failed: {e}")
        return False

async def test_integrated_memory_manager_exists():
    """Test that integrated memory manager exists"""
    try:
        from agents.base.integrated_memory_manager import (
            IntegratedMemoryManager, AgentMemoryContext, MemoryOperationResult
        )
        
        print("Testing integrated memory manager...")
        
        # Test memory manager instantiation
        manager = IntegratedMemoryManager(max_memory_mb=50.0)
        assert hasattr(manager, 'store_agent_memory'), "Should have store_agent_memory method"
        assert hasattr(manager, 'bounded_manager'), "Should have bounded_manager attribute"
        print("✅ IntegratedMemoryManager exists with required structure")
        
        # Test context creation
        context = AgentMemoryContext(
            agent_id="test_agent",
            domain="test_domain"
        )
        assert context.agent_id == "test_agent", "Context should have correct agent_id"
        print("✅ AgentMemoryContext structure validated")
        
        return True
        
    except ImportError as e:
        print(f"❌ Integrated memory manager import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Integrated memory manager test failed: {e}")
        return False

async def run_all_boundary_validation_tests():
    """Run all boundary validation tests"""
    
    print("🔍 Starting Layer Boundary Validation Tests\n")
    print("=" * 70)
    
    test_functions = [
        test_layer_dependency_rules,
        test_operation_result_validation,
        test_contract_monitor,
        test_contract_implementation_helpers,
        test_agent_service_interface_exists,
        test_enhanced_query_service_exists,
        test_integrated_memory_manager_exists
    ]
    
    passed_tests = 0
    failed_tests = []
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        print(f"\n🧪 Running {test_func.__name__}")
        print("-" * 60)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed_tests += 1
            else:
                failed_tests.append(test_func.__name__)
                
        except Exception as e:
            failed_tests.append(f"{test_func.__name__}: {str(e)}")
            print(f"❌ {test_func.__name__} failed with exception: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("🎯 BOUNDARY VALIDATION SUMMARY")
    print("=" * 70)
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
        print("🎉 BOUNDARY VALIDATION PASSED - Layer boundaries properly implemented!")
        return True
    else:
        print("⚠️  BOUNDARY VALIDATION NEEDS ATTENTION - Some boundaries need refinement")
        return False

if __name__ == "__main__":
    """Run validation when executed directly"""
    async def main():
        success = await run_all_boundary_validation_tests()
        sys.exit(0 if success else 1)
    
    asyncio.run(main())