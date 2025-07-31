#!/usr/bin/env python3
"""
Simple Layer Boundary Validation Test

This script validates the core boundary contract implementation
without importing agent components that have circular dependencies.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_boundary_contracts_core():
    """Test that the core boundary contracts are properly implemented"""
    
    print("🔍 Testing Core Layer Boundary Contracts")
    print("=" * 60)
    
    try:
        from config.inter_layer_contracts import (
            LayerType,
            OperationStatus,
            OperationResult,
            LayerBoundaryEnforcer,
            ContractViolationError,
            ContractMonitor,
            ContractImplementationHelper,
            LayerHealthStatus
        )
        print("✅ All core boundary contracts imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import boundary contracts: {e}")
        return False
    
    # Test 1: Layer dependency validation
    print("\n📋 Test 1: Layer Dependency Rules")
    print("-" * 40)
    
    # Test allowed dependencies
    allowed_deps = [
        (LayerType.API, LayerType.SERVICES, "API can call Services"),
        (LayerType.SERVICES, LayerType.AGENTS, "Services can call Agents"),
        (LayerType.SERVICES, LayerType.CORE, "Services can call Core"),
        (LayerType.AGENTS, LayerType.TOOLS, "Agents can call Tools"),
        (LayerType.AGENTS, LayerType.CORE, "Agents can call Core"),
        (LayerType.TOOLS, LayerType.CORE, "Tools can call Core")
    ]
    
    for source, target, description in allowed_deps:
        try:
            result = LayerBoundaryEnforcer.validate_dependency(source, target)
            if result:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - validation returned False")
                return False
        except Exception as e:
            print(f"❌ {description} - {e}")
            return False
    
    # Test forbidden dependencies
    forbidden_deps = [
        (LayerType.API, LayerType.AGENTS, "API cannot directly call Agents"),
        (LayerType.API, LayerType.CORE, "API cannot directly call Core"),
        (LayerType.SERVICES, LayerType.API, "Services cannot call API"),
        (LayerType.AGENTS, LayerType.SERVICES, "Agents cannot call Services"),
        (LayerType.CORE, LayerType.AGENTS, "Core cannot call Agents")
    ]
    
    for source, target, description in forbidden_deps:
        try:
            LayerBoundaryEnforcer.validate_dependency(source, target)
            print(f"❌ {description} - should have been forbidden but was allowed")
            return False
        except ContractViolationError:
            print(f"✅ {description}")
        except Exception as e:
            print(f"❌ {description} - unexpected error: {e}")
            return False
    
    # Test 2: Operation result validation
    print("\n📋 Test 2: Operation Result Contract Validation")
    print("-" * 40)
    
    # Valid operation result
    valid_result = OperationResult(
        status=OperationStatus.SUCCESS,
        data={"boundary_test": "data"},
        execution_time=0.5,
        correlation_id="boundary-test-123",
        layer_source=LayerType.SERVICES,
        performance_met=True
    )
    
    try:
        result = LayerBoundaryEnforcer.validate_operation_result(valid_result, LayerType.SERVICES)
        if result:
            print("✅ Valid OperationResult correctly validated")
        else:
            print("❌ Valid OperationResult validation returned False")
            return False
    except Exception as e:
        print(f"❌ Valid OperationResult validation failed: {e}")
        return False
    
    # Invalid operation result (wrong type)
    try:
        LayerBoundaryEnforcer.validate_operation_result(
            {"invalid": "result"}, LayerType.SERVICES
        )
        print("❌ Invalid operation result should have been rejected")
        return False
    except ContractViolationError:
        print("✅ Invalid operation result correctly rejected")
    except Exception as e:
        print(f"❌ Invalid operation result unexpected error: {e}")
        return False
    
    # Test 3: Contract Implementation Helpers
    print("\n📋 Test 3: Contract Implementation Helpers")
    print("-" * 40)
    
    # Test operation result helper
    try:
        helper_result = ContractImplementationHelper.create_operation_result(
            status=OperationStatus.SUCCESS,
            data={"helper_test": "data"},
            layer_source=LayerType.AGENTS,
            correlation_id="helper-test-456",
            execution_time=0.3
        )
        
        if (isinstance(helper_result, OperationResult) and
            helper_result.status == OperationStatus.SUCCESS and
            helper_result.layer_source == LayerType.AGENTS):
            print("✅ Operation result helper working correctly")
        else:
            print("❌ Operation result helper produced incorrect result")
            return False
    except Exception as e:
        print(f"❌ Operation result helper failed: {e}")
        return False
    
    # Test health status helper
    try:
        health_status = ContractImplementationHelper.create_health_status(
            layer_type=LayerType.CORE,
            overall_status="healthy",
            component_health={"memory": "operational", "monitoring": "healthy"},
            performance_metrics={"avg_response_time": 0.2, "throughput": 100.0}
        )
        
        if (isinstance(health_status, LayerHealthStatus) and
            health_status.layer_type == LayerType.CORE and
            health_status.overall_status == "healthy"):
            print("✅ Health status helper working correctly")
        else:
            print("❌ Health status helper produced incorrect result")
            return False
    except Exception as e:
        print(f"❌ Health status helper failed: {e}")
        return False
    
    # Test 4: Contract Registry (basic test)
    print("\n📋 Test 4: Contract Registry")
    print("-" * 40)
    
    try:
        from config.inter_layer_contracts import ContractRegistry
        
        registry = ContractRegistry()
        
        # Test registration
        class MockImplementation:
            def test_method(self):
                return "test"
        
        mock_impl = MockImplementation()
        registry.register_contract_implementation(
            LayerType.SERVICES, "test_contract", mock_impl
        )
        
        # Test retrieval
        retrieved = registry.get_contract_implementation(LayerType.SERVICES, "test_contract")
        if retrieved == mock_impl:
            print("✅ Contract registry working correctly")
        else:
            print("❌ Contract registry retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Contract registry test failed: {e}")
        return False
    
    print("\n🎉 All Core Boundary Contract Tests Passed!")
    return True

def test_layer_boundary_files_exist():
    """Test that the boundary implementation files exist"""
    
    print("\n🔍 Testing Layer Boundary Implementation Files")
    print("=" * 60)
    
    expected_files = [
        ("contracts/inter_layer_contracts.py", "Inter-layer contracts framework"),
        ("agents/base/agent_service_interface.py", "Service-Agent boundary contract"),
        ("agents/universal_agent_service.py", "Universal agent service implementation"),
        ("services/enhanced_query_service.py", "Enhanced query service with boundaries"),
        ("agents/base/integrated_memory_manager.py", "Core-Agent memory integration"),
        ("docs/architecture/LAYER_BOUNDARY_DEFINITIONS.md", "Layer boundary documentation"),
        ("docs/architecture/ARCHITECTURE_REFACTORING_PLAN.md", "Refactoring plan documentation")
    ]
    
    all_exist = True
    
    for file_path, description in expected_files:
        full_path = os.path.join(os.path.dirname(__file__), '..', '..', file_path)
        if os.path.exists(full_path):
            print(f"✅ {description}")
        else:
            print(f"❌ Missing: {description} ({file_path})")
            all_exist = False
    
    return all_exist

def main():
    """Run all boundary validation tests"""
    
    print("🎯 LAYER BOUNDARY VALIDATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Core Boundary Contracts", test_boundary_contracts_core),
        ("Implementation Files", test_layer_boundary_files_exist)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL BOUNDARY VALIDATION TESTS PASSED!")
        print("✨ Layer boundaries are properly implemented and enforced.")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Boundary implementation needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)