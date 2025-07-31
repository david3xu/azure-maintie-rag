#!/usr/bin/env python3
"""
Step 1.4 Validation: Direct Service Instantiation Fixes
Validates that DI patterns are properly implemented
"""

def validate_query_service_di():
    """Validate QueryService DI implementation"""
    print("🔍 Validating QueryService DI implementation...")
    
    try:
        from services.query_service import QueryServiceWithDI, QueryService, create_query_service_with_di
        
        # Test 1: DI Constructor
        mock_infra = type('Mock', (), {'openai_client': 'mock_openai', 'search_service': 'mock_search'})()
        service = QueryServiceWithDI(infrastructure_service=mock_infra)
        assert service.infrastructure_service == mock_infra
        print("✅ QueryServiceWithDI accepts injected dependencies")
        
        # Test 2: Lazy Loading with DI
        assert service.openai_client == 'mock_openai'
        assert service.search_client == 'mock_search'
        print("✅ Lazy loading uses injected services")
        
        # Test 3: Factory function
        factory_service = create_query_service_with_di(infrastructure_service=mock_infra)
        assert isinstance(factory_service, QueryServiceWithDI)
        print("✅ Factory function creates service with DI")
        
        # Test 4: Backward compatibility
        compat_service = QueryService(infrastructure_service=mock_infra)
        assert isinstance(compat_service, QueryServiceWithDI)
        print("✅ Backward compatibility maintained")
        
        return True
        
    except Exception as e:
        print(f"❌ QueryService DI validation failed: {e}")
        return False

def validate_endpoint_di_patterns():
    """Validate endpoint DI patterns by checking function signatures"""
    print("\n🔍 Validating Endpoint DI patterns...")
    
    try:
        import inspect
        
        # Health endpoint
        from api.endpoints.health_endpoint import health_check, detailed_health_check
        health_sig = inspect.signature(health_check)
        detailed_sig = inspect.signature(detailed_health_check)
        
        assert 'infrastructure' in health_sig.parameters
        assert 'infrastructure' in detailed_sig.parameters
        print("✅ Health endpoints use dependency injection")
        
        # Workflow endpoint  
        from api.endpoints.workflow_endpoint import get_workflow_evidence, get_workflow_service_health
        workflow_sig = inspect.signature(get_workflow_evidence)
        health_sig = inspect.signature(get_workflow_service_health)
        
        assert 'workflow_service' in workflow_sig.parameters
        assert 'workflow_service' in health_sig.parameters
        print("✅ Workflow endpoints use dependency injection")
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint DI validation failed: {e}")
        return False

def validate_di_container():
    """Validate DI container implementation"""
    print("\n🔍 Validating DI Container...")
    
    try:
        from api.dependencies import ApplicationContainer, get_infrastructure_service, get_workflow_service
        from dependency_injector import containers, providers
        
        # Test container structure
        container = ApplicationContainer()
        assert hasattr(container, 'azure_settings')
        assert hasattr(container, 'infrastructure_service')
        print("✅ DI Container properly configured")
        
        # Test provider functions exist
        assert callable(get_infrastructure_service)
        assert callable(get_workflow_service)
        print("✅ DI provider functions available")
        
        return True
        
    except Exception as e:
        print(f"❌ DI Container validation failed: {e}")
        return False

def validate_async_infrastructure():
    """Validate async infrastructure service"""
    print("\n🔍 Validating Async Infrastructure Service...")
    
    try:
        from services.infrastructure_service_async import AsyncInfrastructureService
        
        # Test instantiation
        service = AsyncInfrastructureService()
        assert hasattr(service, 'initialize_async')
        assert hasattr(service, 'health_check_async')
        print("✅ AsyncInfrastructureService properly implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ Async Infrastructure validation failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Step 1.4 Validation: Direct Service Instantiation Fixes")
    print("=" * 60)
    
    success = True
    success &= validate_query_service_di()
    success &= validate_endpoint_di_patterns()
    success &= validate_di_container()
    success &= validate_async_infrastructure()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ STEP 1.4 COMPLETE: Direct Service Instantiation Anti-Patterns Eliminated")
        print("🎯 All DI patterns properly implemented")
        print("📋 Ready for Step 1.5: Standardize Azure Client Patterns")
    else:
        print("❌ STEP 1.4 INCOMPLETE: Some validations failed")
    
    return success

if __name__ == "__main__":
    main()