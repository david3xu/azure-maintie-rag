"""
Simple validation test for new dependency injection implementation.
Tests basic functionality without mocking complex internal behavior.
"""

import pytest
from unittest.mock import Mock

# Test that we can import the new dependencies without errors
def test_import_new_dependencies():
    """Test that new dependencies module can be imported"""
    try:
        from api.dependencies import (
            ApplicationContainer,
            container,
            get_infrastructure_service,
            get_data_service,
            get_query_service,
            get_workflow_service,
            get_azure_settings,
            initialize_application,
            shutdown_application,
            validate_dependencies,
            wire_container
        )
        assert True  # Import successful
    except ImportError as e:
        pytest.fail(f"Failed to import new dependencies: {e}")


def test_container_exists():
    """Test that container is properly instantiated"""
    from api.dependencies import container
    
    assert container is not None
    assert hasattr(container, 'providers')


def test_container_has_all_providers():
    """Test that container has all required providers"""
    from api.dependencies import container
    
    # Check that all expected providers exist
    expected_providers = [
        'azure_settings',
        'infrastructure_service', 
        'data_service',
        'workflow_service',
        'query_service'
    ]
    
    for provider_name in expected_providers:
        assert hasattr(container, provider_name), f"Missing provider: {provider_name}"


@pytest.mark.asyncio
async def test_application_lifecycle():
    """Test basic application lifecycle functions"""
    from api.dependencies import initialize_application, shutdown_application
    
    # These should not raise exceptions
    await initialize_application()
    await shutdown_application()


def test_no_global_state_variables():
    """Test that new implementation doesn't use global state variables"""
    import api.dependencies as new_deps
    
    # Should not have global service variables like old implementation
    global_service_vars = [
        attr for attr in dir(new_deps) 
        if attr.startswith('_') and 'service' in attr.lower()
    ]
    
    assert len(global_service_vars) == 0, f"Found global service variables: {global_service_vars}"


def test_no_setter_functions():
    """Test that new implementation doesn't have global setter functions"""
    import api.dependencies as new_deps
    
    # Should not have setter functions like old implementation
    setter_functions = [
        attr for attr in dir(new_deps) 
        if attr.startswith('set_')
    ]
    
    assert len(setter_functions) == 0, f"Found setter functions: {setter_functions}"


@pytest.mark.asyncio 
async def test_dependency_functions_exist():
    """Test that all FastAPI dependency functions exist and are callable"""
    from api.dependencies import (
        get_infrastructure_service,
        get_data_service, 
        get_query_service,
        get_workflow_service,
        get_azure_settings
    )
    
    # Create mock services for testing
    mock_service = Mock()
    
    # Test that functions are callable (they will fail without proper DI wiring, but shouldn't crash on call)
    dependency_functions = [
        get_infrastructure_service,
        get_data_service,
        get_query_service, 
        get_workflow_service,
        get_azure_settings
    ]
    
    for func in dependency_functions:
        assert callable(func), f"Function {func.__name__} is not callable"


def test_wire_container_function():
    """Test that wire_container function exists and is callable"""
    from api.dependencies import wire_container
    
    assert callable(wire_container)
    
    # Should not raise exception
    wire_container()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])