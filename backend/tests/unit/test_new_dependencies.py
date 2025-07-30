"""
Tests for the new proper dependency injection container.
Validates that the DI container properly replaces global state anti-pattern.
"""

import pytest
from unittest.mock import Mock, patch
import asyncio

from api.dependencies_new import (
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


class TestDependencyInjectionContainer:
    """Test the new DI container implementation"""

    def test_container_configuration(self):
        """Test that container is properly configured"""
        # Verify container exists and is properly configured
        assert container is not None
        
        # Verify all providers are configured
        assert hasattr(container, 'azure_settings')
        assert hasattr(container, 'infrastructure_service') 
        assert hasattr(container, 'data_service')
        assert hasattr(container, 'workflow_service')
        assert hasattr(container, 'query_service')
        
        # Verify container has providers
        assert hasattr(container, 'providers')

    @patch('api.dependencies_new.AzureSettings')
    def test_azure_settings_singleton(self, mock_settings):
        """Test that AzureSettings is properly configured as singleton"""
        mock_instance = Mock()
        mock_settings.return_value = mock_instance
        
        # Get settings multiple times
        settings1 = container.azure_settings()
        settings2 = container.azure_settings()
        
        # Should be the same instance (singleton)
        assert settings1 is settings2
        # Should only call constructor once
        mock_settings.assert_called_once()

    @patch('api.dependencies_new.InfrastructureService')
    @patch('api.dependencies_new.AzureSettings')
    def test_infrastructure_service_singleton(self, mock_settings, mock_infrastructure):
        """Test that InfrastructureService is properly configured as singleton"""
        mock_settings_instance = Mock()
        mock_settings.return_value = mock_settings_instance
        mock_infra_instance = Mock()
        mock_infrastructure.return_value = mock_infra_instance
        
        # Get infrastructure service multiple times
        infra1 = container.infrastructure_service()
        infra2 = container.infrastructure_service()
        
        # Should be the same instance (singleton)
        assert infra1 is infra2
        # Should only call constructor once
        mock_infrastructure.assert_called_once_with(azure_settings=mock_settings_instance)

    @patch('api.dependencies_new.QueryService')
    @patch('api.dependencies_new.DataService')
    @patch('api.dependencies_new.InfrastructureService')
    @patch('api.dependencies_new.AzureSettings')
    def test_query_service_factory(self, mock_settings, mock_infrastructure, 
                                  mock_data_service, mock_query_service):
        """Test that QueryService is properly configured as factory"""
        mock_settings_instance = Mock()
        mock_settings.return_value = mock_settings_instance
        mock_infra_instance = Mock()
        mock_infrastructure.return_value = mock_infra_instance
        mock_data_instance = Mock()
        mock_data_service.return_value = mock_data_instance
        mock_query_instance1 = Mock()
        mock_query_instance2 = Mock()
        mock_query_service.side_effect = [mock_query_instance1, mock_query_instance2]
        
        # Get query service multiple times
        query1 = container.query_service()
        query2 = container.query_service()
        
        # Should be different instances (factory)
        assert query1 is not query2
        # Should call constructor twice
        assert mock_query_service.call_count == 2

    @pytest.mark.asyncio
    async def test_dependency_provider_functions(self):
        """Test that FastAPI dependency provider functions work correctly"""
        with patch.object(container, 'infrastructure_service') as mock_provider:
            mock_service = Mock()
            mock_provider.return_value = mock_service
            
            # Test dependency provider function
            result = await get_infrastructure_service(mock_service)
            assert result is mock_service

    @pytest.mark.asyncio
    async def test_initialize_application(self):
        """Test application initialization"""
        # Should not raise exception
        await initialize_application()

    @pytest.mark.asyncio
    async def test_shutdown_application(self):
        """Test application shutdown"""
        # Should not raise exception
        await shutdown_application()

    @pytest.mark.asyncio
    @patch.object(container, 'azure_settings')
    @patch.object(container, 'infrastructure_service')
    @patch.object(container, 'data_service')
    @patch.object(container, 'workflow_service')
    @patch.object(container, 'query_service')
    async def test_validate_dependencies_success(self, mock_query, mock_workflow, 
                                               mock_data, mock_infra, mock_settings):
        """Test successful dependency validation"""
        # Mock all services to return valid instances
        mock_settings.return_value = Mock()
        mock_infra.return_value = Mock()
        mock_data.return_value = Mock()
        mock_workflow.return_value = Mock()
        mock_query.return_value = Mock()
        
        result = await validate_dependencies()
        
        assert result["dependency_injection"] == "healthy"
        assert result["container_wired"] is True
        assert all(result["services_available"].values())

    @pytest.mark.asyncio
    @patch.object(container, 'azure_settings')
    async def test_validate_dependencies_failure(self, mock_settings):
        """Test dependency validation with failure"""
        # Mock settings to raise exception
        mock_settings.side_effect = Exception("Test error")
        
        result = await validate_dependencies()
        
        assert result["dependency_injection"] == "unhealthy"
        assert "error" in result
        assert result["container_wired"] is False

    def test_wire_container(self):
        """Test container wiring"""
        # Should not raise exception
        wire_container()


class TestComparisonWithOldSystem:
    """Test comparison between old global state and new DI system"""

    def test_no_global_variables(self):
        """Verify new system doesn't use global state variables"""
        import api.dependencies_new as new_deps
        
        # Check that there are no global service variables
        global_vars = [name for name in dir(new_deps) if name.startswith('_') and 'service' in name.lower()]
        
        # Should not have any global service variables
        assert len(global_vars) == 0, f"Found global variables: {global_vars}"

    def test_proper_dependency_injection_pattern(self):
        """Verify new system follows proper DI patterns"""
        # Verify container-based approach
        assert hasattr(container, 'providers'), "Container should have providers"
        
        # Verify no setter functions for global state
        import api.dependencies_new as new_deps
        setter_functions = [name for name in dir(new_deps) if name.startswith('set_')]
        
        # Should not have any global setter functions
        assert len(setter_functions) == 0, f"Found setter functions: {setter_functions}"

    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self):
        """Test that services have proper lifecycle management"""
        # Initialize application
        await initialize_application()
        
        # Validate dependencies
        validation_result = await validate_dependencies()
        assert validation_result["dependency_injection"] == "healthy"
        
        # Shutdown application
        await shutdown_application()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])