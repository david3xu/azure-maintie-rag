"""
Test Direct Service Instantiation Fixes - Step 1.4 of IMPLEMENTATION_ROADMAP.md
Validates that direct service instantiation anti-patterns have been eliminated.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import inspect

from services.query_service import QueryServiceWithDI, create_query_service_with_di
from api.endpoints.health_endpoint import health_check, detailed_health_check
from api.endpoints.workflow_endpoint import get_workflow_evidence


class TestDirectInstantiationFixes:
    """Test that direct service instantiation anti-patterns are fixed"""

    def test_query_service_accepts_injected_dependencies(self):
        """Test that QueryServiceWithDI accepts injected dependencies"""
        # Create mock dependencies
        mock_infrastructure = Mock()
        mock_graph_service = Mock()
        mock_performance_service = Mock()
        
        # Create service with injected dependencies
        service = QueryServiceWithDI(
            infrastructure_service=mock_infrastructure,
            graph_service=mock_graph_service,
            performance_service=mock_performance_service
        )
        
        # Verify dependencies are injected
        assert service.infrastructure_service is mock_infrastructure
        assert service._graph_service is mock_graph_service
        assert service._performance_service is mock_performance_service

    def test_query_service_lazy_loading_with_di(self):
        """Test that QueryService uses lazy loading with DI"""
        # Create mock infrastructure with services
        mock_infrastructure = Mock()
        mock_openai = Mock()
        mock_search = Mock()
        mock_cosmos = Mock()
        
        mock_infrastructure.openai_client = mock_openai
        mock_infrastructure.search_service = mock_search
        mock_infrastructure.cosmos_client = mock_cosmos
        
        # Create service
        service = QueryServiceWithDI(infrastructure_service=mock_infrastructure)
        
        # Verify lazy loading uses injected services
        assert service.openai_client is mock_openai
        assert service.search_client is mock_search
        assert service.cosmos_client is mock_cosmos

    def test_query_service_fallback_during_migration(self):
        """Test that QueryService has fallback for migration period"""
        # Create service without dependencies
        service = QueryServiceWithDI()
        
        # Verify fallback instantiation works
        with patch('services.query_service.UnifiedAzureOpenAIClient') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Access client should trigger fallback instantiation
            client = service.openai_client
            mock_openai.assert_called_once()
            assert service._openai_client is mock_client

    @pytest.mark.asyncio
    async def test_query_service_process_universal_query_with_di(self):
        """Test that process_universal_query uses injected services"""
        # Create comprehensive mocks
        mock_infrastructure = Mock()
        mock_performance_service = Mock()
        
        # Mock performance context
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.record_metric = Mock()
        mock_performance_service.create_performance_context.return_value = mock_context
        
        # Create service with mocked dependencies
        service = QueryServiceWithDI(
            infrastructure_service=mock_infrastructure,
            performance_service=mock_performance_service
        )
        
        # Mock the internal methods to avoid complex setup
        service._perform_semantic_search = AsyncMock(return_value={"documents": []})
        service._enhance_with_graph_data = AsyncMock(return_value={"entities": []})
        service._generate_comprehensive_response = AsyncMock(return_value={"response": "test"})
        
        # Mock cache service
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # No cache hit
        mock_cache.set.return_value = None
        service._cache_service = mock_cache
        
        # Execute query
        result = await service.process_universal_query("test query", "test_domain")
        
        # Verify DI was used
        assert result["success"] is True
        assert result["dependency_injection"]["infrastructure_service"] is True
        assert result["dependency_injection"]["services_injected"] is True
        
        # Verify performance service was used
        mock_performance_service.create_performance_context.assert_called_once()
        mock_context.record_metric.assert_called()

    def test_factory_function_creates_service_with_di(self):
        """Test that factory function creates service with proper DI"""
        mock_infra = Mock()
        mock_graph = Mock()
        mock_perf = Mock()
        
        service = create_query_service_with_di(
            infrastructure_service=mock_infra,
            graph_service=mock_graph,
            performance_service=mock_perf
        )
        
        assert isinstance(service, QueryServiceWithDI)
        assert service.infrastructure_service is mock_infra
        assert service._graph_service is mock_graph
        assert service._performance_service is mock_perf


class TestHealthEndpointDIFixes:
    """Test that health endpoint uses proper dependency injection"""

    def test_health_check_requires_infrastructure_dependency(self):
        """Test that health_check function requires infrastructure dependency"""
        # Get function signature
        sig = inspect.signature(health_check)
        
        # Verify it has infrastructure parameter with Depends
        assert 'infrastructure' in sig.parameters
        param = sig.parameters['infrastructure']
        
        # The parameter should have a default (the Depends call)
        assert param.default is not None

    @pytest.mark.asyncio
    async def test_health_check_uses_injected_infrastructure(self):
        """Test that health_check uses injected infrastructure service"""
        # Create mock infrastructure
        mock_infrastructure = AsyncMock()
        mock_infrastructure.initialized = True
        mock_infrastructure.health_check_async.return_value = {
            "status": "healthy",
            "services": {"test": True},
            "summary": {"total_services": 1}
        }
        mock_infrastructure.openai_client = Mock()
        mock_infrastructure.cosmos_client = Mock()
        mock_infrastructure.ml_client = Mock()
        mock_infrastructure._get_initialization_summary.return_value = {"services": {}}
        
        # Call health check with injected infrastructure
        result = await health_check(mock_infrastructure)
        
        # Verify injected service was used
        mock_infrastructure.health_check_async.assert_called_once()
        assert result["status"] == "healthy"
        assert result["architecture"] == "Clean Architecture with DI Container"
        assert result["capabilities"]["tri_modal_search"] is True

    @pytest.mark.asyncio
    async def test_detailed_health_check_uses_injected_infrastructure(self):
        """Test that detailed_health_check uses injected infrastructure service"""
        # Create mock infrastructure
        mock_infrastructure = AsyncMock()
        mock_infrastructure.initialized = True
        mock_infrastructure.health_check_async.return_value = {
            "status": "healthy",
            "services": {"openai": True, "search": True},
            "summary": {"healthy_services": 2, "total_services": 2}
        }
        mock_infrastructure._get_initialization_summary.return_value = {
            "total_initialization_time": 1.5,
            "services": {"total": 2, "successful": 2, "failed": 0}
        }
        
        # Call detailed health check
        result = await detailed_health_check(mock_infrastructure)
        
        # Verify DI compliance is reported
        assert result["component_diagnostics"]["dependency_injection"]["status"] == "healthy"
        assert result["component_diagnostics"]["dependency_injection"]["global_state_eliminated"] is True
        assert result["architecture_compliance"]["dependency_injection"] is True
        assert result["overall_status"] in ["healthy", "degraded"]


class TestWorkflowEndpointDIFixes:
    """Test that workflow endpoint uses proper dependency injection"""

    def test_workflow_evidence_requires_workflow_service_dependency(self):
        """Test that get_workflow_evidence requires workflow service dependency"""
        # Get function signature
        sig = inspect.signature(get_workflow_evidence)
        
        # Verify it has workflow_service parameter with Depends
        assert 'workflow_service' in sig.parameters
        param = sig.parameters['workflow_service']
        
        # The parameter should have a default (the Depends call)
        assert param.default is not None

    @pytest.mark.asyncio
    async def test_workflow_evidence_uses_injected_service(self):
        """Test that get_workflow_evidence uses injected workflow service"""
        # Create mock workflow service
        mock_workflow_service = Mock()
        mock_workflow_service.evidence_collectors = {
            "test_workflow": Mock()
        }
        
        # Mock the evidence collector
        mock_collector = mock_workflow_service.evidence_collectors["test_workflow"]
        mock_collector.generate_workflow_evidence_report = AsyncMock(return_value={
            "total_steps": 5,
            "azure_services_used": ["openai", "search"],
            "total_processing_time_ms": 1500,
            "quality_assessment": {"success_rate": 0.95},
            "evidence_chain": ["step1", "step2"]
        })
        
        # Call endpoint with injected service
        result = await get_workflow_evidence(
            workflow_id="test_workflow",
            workflow_service=mock_workflow_service
        )
        
        # Verify injected service was used
        assert result["workflow_id"] == "test_workflow"
        assert result["service_injection"]["dependency_injected"] is True
        assert result["service_injection"]["global_state_avoided"] is True
        assert result["service_injection"]["workflow_service_type"] == "Mock"


class TestDIPatternCompliance:
    """Test overall DI pattern compliance"""

    def test_fixed_services_eliminate_global_state(self):
        """Test that fixed services don't use global state"""
        # Test QueryServiceWithDI
        service = QueryServiceWithDI()
        
        # Should not have global state - everything should be instance-based
        assert hasattr(service, 'infrastructure_service')
        assert hasattr(service, '_graph_service')
        assert hasattr(service, '_performance_service')
        
        # Lazy properties should exist
        assert hasattr(service, 'openai_client')
        assert hasattr(service, 'search_client')
        assert hasattr(service, 'cosmos_client')

    def test_fixed_services_support_injection(self):
        """Test that fixed services support dependency injection"""
        mock_infra = Mock()
        mock_graph = Mock()
        mock_perf = Mock()
        
        # Should accept injected dependencies
        service = QueryServiceWithDI(
            infrastructure_service=mock_infra,
            graph_service=mock_graph,
            performance_service=mock_perf
        )
        
        # Verify injection worked
        assert service.infrastructure_service is mock_infra
        assert service._graph_service is mock_graph
        assert service._performance_service is mock_perf

    def test_backward_compatibility_maintained(self):
        """Test that backward compatibility is maintained during migration"""
        # Old-style instantiation should still work
        from services.query_service import QueryService
        
        service = QueryService()
        assert isinstance(service, QueryServiceWithDI)
        
        # With infrastructure service
        mock_infra = Mock()
        service_with_infra = QueryService(infrastructure_service=mock_infra)
        assert service_with_infra.infrastructure_service is mock_infra

    def test_lazy_loading_prevents_circular_dependencies(self):
        """Test that lazy loading prevents circular dependency issues"""
        service = QueryServiceWithDI()
        
        # Properties should be lazy-loaded
        assert service._graph_service is None
        assert service._performance_service is None
        assert service._cache_service is None
        
        # Accessing properties should trigger lazy loading
        with patch('services.query_service.GraphService') as mock_graph:
            mock_instance = Mock()
            mock_graph.return_value = mock_instance
            
            graph_service = service.graph_service
            mock_graph.assert_called_once()
            assert service._graph_service is mock_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])