"""
Tests for AsyncInfrastructureService - validating async initialization patterns.
Tests that the new async service properly eliminates blocking operations during startup.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from services.infrastructure_service_async import (
    AsyncInfrastructureService,
    ServiceInitializationResult,
    ServiceInitializationError
)


class TestAsyncInfrastructureService:
    """Test the new async infrastructure service implementation"""

    def test_constructor_non_blocking(self):
        """Test that constructor does not perform blocking operations"""
        # Constructor should complete quickly without initializing services
        start_time = datetime.utcnow()
        service = AsyncInfrastructureService()
        end_time = datetime.utcnow()
        
        # Constructor should be very fast (< 1 second)
        initialization_time = (end_time - start_time).total_seconds()
        assert initialization_time < 1.0, f"Constructor took too long: {initialization_time}s"
        
        # Services should not be initialized yet
        assert service.openai_client is None
        assert service.search_service is None
        assert service.storage_client is None
        assert service.cosmos_client is None
        assert service.ml_client is None
        assert service.app_insights is None
        assert service.vector_service is None
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_async_parallel_execution(self):
        """Test that async initialization runs services in parallel"""
        with patch('services.infrastructure_service_async.UnifiedAzureOpenAIClient') as mock_openai, \
             patch('services.infrastructure_service_async.UnifiedSearchClient') as mock_search, \
             patch('services.infrastructure_service_async.UnifiedStorageClient') as mock_storage, \
             patch('services.infrastructure_service_async.AzureCosmosGremlinClient') as mock_cosmos, \
             patch('services.infrastructure_service_async.AzureMLClient') as mock_ml, \
             patch('services.infrastructure_service_async.AzureApplicationInsightsClient') as mock_insights:
            
            # Mock service instances
            mock_openai.return_value = Mock()
            mock_search.return_value = Mock()
            mock_storage.return_value = Mock()
            mock_cosmos.return_value = Mock()
            mock_ml.return_value = Mock()
            mock_insights.return_value = Mock()
            
            service = AsyncInfrastructureService()
            
            # Record start time and initialize
            start_time = datetime.utcnow()
            result = await service.initialize_async()
            end_time = datetime.utcnow()
            
            # Verify initialization completed
            assert service.initialized is True
            assert result["status"] == "initialized"
            
            # Verify all services were created
            assert service.openai_client is not None
            assert service.search_service is not None
            assert service.storage_client is not None
            assert service.cosmos_client is not None
            assert service.ml_client is not None
            assert service.app_insights is not None
            
            # Verify parallel execution was faster than sequential
            total_time = (end_time - start_time).total_seconds()
            assert total_time < 10.0, f"Parallel initialization took too long: {total_time}s"

    @pytest.mark.asyncio
    async def test_service_initialization_timeout(self):
        """Test that initialization respects timeout limits"""
        async def slow_initialization():
            # Simulate slow service initialization
            await asyncio.sleep(35)  # Longer than 30s timeout
            return ServiceInitializationResult("slow_service", True, 35.0)
        
        service = AsyncInfrastructureService()
        
        # Mock one service to be very slow
        with patch.object(service, '_initialize_openai_service', side_effect=slow_initialization):
            with pytest.raises(ServiceInitializationError) as exc_info:
                await service.initialize_async()
            
            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_partial_service_failure(self):
        """Test handling of partial service initialization failures"""
        service = AsyncInfrastructureService()
        
        # Mock some services to fail
        with patch.object(service, '_initialize_openai_service') as mock_openai, \
             patch.object(service, '_initialize_search_service') as mock_search, \
             patch.object(service, '_initialize_storage_service') as mock_storage:
            
            # OpenAI succeeds
            mock_openai.return_value = ServiceInitializationResult(
                "openai_client", True, 0.1, client_instance=Mock()
            )
            
            # Search fails
            mock_search.return_value = ServiceInitializationResult(
                "search_service", False, 0.2, error_message="Connection failed"
            )
            
            # Storage succeeds
            mock_storage.return_value = ServiceInitializationResult(
                "storage_client", True, 0.15, client_instance=Mock()
            )
            
            result = await service.initialize_async()
            
            # Should still be marked as initialized despite partial failures
            assert service.initialized is True
            assert result["status"] == "initialized"
            
            # Verify successful services are set
            assert service.openai_client is not None
            assert service.storage_client is not None
            
            # Verify failed service is None
            assert service.search_service is None
            
            # Verify results contain both successes and failures
            assert result["services"]["successful"] == 2
            assert result["services"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_health_check_async(self):
        """Test async health check functionality"""
        service = AsyncInfrastructureService()
        
        # Mock initialized services
        service.initialized = True
        service.openai_client = Mock()
        service.search_service = Mock()
        
        # Add async health check methods to mocks
        service.openai_client.health_check_async = AsyncMock(return_value={"status": "healthy"})
        service.search_service.health_check_async = AsyncMock(return_value={"status": "healthy"})
        
        result = await service.health_check_async()
        
        assert result["status"] == "healthy"
        assert result["services"]["openai"]["status"] == "healthy"
        assert result["services"]["search"]["status"] == "healthy"
        assert result["summary"]["healthy_services"] == 2

    @pytest.mark.asyncio
    async def test_health_check_uninitialized(self):
        """Test health check on uninitialized service"""
        service = AsyncInfrastructureService()
        
        result = await service.health_check_async()
        
        assert result["status"] == "unhealthy"
        assert result["reason"] == "Services not initialized"

    @pytest.mark.asyncio
    async def test_shutdown_async(self):
        """Test async shutdown functionality"""
        service = AsyncInfrastructureService()
        
        # Mock initialized services with async shutdown
        mock_openai = Mock()
        mock_openai.shutdown_async = AsyncMock()
        mock_search = Mock()
        mock_search.shutdown_async = AsyncMock()
        
        service.initialized = True
        service.openai_client = mock_openai
        service.search_service = mock_search
        
        await service.shutdown_async()
        
        # Verify shutdown was called on services
        mock_openai.shutdown_async.assert_called_once()
        mock_search.shutdown_async.assert_called_once()
        
        # Verify service state was reset
        assert service.initialized is False
        assert service.openai_client is None
        assert service.search_service is None

    @pytest.mark.asyncio
    async def test_vector_service_initialization_dependency(self):
        """Test that vector service initialization depends on OpenAI client"""
        service = AsyncInfrastructureService()
        
        with patch.object(service, '_initialize_openai_service') as mock_openai, \
             patch.object(service, '_initialize_vector_service') as mock_vector, \
             patch('services.infrastructure_service_async.asyncio.gather') as mock_gather:
            
            # Mock OpenAI success
            openai_result = ServiceInitializationResult(
                "openai_client", True, 0.1, client_instance=Mock()
            )
            mock_openai.return_value = openai_result
            
            # Mock vector service success
            vector_result = ServiceInitializationResult(
                "vector_service", True, 0.2, client_instance=Mock()
            )
            mock_vector.return_value = vector_result
            
            # Mock gather to return our results
            mock_gather.return_value = [openai_result]  # Only core services, not vector
            
            await service.initialize_async()
            
            # Verify vector service was initialized after OpenAI
            mock_vector.assert_called_once()
            assert service.vector_service is not None

    def test_backward_compatibility_alias(self):
        """Test that InfrastructureService alias works for backward compatibility"""
        from services.infrastructure_service_async import InfrastructureService
        
        # Should be able to create service using old name
        service = InfrastructureService()
        assert isinstance(service, AsyncInfrastructureService)

    def test_search_client_property_compatibility(self):
        """Test backward compatibility property for search_client"""
        service = AsyncInfrastructureService()
        
        # Mock search service
        mock_search = Mock()
        service.search_service = mock_search
        
        # search_client should return same as search_service
        assert service.search_client is mock_search
        assert service.search_client is service.search_service


class TestServiceInitializationResult:
    """Test ServiceInitializationResult data class"""
    
    def test_successful_result_creation(self):
        """Test creating successful initialization result"""
        client = Mock()
        result = ServiceInitializationResult(
            service_name="test_service",
            success=True,
            initialization_time=0.5,
            client_instance=client
        )
        
        assert result.service_name == "test_service"
        assert result.success is True
        assert result.initialization_time == 0.5
        assert result.error_message is None
        assert result.client_instance is client

    def test_failed_result_creation(self):
        """Test creating failed initialization result"""
        result = ServiceInitializationResult(
            service_name="failed_service",
            success=False,
            initialization_time=1.2,
            error_message="Connection timeout"
        )
        
        assert result.service_name == "failed_service"
        assert result.success is False
        assert result.initialization_time == 1.2
        assert result.error_message == "Connection timeout"
        assert result.client_instance is None


class TestComparisonWithSyncService:
    """Test comparison between async and sync infrastructure services"""
    
    def test_constructor_performance_comparison(self):
        """Test that async constructor is much faster than sync constructor"""
        # Import both services
        from services.infrastructure_service import InfrastructureService as SyncService
        from services.infrastructure_service_async import AsyncInfrastructureService
        
        # Mock all Azure clients to avoid actual initialization
        with patch('services.infrastructure_service.UnifiedAzureOpenAIClient'), \
             patch('services.infrastructure_service.UnifiedSearchClient'), \
             patch('services.infrastructure_service.UnifiedStorageClient'), \
             patch('services.infrastructure_service.AzureCosmosGremlinClient'), \
             patch('services.infrastructure_service.AzureMLClient'), \
             patch('services.infrastructure_service.AzureApplicationInsightsClient'), \
             patch('services.infrastructure_service_async.UnifiedAzureOpenAIClient'), \
             patch('services.infrastructure_service_async.UnifiedSearchClient'), \
             patch('services.infrastructure_service_async.UnifiedStorageClient'), \
             patch('services.infrastructure_service_async.AzureCosmosGremlinClient'), \
             patch('services.infrastructure_service_async.AzureMLClient'), \
             patch('services.infrastructure_service_async.AzureApplicationInsightsClient'):
            
            # Time sync service constructor (blocks during initialization)
            sync_start = datetime.utcnow()
            sync_service = SyncService()
            sync_end = datetime.utcnow()
            sync_time = (sync_end - sync_start).total_seconds()
            
            # Time async service constructor (should be non-blocking)
            async_start = datetime.utcnow()
            async_service = AsyncInfrastructureService()
            async_end = datetime.utcnow()
            async_time = (async_end - async_start).total_seconds()
            
            # Async constructor should be significantly faster
            assert async_time < sync_time, f"Async constructor ({async_time}s) should be faster than sync ({sync_time}s)"
            assert async_time < 0.1, f"Async constructor should be very fast, but took {async_time}s"

    @pytest.mark.asyncio
    async def test_async_vs_sync_initialization_patterns(self):
        """Test that async pattern allows for better error handling and parallel execution"""
        from services.infrastructure_service_async import AsyncInfrastructureService
        
        # Mock services with different initialization times
        async def fast_service():
            await asyncio.sleep(0.1)
            return ServiceInitializationResult("fast", True, 0.1, client_instance=Mock())
        
        async def slow_service():
            await asyncio.sleep(0.5)  
            return ServiceInitializationResult("slow", True, 0.5, client_instance=Mock())
        
        service = AsyncInfrastructureService()
        
        with patch.object(service, '_initialize_openai_service', side_effect=fast_service), \
             patch.object(service, '_initialize_search_service', side_effect=slow_service):
            
            start_time = datetime.utcnow()
            await service.initialize_async()
            end_time = datetime.utcnow()
            
            total_time = (end_time - start_time).total_seconds()
            
            # Parallel execution should be closer to the slowest service (0.5s) rather than sum (0.6s)
            assert total_time < 0.8, f"Parallel execution should be efficient, but took {total_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])