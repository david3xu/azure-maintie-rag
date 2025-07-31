"""
Simple validation test for AsyncInfrastructureService core functionality.
Tests basic async patterns without complex mocking.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch

from services.infrastructure_service import (
    AsyncInfrastructureService,
    ServiceInitializationResult,
    ServiceInitializationError
)


class TestAsyncInfrastructureBasics:
    """Test basic async infrastructure service functionality"""

    def test_non_blocking_constructor(self):
        """Test that constructor is non-blocking"""
        start_time = datetime.utcnow()
        service = AsyncInfrastructureService()
        end_time = datetime.utcnow()
        
        # Constructor should be very fast
        duration = (end_time - start_time).total_seconds()
        assert duration < 0.5  # Should be nearly instantaneous
        
        # Services should not be initialized
        assert not service.initialized
        assert service.openai_client is None
        assert service.search_service is None

    def test_service_initialization_result(self):
        """Test ServiceInitializationResult data structure"""
        # Successful result
        success_result = ServiceInitializationResult(
            service_name="test_service",
            success=True,
            initialization_time=0.123,
            client_instance="mock_client"
        )
        
        assert success_result.service_name == "test_service"
        assert success_result.success is True
        assert success_result.initialization_time == 0.123
        assert success_result.client_instance == "mock_client"
        assert success_result.error_message is None
        
        # Failed result
        failure_result = ServiceInitializationResult(
            service_name="failed_service",
            success=False,
            initialization_time=0.456,
            error_message="Connection failed"
        )
        
        assert failure_result.service_name == "failed_service"
        assert failure_result.success is False
        assert failure_result.initialization_time == 0.456
        assert failure_result.error_message == "Connection failed"
        assert failure_result.client_instance is None

    @pytest.mark.asyncio
    async def test_initialize_async_basic_flow(self):
        """Test basic async initialization flow"""
        service = AsyncInfrastructureService()
        
        # Mock all Azure clients to avoid real connections
        with patch('services.infrastructure_service.UnifiedAzureOpenAIClient'), \
             patch('services.infrastructure_service.UnifiedSearchClient'), \
             patch('services.infrastructure_service.UnifiedStorageClient'), \
             patch('services.infrastructure_service.AzureCosmosGremlinClient'), \
             patch('services.infrastructure_service.AzureMLClient'), \
             patch('services.infrastructure_service.AzureApplicationInsightsClient'), \
             patch('services.vector_service.VectorService'):
            
            # Should not be initialized initially
            assert not service.initialized
            
            # Initialize asynchronously
            result = await service.initialize_async()
            
            # Should be initialized after async init
            assert service.initialized
            assert result["status"] == "initialized"
            assert "total_initialization_time" in result
            assert result["services"]["total"] > 0

    @pytest.mark.asyncio
    async def test_repeated_initialization(self):
        """Test that repeated initialization is handled correctly"""
        service = AsyncInfrastructureService()
        
        with patch('services.infrastructure_service.UnifiedAzureOpenAIClient'), \
             patch('services.infrastructure_service.UnifiedSearchClient'), \
             patch('services.infrastructure_service.UnifiedStorageClient'), \
             patch('services.infrastructure_service.AzureCosmosGremlinClient'), \
             patch('services.infrastructure_service.AzureMLClient'), \
             patch('services.infrastructure_service.AzureApplicationInsightsClient'), \
             patch('services.vector_service.VectorService'):
            
            # First initialization
            result1 = await service.initialize_async()
            assert service.initialized
            
            # Second initialization should return quickly
            start_time = datetime.utcnow()
            result2 = await service.initialize_async()
            end_time = datetime.utcnow()
            
            # Should complete very quickly (already initialized)
            duration = (end_time - start_time).total_seconds()
            assert duration < 0.1
            
            # Should still be initialized
            assert service.initialized
            assert result2["status"] == "initialized"

    @pytest.mark.asyncio 
    async def test_health_check_uninitialized(self):
        """Test health check on uninitialized service"""
        service = AsyncInfrastructureService()
        
        result = await service.health_check_async()
        
        assert result["status"] == "unhealthy"
        assert result["reason"] == "Services not initialized"
        assert result["services"] == {}

    def test_backward_compatibility(self):
        """Test backward compatibility features"""
        service = AsyncInfrastructureService()
        
        # Test alias import
        from services.infrastructure_service import InfrastructureService
        alias_service = InfrastructureService()
        assert isinstance(alias_service, AsyncInfrastructureService)
        
        # Test search_client property
        service.search_service = "mock_search"
        assert service.search_client == "mock_search"

    @pytest.mark.asyncio
    async def test_initialization_summary(self):
        """Test initialization summary generation"""
        service = AsyncInfrastructureService()
        
        # Before initialization
        summary = service._get_initialization_summary()
        assert summary["status"] == "not_initialized"
        assert summary["services"] == []
        
        # Mock successful initialization
        service.initialized = True
        service.initialization_start_time = datetime.utcnow()
        service.initialization_end_time = datetime.utcnow()
        service.initialization_results = [
            ServiceInitializationResult("service1", True, 0.1),
            ServiceInitializationResult("service2", False, 0.2, error_message="Failed")
        ]
        
        summary = service._get_initialization_summary()
        assert summary["status"] == "initialized"
        assert summary["services"]["successful"] == 1
        assert summary["services"]["failed"] == 1
        assert summary["services"]["total"] == 2
        assert len(summary["successful_services"]) == 1
        assert len(summary["failed_services"]) == 1


class TestComparisonWithSyncService:
    """Compare async vs sync infrastructure service"""
    
    def test_constructor_performance_difference(self):
        """Test that async constructor is much faster than sync"""
        from services.infrastructure_service import InfrastructureService as SyncService
        
        # Mock all Azure clients for both services
        with patch('services.infrastructure_service.UnifiedAzureOpenAIClient'), \
             patch('services.infrastructure_service.UnifiedSearchClient'), \
             patch('services.infrastructure_service.UnifiedStorageClient'), \
             patch('services.infrastructure_service.AzureCosmosGremlinClient'), \
             patch('services.infrastructure_service.AzureMLClient'), \
             patch('services.infrastructure_service.AzureApplicationInsightsClient'), \
             patch('services.vector_service.VectorService'):
            
            # Time sync service constructor
            sync_start = datetime.utcnow()
            sync_service = SyncService()
            sync_end = datetime.utcnow()
            sync_time = (sync_end - sync_start).total_seconds()
            
            # Time async service constructor  
            async_start = datetime.utcnow()
            async_service = AsyncInfrastructureService()
            async_end = datetime.utcnow()
            async_time = (async_end - async_start).total_seconds()
            
            # Async should be much faster
            assert async_time < sync_time * 0.5  # At least 50% faster
            assert async_time < 0.1  # Should be very fast
            
            # Sync service should be initialized immediately
            assert sync_service.initialized  # Sync initializes in constructor
            
            # Async service should not be initialized yet
            assert not async_service.initialized  # Async waits for explicit init


if __name__ == "__main__":
    pytest.main([__file__, "-v"])