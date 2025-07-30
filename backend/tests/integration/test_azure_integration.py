"""
Integration tests for Azure services
Tests actual Azure service integration (requires credentials)
"""

import pytest
import os
from unittest.mock import patch


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_ENDPOINT"),
    reason="Azure credentials not configured"
)
class TestAzureIntegration:
    """Test actual Azure service integration"""
    
    @pytest.mark.asyncio
    async def test_openai_integration(self):
        """Test Azure OpenAI integration"""
        from core.azure_openai.openai_client import UnifiedAzureOpenAIClient as AzureOpenAIClient
        
        client = AzureOpenAIClient()
        
        # Test simple completion
        response = await client.generate_completion(
            "Test prompt",
            max_tokens=10
        )
        
        assert response is not None
        assert 'content' in response
    
    @pytest.mark.asyncio
    async def test_storage_integration(self):
        """Test Azure Storage integration"""
        from core.azure_storage.storage_factory import get_storage_factory
        
        factory = get_storage_factory()
        storage_client = factory.get_rag_storage_client()
        
        # Test container operations
        test_container = "test-integration-container"
        
        # Create container
        created = await storage_client.create_container(test_container)
        assert created is True
        
        # Check existence
        exists = await storage_client.container_exists(test_container)
        assert exists is True
        
        # Cleanup
        await storage_client.delete_container(test_container)
    
    @pytest.mark.asyncio
    async def test_search_integration(self):
        """Test Azure Cognitive Search integration"""
        from services.infrastructure_service import InfrastructureService
        
        infrastructure = InfrastructureService()
        search_client = infrastructure.get_service('search')
        
        if search_client:
            # Test index operations
            test_index = "test-integration-index"
            
            # Create index
            created = await search_client.create_index(test_index)
            assert created is True
            
            # Check existence
            exists = await search_client.index_exists(test_index)
            assert exists is True
            
            # Cleanup
            await search_client.delete_index(test_index)
    
    def test_service_health_check(self):
        """Test integrated service health check"""
        from services.infrastructure_service import InfrastructureService
        
        infrastructure = InfrastructureService()
        health = infrastructure.check_all_services_health()
        
        assert 'overall_status' in health
        assert 'services' in health
        assert health['summary']['total_services'] > 0


@pytest.mark.integration
class TestServiceIntegration:
    """Test service layer integration"""
    
    @pytest.mark.asyncio
    async def test_data_workflow_integration(self):
        """Test data service with infrastructure"""
        from services.infrastructure_service import InfrastructureService
        from services.data_service import DataService
        
        infrastructure = InfrastructureService()
        data_service = DataService(infrastructure)
        
        # Test domain validation
        validation = await data_service.validate_domain_data_state("test-domain")
        
        assert 'azure_blob_storage' in validation
        assert 'azure_cognitive_search' in validation
        assert 'azure_cosmos_db' in validation
        assert 'raw_data_directory' in validation
    
    @pytest.mark.asyncio
    async def test_workflow_service_integration(self):
        """Test workflow service with infrastructure"""
        from services.infrastructure_service import InfrastructureService
        from services.workflow_service import WorkflowService
        
        infrastructure = InfrastructureService()
        workflow_service = WorkflowService(infrastructure)
        
        # Create test workflow
        workflow_id = workflow_service.create_workflow(
            "integration_test",
            {"test": "data"}
        )
        
        # Update progress
        from services.workflow_service import WorkflowStep
        workflow_service.update_progress(
            workflow_id,
            WorkflowStep.INITIALIZATION,
            0.1,
            "Starting integration test"
        )
        
        # Complete workflow
        final_status = workflow_service.complete_workflow(
            workflow_id,
            "Integration test completed"
        )
        
        assert final_status['status'] == 'completed'
        assert 'final_cost_analysis' in final_status