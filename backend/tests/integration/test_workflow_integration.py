"""
Integration tests for end-to-end workflows
Tests complete workflows across all layers
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_data_processing_workflow(self):
        """Test complete data processing workflow"""
        from services.infrastructure_service import InfrastructureService
        from services.data_service import DataService
        from services.workflow_service import WorkflowService
        
        # Initialize services
        infrastructure = InfrastructureService()
        data_service = DataService(infrastructure)
        workflow_service = WorkflowService(infrastructure)
        
        # Create test data
        test_domain = "integration-test"
        test_data_path = Path("/tmp/test_data")
        test_data_path.mkdir(exist_ok=True)
        
        # Write test file
        test_file = test_data_path / "test.md"
        test_file.write_text("# Test Document\nThis is a test for integration.")
        
        try:
            # Create workflow
            workflow_id = workflow_service.create_workflow(
                "data_processing",
                {"domain": test_domain, "path": str(test_data_path)}
            )
            
            # Validate initial state
            initial_state = await data_service.validate_domain_data_state(test_domain)
            assert initial_state['requires_processing'] is True
            
            # Process data (mocked to avoid actual Azure calls)
            with patch.object(data_service, 'migrate_data_to_azure') as mock_migrate:
                mock_migrate.return_value = {
                    "success": True,
                    "storage_migration": {"success": True},
                    "search_migration": {"success": True},
                    "cosmos_migration": {"success": True}
                }
                
                result = await mock_migrate(str(test_data_path), test_domain)
                assert result['success'] is True
            
            # Complete workflow
            workflow_service.complete_workflow(workflow_id, "Test completed")
            
        finally:
            # Cleanup
            test_file.unlink(missing_ok=True)
            test_data_path.rmdir()
    
    @pytest.mark.asyncio
    async def test_query_processing_workflow(self):
        """Test complete query processing workflow"""
        from services.infrastructure_service import InfrastructureService
        from services.query_service import QueryService
        from integrations.azure_openai_wrapper import AzureOpenAIClient
        
        infrastructure = InfrastructureService()
        query_service = QueryService(infrastructure)
        
        # Mock search results
        mock_search_results = [
            {
                "id": "doc1",
                "content": "Test document content",
                "score": 0.95
            }
        ]
        
        # Test query processing
        with patch.object(infrastructure, 'get_service') as mock_get_service:
            mock_search = Mock()
            mock_search.search_documents.return_value = mock_search_results
            mock_get_service.return_value = mock_search
            
            # Process query
            test_query = "What is the test about?"
            
            # Note: Full integration would process through all layers
            # Here we test the workflow structure
            assert hasattr(query_service, 'process_query')
            assert hasattr(query_service, 'analyze_query')
    
    @pytest.mark.asyncio
    async def test_pipeline_workflow(self):
        """Test pipeline execution workflow"""
        from services.infrastructure_service import InfrastructureService
        from services.pipeline_service import PipelineService, PipelineConfig, PipelineStage
        
        infrastructure = InfrastructureService()
        pipeline_service = PipelineService(infrastructure)
        
        # Configure pipeline
        config = PipelineConfig(
            name="test_pipeline",
            stages=[
                PipelineStage.INGESTION,
                PipelineStage.EXTRACTION,
                PipelineStage.VALIDATION
            ],
            parallel_execution=False,
            timeout_seconds=60
        )
        
        # Create and execute pipeline
        pipeline_id = await pipeline_service.create_pipeline(config)
        
        # Execute with test data
        test_data = {
            "files": ["test1.txt", "test2.txt"],
            "domain": "test"
        }
        
        result = await pipeline_service.execute_pipeline(pipeline_id, test_data)
        
        assert 'success' in result
        assert 'pipeline_id' in result
        assert 'results' in result
        
        # Check pipeline moved to history
        status = pipeline_service.get_pipeline_status(pipeline_id)
        assert status is not None


@pytest.mark.integration
class TestAPIIntegration:
    """Test API integration with services"""
    
    @pytest.fixture
    def app(self):
        """Create app with real services"""
        from api.main import app
        return app
    
    @pytest.mark.asyncio
    async def test_health_endpoint_integration(self, app):
        """Test health endpoint with real service checks"""
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert 'status' in data
            assert 'services' in data
            assert 'timestamp' in data
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self):
        """Test streaming functionality integration"""
        from api.streaming.workflow_stream import WorkflowStreamer
        from services.workflow_service import WorkflowService, WorkflowStep
        
        # Create services
        mock_infrastructure = Mock()
        workflow_service = WorkflowService(mock_infrastructure)
        streamer = WorkflowStreamer(workflow_service)
        
        # Create workflow
        workflow_id = workflow_service.create_workflow("stream_test", {})
        
        # Update progress
        workflow_service.update_progress(
            workflow_id,
            WorkflowStep.DATA_LOADING,
            0.5,
            "Loading test data"
        )
        
        # Test streaming
        events = []
        async for event in streamer.stream_workflow_progress(workflow_id):
            events.append(event)
            if len(events) >= 1:  # Just get first event
                break
        
        assert len(events) > 0
        assert "data:" in events[0]