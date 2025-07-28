"""
Unit tests for business logic services layer
Tests service implementations and business logic
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestInfrastructureService:
    """Test infrastructure service"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test infrastructure service initialization"""
        from services.infrastructure_service import InfrastructureService
        
        service = InfrastructureService()
        assert service is not None
        assert hasattr(service, 'check_all_services_health')
    
    def test_health_check_structure(self):
        """Test health check response structure"""
        from services.infrastructure_service import InfrastructureService
        
        service = InfrastructureService()
        # Mock the clients to avoid actual Azure calls
        service.openai_client = Mock()
        service.search_service = Mock()
        
        health = service.check_all_services_health()
        
        assert 'overall_status' in health
        assert 'services' in health
        assert 'summary' in health


class TestDataService:
    """Test data service"""
    
    @pytest.mark.asyncio
    async def test_data_validation(self):
        """Test data validation methods"""
        from services.data_service import DataService
        
        # Mock infrastructure
        mock_infrastructure = Mock()
        service = DataService(mock_infrastructure)
        
        # Test raw data validation
        result = service._validate_raw_data_directory()
        assert 'directory_exists' in result
        assert 'file_count' in result
    
    def test_processing_requirement_calculation(self):
        """Test processing requirement logic"""
        from services.data_service import DataService
        
        mock_infrastructure = Mock()
        service = DataService(mock_infrastructure)
        
        # Test different scenarios
        blob_state = {"has_data": True}
        search_state = {"has_data": True}
        cosmos_state = {"has_data": True}
        raw_state = {"has_files": True}
        
        requirement = service._calculate_processing_requirement(
            blob_state, search_state, cosmos_state, raw_state
        )
        assert requirement == "data_exists_check_policy"


class TestWorkflowService:
    """Test workflow service"""
    
    def test_workflow_creation(self):
        """Test workflow creation and tracking"""
        from services.workflow_service import WorkflowService
        
        mock_infrastructure = Mock()
        service = WorkflowService(mock_infrastructure)
        
        # Create workflow
        workflow_id = service.create_workflow(
            "test_workflow",
            {"param": "value"}
        )
        
        assert workflow_id is not None
        assert workflow_id.startswith("wf_")
        
        # Check status
        status = service.get_workflow_status(workflow_id)
        assert status is not None
        assert status['status'] == 'created'
    
    def test_progress_tracking(self):
        """Test workflow progress tracking"""
        from services.workflow_service import WorkflowService, WorkflowStep
        
        mock_infrastructure = Mock()
        service = WorkflowService(mock_infrastructure)
        
        # Create and update workflow
        workflow_id = service.create_workflow("test", {})
        service.update_progress(
            workflow_id,
            WorkflowStep.DATA_LOADING,
            0.5,
            "Loading data"
        )
        
        status = service.get_workflow_status(workflow_id)
        assert status['current_step'] == WorkflowStep.DATA_LOADING.value
        assert status['percentage'] == 0.5


class TestQueryService:
    """Test query service"""
    
    @pytest.mark.asyncio
    async def test_query_processing_structure(self):
        """Test query processing structure"""
        from services.query_service import QueryService
        
        mock_infrastructure = Mock()
        service = QueryService(mock_infrastructure)
        
        # Mock search results
        mock_infrastructure.get_service.return_value = AsyncMock()
        
        # Test query structure
        assert hasattr(service, 'process_query')
        assert hasattr(service, 'analyze_query')


class TestPipelineService:
    """Test pipeline service"""
    
    @pytest.mark.asyncio
    async def test_pipeline_creation(self):
        """Test pipeline creation and configuration"""
        from services.pipeline_service import PipelineService, PipelineConfig, PipelineStage
        
        mock_infrastructure = Mock()
        service = PipelineService(mock_infrastructure)
        
        # Create pipeline config
        config = PipelineConfig(
            name="test_pipeline",
            stages=[PipelineStage.INGESTION, PipelineStage.EXTRACTION],
            parallel_execution=False
        )
        
        pipeline_id = await service.create_pipeline(config)
        assert pipeline_id is not None
        assert "test_pipeline" in pipeline_id
        
        # Check status
        status = service.get_pipeline_status(pipeline_id)
        assert status is not None
        assert status['status'] == 'created'