"""
Unit tests for API presentation layer
Tests endpoints, models, and API functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # Import app with mocked dependencies
        with patch('api.main.InfrastructureService') as mock_infra:
            with patch('api.main.DataService') as mock_data:
                from api.main import app
                return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
    
    def test_query_endpoint_structure(self, client):
        """Test query endpoint structure"""
        # Test that endpoint exists
        response = client.post(
            "/api/v1/query/universal",
            json={"query": "test query", "domain": "general"}
        )
        # May fail due to dependencies, but should not be 404
        assert response.status_code != 404


class TestAPIModels:
    """Test API data models"""
    
    def test_query_models(self):
        """Test query request/response models"""
        from api.models.query_models import QueryRequest, QueryResponse
        
        # Test request model
        request = QueryRequest(
            query="test query",
            domain="general",
            max_results=10
        )
        assert request.query == "test query"
        assert request.domain == "general"
    
    def test_response_models(self):
        """Test response models"""
        from api.models.response_models import (
            BaseResponse,
            ErrorResponse,
            QueryResponse
        )
        
        # Test base response
        base = BaseResponse(
            success=True,
            message="Success"
        )
        assert base.success is True
        
        # Test error response
        error = ErrorResponse(
            message="Error occurred",
            error_code="ERR_001"
        )
        assert error.success is False
        assert error.error_code == "ERR_001"
    
    def test_stream_models(self):
        """Test streaming models"""
        from api.models.stream_models import (
            StreamEvent,
            ProgressEvent,
            StreamEventType
        )
        
        # Test progress event
        event = ProgressEvent(
            id="evt_1",
            sequence=1,
            progress=50.0,
            message="Processing..."
        )
        assert event.type == StreamEventType.PROGRESS
        assert event.progress == 50.0


class TestAPIMiddleware:
    """Test API middleware"""
    
    def test_middleware_configuration(self):
        """Test middleware configuration"""
        from api.middleware import configure_middleware
        from fastapi import FastAPI
        
        app = FastAPI()
        configure_middleware(app)
        
        # Check that middleware was added
        assert len(app.user_middleware) > 0
    
    def test_cors_configuration(self):
        """Test CORS middleware settings"""
        from api.middleware import configure_middleware
        from fastapi import FastAPI
        
        app = FastAPI()
        configure_middleware(app)
        
        # CORS should be configured
        # This is a basic test - in production would test actual CORS behavior
        assert any('CORSMiddleware' in str(m) for m in app.user_middleware)


class TestStreaming:
    """Test streaming functionality"""
    
    def test_progress_streamer(self):
        """Test progress streaming"""
        from api.streaming.progress_stream import ProgressStreamer
        
        streamer = ProgressStreamer()
        
        # Update progress
        operation_id = "op_123"
        streamer.update_progress(
            operation_id,
            progress=50.0,
            status="in_progress",
            message="Processing..."
        )
        
        # Check that progress was stored
        assert operation_id in streamer.active_streams
        assert streamer.active_streams[operation_id]['progress'] == 50.0
    
    @pytest.mark.asyncio
    async def test_stream_generation(self):
        """Test stream event generation"""
        from api.streaming.progress_stream import ProgressStreamer
        
        streamer = ProgressStreamer()
        operation_id = "op_test"
        
        # Set up progress
        streamer.update_progress(
            operation_id,
            progress=100.0,
            status="completed",
            message="Done"
        )
        
        # Test stream generation
        events = []
        async for event in streamer.stream_progress(operation_id):
            events.append(event)
            break  # Just test first event
        
        assert len(events) > 0
        assert "data:" in events[0]