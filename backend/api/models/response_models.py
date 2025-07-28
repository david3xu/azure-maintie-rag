"""
Response Models
Separated from query_models.py for better organization
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Human-readable message about the operation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: Optional[str] = Field(None, description="Specific error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class DataResponse(BaseResponse):
    """Generic data response model"""
    data: Dict[str, Any] = Field(description="Response data payload")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QueryResponse(BaseResponse):
    """Query-specific response model"""
    query_id: str = Field(description="Unique query identifier")
    answer: str = Field(description="Generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
    processing_time_ms: int = Field(description="Processing time in milliseconds")


class WorkflowResponse(BaseResponse):
    """Workflow execution response model"""
    workflow_id: str = Field(description="Unique workflow identifier")
    status: str = Field(description="Workflow status")
    steps_completed: int = Field(description="Number of steps completed")
    total_steps: int = Field(description="Total number of steps")
    results: Optional[Dict[str, Any]] = Field(None, description="Workflow results")


class HealthResponse(BaseResponse):
    """Health check response model"""
    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    services: Dict[str, Dict[str, Any]] = Field(description="Individual service health")
    uptime_seconds: float = Field(description="API uptime in seconds")


class SearchResponse(BaseResponse):
    """Search results response model"""
    query: str = Field(description="Search query")
    total_results: int = Field(description="Total number of results found")
    results: List[Dict[str, Any]] = Field(description="Search results")
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="Search facets")
    next_page_token: Optional[str] = Field(None, description="Token for pagination")


class GraphResponse(BaseResponse):
    """Knowledge graph response model"""
    entities: List[Dict[str, Any]] = Field(description="Graph entities")
    relationships: List[Dict[str, Any]] = Field(description="Graph relationships")
    metadata: Dict[str, Any] = Field(description="Graph metadata")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="Data for graph visualization")