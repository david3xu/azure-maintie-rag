"""
Shared models for query processing endpoints
Used across multi-modal, structured, and comparison endpoints
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Request model for universal queries"""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Query in natural language"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of search results to return"
    )
    include_explanations: bool = Field(
        default=True,
        description="Include explanations and reasoning in response"
    )
    enable_safety_warnings: bool = Field(
        default=True,
        description="Include safety warnings in response"
    )
    response_format: str = Field(
        default="detailed",
        description="Response format: 'detailed', 'summary', or 'minimal'"
    )

    @validator('query')
    def validate_query(cls, v):
        """Validate query content"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator('response_format')
    def validate_response_format(cls, v):
        """Validate response format"""
        valid_formats = ['detailed', 'summary', 'minimal']
        if v not in valid_formats:
            raise ValueError(f"response_format must be one of: {valid_formats}")
        return v


class QueryResponse(BaseModel):
    """Response model for universal queries"""

    query: str = Field(description="Original query")
    generated_response: str = Field(description="Generated response")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the response"
    )
    processing_time: float = Field(description="Processing time in seconds")

    # Enhanced query information
    query_analysis: Dict[str, Any] = Field(description="Query analysis results")
    expanded_concepts: List[str] = Field(description="Expanded concepts")

    # Search and sources
    sources: List[str] = Field(description="Source document IDs")
    citations: List[str] = Field(description="Formatted citations")
    search_results_count: int = Field(description="Number of search results used")

    # Safety and quality
    safety_warnings: List[str] = Field(description="Safety warnings and considerations")
    quality_indicators: Dict[str, Any] = Field(description="Response quality indicators")

    # Metadata
    timestamp: float = Field(description="Response timestamp")
    model_info: Dict[str, Any] = Field(description="Model and system information")


class QuerySuggestionResponse(BaseModel):
    """Response model for query suggestions"""

    suggestions: List[str] = Field(description="Suggested maintenance queries")
    categories: Dict[str, List[str]] = Field(description="Suggestions by category")


class ComparisonRequest(BaseModel):
    """Request model for comparison queries"""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Query to compare between methods"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of search results to return"
    )
    include_explanations: bool = Field(
        default=True,
        description="Include explanations and reasoning in response"
    )
    enable_safety_warnings: bool = Field(
        default=True,
        description="Include safety warnings in response"
    )