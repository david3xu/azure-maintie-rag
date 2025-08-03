"""
PydanticAI Agent Models

Structured input/output models for Azure RAG system with Pydantic V2 BaseModel
validation and SLA enforcement.
"""

from .requests import (
    QueryRequest,
    VectorSearchRequest,
    GraphSearchRequest,
    TriModalSearchRequest,
    DomainDetectionRequest,
    PatternLearningRequest,
    SearchType
)

from .responses import (
    TriModalSearchResult,
    DomainDetectionResult,
    PatternLearningResult,
    SearchDocument,
    GraphEntity,
    GraphRelationship,
    AgentHealthStatus,
    SearchResultType,
    ConfidenceLevel
)

__all__ = [
    # Request models
    "QueryRequest",
    "VectorSearchRequest", 
    "GraphSearchRequest",
    "TriModalSearchRequest",
    "DomainDetectionRequest",
    "PatternLearningRequest",
    
    # Response models
    "TriModalSearchResult",
    "DomainDetectionResult", 
    "PatternLearningResult",
    "SearchDocument",
    "GraphEntity",
    "GraphRelationship",
    "AgentHealthStatus",
    
    # Enums
    "SearchType",
    "SearchResultType",
    "ConfidenceLevel"
]