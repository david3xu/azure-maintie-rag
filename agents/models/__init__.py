"""
PydanticAI Agent Models

Structured input/output models for Azure RAG system with Pydantic V2 BaseModel
validation and SLA enforcement.
"""

from .requests import (
    DomainDetectionRequest,
    GraphSearchRequest,
    PatternLearningRequest,
    QueryRequest,
    SearchType,
    TriModalSearchRequest,
    VectorSearchRequest,
)
from .responses import (
    AgentHealthStatus,
    ConfidenceLevel,
    DomainDetectionResult,
    GraphEntity,
    GraphRelationship,
    PatternLearningResult,
    SearchDocument,
    SearchResultType,
    TriModalSearchResult,
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
    "ConfidenceLevel",
]
