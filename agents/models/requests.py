"""
PydanticAI Agent Request Models

Structured input models for Azure RAG system using Pydantic V2 BaseModel
with comprehensive validation and SLA enforcement.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

# Clean configuration (CODING_STANDARDS compliant)
# Simple security patterns
DANGEROUS_PATTERNS = [
    r"<script",
    r"javascript:",
    r"eval\(",
    r"exec\(",
    r"import\s+os",
    r"subprocess",
]

# Backward compatibility
class PatternConfig:
    dangerous_patterns = DANGEROUS_PATTERNS

get_pattern_recognition_config = lambda: PatternConfig()


class SearchType(str, Enum):
    """Supported search types for tri-modal search"""

    VECTOR = "vector"
    GRAPH = "graph"
    GNN = "gnn"
    ALL = "all"


# No hardcoded domain types - domains are discovered from data/raw/ subdirectories


class QueryRequest(BaseModel):
    """
    Structured query request with V2 validation for PydanticAI agent tools.

    Enforces SLA requirements:
    - Query length limits for sub-3s performance
    - Result count limits for memory efficiency
    - Domain validation for accuracy
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User query text with length optimized for sub-3s response",
    )

    domain: Optional[str] = Field(
        default=None,
        description="Target domain name (from data/raw/ subdirectories, or None for auto-detect)",
    )

    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return (SLA: memory optimization)",
    )

    search_types: List[SearchType] = Field(
        default=[SearchType.ALL],
        description="Search types to execute (vector, graph, gnn, or all)",
    )

    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for query processing"
    )

    include_metadata: bool = Field(
        default=True, description="Include search metadata in response"
    )

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results",
    )

    @validator("query")
    def validate_query_content(cls, v):
        """Validate query content for security and performance"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        # Basic SQL injection prevention (from centralized configuration)
        pattern_config = get_pattern_recognition_config()
        dangerous_patterns = pattern_config.dangerous_patterns
        query_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                raise ValueError(
                    f"Query contains potentially dangerous pattern: {pattern}"
                )

        return v.strip()

    class Config:
        """Pydantic V2 configuration"""

        use_enum_values = True
        extra = "forbid"  # Strict validation - no extra fields allowed
        str_strip_whitespace = True


class VectorSearchRequest(BaseModel):
    """Structured request for vector search operations"""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    domain: Optional[str] = Field(
        default=None, description="Domain name from data/raw/ subdirectories"
    )
    include_metadata: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True
        extra = "forbid"


class GraphSearchRequest(BaseModel):
    """Structured request for graph search operations"""

    query: str = Field(..., min_length=1, max_length=500)
    max_depth: int = Field(default=3, ge=1, le=5)
    domain: Optional[str] = Field(
        default=None, description="Domain name from data/raw/ subdirectories"
    )
    relationship_types: List[str] = Field(
        default_factory=lambda: [
            "connects"
        ],  # Minimal fallback - should be learned from domain patterns
        max_items=20,
    )
    max_results: int = Field(default=10, ge=1, le=50)

    class Config:
        use_enum_values = True
        extra = "forbid"


class TriModalSearchRequest(BaseModel):
    """Structured request for tri-modal search (Vector + Graph + GNN)"""

    query: str = Field(..., min_length=1, max_length=500)
    search_types: List[SearchType] = Field(default=[SearchType.ALL])
    domain: Optional[str] = Field(
        default=None, description="Domain name from data/raw/ subdirectories"
    )
    max_results: int = Field(default=10, ge=1, le=50)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = Field(default=True)
    parallel_execution: bool = Field(
        default=True, description="Execute searches in parallel for sub-3s performance"
    )

    class Config:
        use_enum_values = True
        extra = "forbid"


class DomainDetectionRequest(BaseModel):
    """Structured request for domain detection operations"""

    text: str = Field(..., min_length=1, max_length=2000)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    include_probabilities: bool = Field(default=True)
    max_domains: int = Field(default=3, ge=1, le=10)

    class Config:
        use_enum_values = True
        extra = "forbid"


class PatternLearningRequest(BaseModel):
    """Structured request for pattern learning operations"""

    data: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    learning_type: Literal["incremental", "batch", "reinforcement"] = Field(
        default="incremental"
    )
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_patterns: int = Field(default=50, ge=1, le=200)

    class Config:
        use_enum_values = True
        extra = "forbid"
