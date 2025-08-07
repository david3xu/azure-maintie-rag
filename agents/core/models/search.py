"""
Search Request and Response Models
=================================

Search-related data models for the Universal Search system including
tri-modal search orchestration, domain detection, and result synthesis.
These models handle Vector, Graph, and GNN search coordination.

This module provides:
- Unified query request/response models
- Vector, Graph, and GNN search configurations
- Tri-modal search orchestration models
- Domain detection and pattern learning requests
- Search result synthesis and coordination
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from agents.core.constants import StatisticalConstants

from .agents import GNNSearchConfig, GraphSearchConfig, VectorSearchConfig
from .base import (
    BaseAnalysisResult,
    BaseRequest,
    BaseResponse,
    PydanticAIContextualModel,
    SearchType,
)

# =============================================================================
# SEARCH REQUEST MODELS
# =============================================================================


class QueryRequest(BaseRequest):
    """Unified query request model - eliminates duplicates across agents"""

    query: str = Field(min_length=3, max_length=500, description="User query text")
    domain: Optional[str] = Field(
        default=None, description="Target domain (auto-detect if None)"
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum results to return"
    )
    search_types: List[SearchType] = Field(
        default=[SearchType.ALL], description="Search types to execute"
    )
    confidence_threshold: float = Field(
        default=StatisticalConstants.MIN_DOMAIN_CONFIDENCE, ge=0.0, le=1.0
    )
    include_metadata: bool = Field(default=True, description="Include search metadata")
    parallel_execution: bool = Field(
        default=True, description="Execute searches in parallel"
    )


class VectorSearchRequest(BaseRequest):
    """Vector search specific request"""

    query: str = Field(min_length=1, max_length=500, description="Search query")
    config: VectorSearchConfig = Field(
        default_factory=VectorSearchConfig, description="Vector search configuration"
    )
    domain: Optional[str] = Field(default=None, description="Target domain")


class GraphSearchRequest(BaseRequest):
    """Graph search specific request"""

    query: str = Field(min_length=1, max_length=500, description="Search query")
    config: GraphSearchConfig = Field(
        default_factory=GraphSearchConfig, description="Graph search configuration"
    )
    domain: Optional[str] = Field(default=None, description="Target domain")


class TriModalSearchRequest(BaseRequest):
    """Tri-modal search request (Vector + Graph + GNN)"""

    query: str = Field(min_length=1, max_length=500, description="Search query")
    search_types: List[SearchType] = Field(
        default=[SearchType.ALL], description="Search modalities"
    )
    vector_config: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    graph_config: GraphSearchConfig = Field(default_factory=GraphSearchConfig)
    gnn_config: GNNSearchConfig = Field(default_factory=GNNSearchConfig)
    domain: Optional[str] = Field(default=None, description="Target domain")
    max_results: int = Field(default=10, ge=1, le=50)
    confidence_threshold: float = Field(
        default=StatisticalConstants.MIN_DOMAIN_CONFIDENCE, ge=0.0, le=1.0
    )
    parallel_execution: bool = Field(default=True)


class DomainDetectionRequest(BaseRequest):
    """Domain detection request"""

    text: str = Field(min_length=1, max_length=2000, description="Text to analyze")
    confidence_threshold: float = Field(
        default=StatisticalConstants.DOMAIN_CLASSIFICATION_THRESHOLD, ge=0.0, le=1.0
    )
    include_probabilities: bool = Field(
        default=True, description="Include probability scores"
    )
    max_domains: int = Field(
        default=3, ge=1, le=10, description="Maximum domains to return"
    )


class PatternLearningRequest(BaseRequest):
    """Pattern learning request"""

    data: List[Dict[str, Any]] = Field(
        min_items=1, max_items=1000, description="Training data"
    )
    learning_type: Literal["incremental", "batch", "reinforcement"] = Field(
        default="incremental"
    )
    confidence_threshold: float = Field(
        default=StatisticalConstants.MIN_DOMAIN_CONFIDENCE, ge=0.0, le=1.0
    )
    max_patterns: int = Field(
        default=50, ge=1, le=200, description="Maximum patterns to extract"
    )


# =============================================================================
# SEARCH RESPONSE MODELS
# =============================================================================


class SearchResult(BaseModel):
    """Individual search result"""

    content: str = Field(description="Result content")
    confidence: float = Field(ge=0.0, le=1.0, description="Result confidence score")
    source: str = Field(description="Result source identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")


class SearchResponse(BaseResponse):
    """Unified search response model"""

    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(ge=0, description="Total results found")
    search_types_used: List[SearchType] = Field(description="Search types executed")
    domain_detected: Optional[str] = Field(default=None, description="Detected domain")
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    synthesis_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Result synthesis score"
    )


class DomainDetectionResult(BaseResponse):
    """Domain detection response"""

    domain: str = Field(description="Detected domain")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    reasoning: str = Field(description="Detection reasoning")
    alternative_domains: List[Dict[str, float]] = Field(
        default_factory=list, description="Alternative domains with scores"
    )


class AnalysisResult(BaseAnalysisResult):
    """Generic analysis result"""

    domain: str = Field(description="Analysis domain")
    analysis_type: str = Field(description="Type of analysis performed")
    results: Dict[str, Any] = Field(description="Analysis results")
    recommendations: List[str] = Field(
        default_factory=list, description="Analysis recommendations"
    )


# =============================================================================
# TRI-MODAL SEARCH RESULT MODELS
# =============================================================================


@dataclass
class TriModalSearchResult:
    """Tri-modal search orchestration result"""

    query: str
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    gnn_results: List[Dict[str, Any]]
    synthesis_score: float
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchCoordinationResult:
    """Search coordination result"""

    coordination_successful: bool
    search_strategies_used: List[str]
    total_results: int
    coordination_time: float
    error_messages: List[str] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


@dataclass
class TriModalResult:
    """Tri-modal search result"""

    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    gnn_results: List[Dict[str, Any]]
    combined_score: float
    result_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.result_metadata is None:
            self.result_metadata = {}


class ModalityResult(PydanticAIContextualModel):
    """Result from individual search modality"""

    content: str = Field(..., description="Result content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Result confidence")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    source: str = Field(..., description="Result source")
    search_type: str = Field(..., description="Type of search performed")

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI modality result operations"""
        return {
            "modality_result": {
                "search_details": {
                    "search_type": self.search_type,
                    "source": self.source,
                    "content_length": len(self.content),
                },
                "performance_metrics": {
                    "confidence": self.confidence,
                    "execution_time": self.execution_time,
                },
                "metadata": self.metadata,
            }
        }


# =============================================================================
# SEARCH CONFIGURATION MODELS
# =============================================================================


class ConsolidatedSearchConfiguration(PydanticAIContextualModel):
    """Consolidated search configuration for all search modalities"""

    # Vector search configuration
    vector_top_k: int = Field(
        default=10, ge=1, le=50, description="Vector search top-k results"
    )
    vector_similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Vector similarity threshold"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002", description="Embedding model name"
    )

    # Graph search configuration
    graph_max_depth: int = Field(
        default=3, ge=1, le=5, description="Maximum graph traversal depth"
    )
    graph_max_entities: int = Field(
        default=50, ge=1, le=100, description="Maximum entities to explore"
    )
    relationship_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Relationship confidence threshold"
    )

    # GNN search configuration
    gnn_model_endpoint: Optional[str] = Field(
        default=None, description="GNN model endpoint URL"
    )
    gnn_embedding_dimension: int = Field(
        default=256, ge=50, le=1024, description="GNN embedding dimension"
    )
    gnn_max_hops: int = Field(default=2, ge=1, le=3, description="Maximum GNN hops")
    gnn_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="GNN confidence threshold"
    )

    # Search orchestration
    parallel_search_enabled: bool = Field(
        default=True, description="Enable parallel search execution"
    )
    result_synthesis_enabled: bool = Field(
        default=True, description="Enable result synthesis"
    )
    max_total_results: int = Field(
        default=20, ge=1, le=100, description="Maximum total results"
    )

    # Performance settings
    search_timeout_seconds: int = Field(
        default=30, ge=5, le=120, description="Search timeout"
    )
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(
        default=3600, ge=300, le=86400, description="Cache TTL"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "search_config": {
                "vector_config": {
                    "top_k": self.vector_top_k,
                    "similarity_threshold": self.vector_similarity_threshold,
                    "embedding_model": self.embedding_model,
                },
                "graph_config": {
                    "max_depth": self.graph_max_depth,
                    "max_entities": self.graph_max_entities,
                    "relationship_threshold": self.relationship_threshold,
                },
                "gnn_config": {
                    "model_endpoint": self.gnn_model_endpoint,
                    "embedding_dimension": self.gnn_embedding_dimension,
                    "max_hops": self.gnn_max_hops,
                    "confidence_threshold": self.gnn_confidence_threshold,
                },
                "orchestration": {
                    "parallel_enabled": self.parallel_search_enabled,
                    "synthesis_enabled": self.result_synthesis_enabled,
                    "max_results": self.max_total_results,
                    "timeout_seconds": self.search_timeout_seconds,
                },
            }
        }


class DynamicSearchConfig(PydanticAIContextualModel):
    """Dynamic search configuration that adapts based on query and domain"""

    base_configuration: ConsolidatedSearchConfiguration = Field(
        description="Base search configuration"
    )
    domain_adaptations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Domain-specific adaptations"
    )
    query_complexity_adaptations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Query complexity adaptations"
    )
    performance_adaptations: Dict[str, float] = Field(
        default_factory=dict, description="Performance-based adaptations"
    )

    # Configuration generation metadata
    generation_timestamp: str = Field(description="Configuration generation timestamp")
    generation_confidence: float = Field(
        ge=0.0, le=1.0, description="Configuration generation confidence"
    )
    learning_source: str = Field(description="Source of learning data")

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "dynamic_config": {
                "base_config": self.base_configuration.run_context_data,
                "adaptations": {
                    "domain_count": len(self.domain_adaptations),
                    "complexity_count": len(self.query_complexity_adaptations),
                    "performance_count": len(self.performance_adaptations),
                },
                "generation_metadata": {
                    "timestamp": self.generation_timestamp,
                    "confidence": self.generation_confidence,
                    "learning_source": self.learning_source,
                },
            }
        }

    def get_adapted_config(
        self, domain: Optional[str] = None, query_complexity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration adapted for specific domain and query complexity"""
        config = self.base_configuration.model_dump()

        # Apply domain-specific adaptations
        if domain and domain in self.domain_adaptations:
            domain_adaptations = self.domain_adaptations[domain]
            for key, value in domain_adaptations.items():
                if key in config:
                    config[key] = value

        # Apply query complexity adaptations
        if query_complexity and query_complexity in self.query_complexity_adaptations:
            complexity_adaptations = self.query_complexity_adaptations[query_complexity]
            for key, value in complexity_adaptations.items():
                if key in config:
                    config[key] = value

        return config
