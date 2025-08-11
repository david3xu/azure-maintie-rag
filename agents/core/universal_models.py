"""
Universal Data Models - Zero Hardcoded Domain Knowledge
======================================================

These models support truly universal RAG processing without predetermined
domain assumptions, classifications, or hardcoded configurations.

All data structures are designed to adapt to ANY content type through
data-driven discovery rather than predetermined categories.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Re-export the universal models from domain intelligence agent
# This provides a central location for all universal data structures


class UniversalDomainCharacteristics(BaseModel):
    """Data-driven domain characteristics discovered from content analysis"""

    # Content structure metrics (measured, not assumed)
    avg_document_length: int = Field(
        ..., description="Average document length in characters"
    )
    document_count: int = Field(..., description="Total documents analyzed")
    vocabulary_richness: float = Field(
        ..., ge=0.0, le=1.0, description="Unique words / total words ratio"
    )
    sentence_complexity: float = Field(
        ..., ge=0.0, description="Average words per sentence"
    )

    # Discovered content patterns (learned from data)
    most_frequent_terms: List[str] = Field(
        default_factory=list, description="Top terms found in content"
    )
    content_patterns: List[str] = Field(
        default_factory=list, description="Structural patterns discovered"
    )
    language_indicators: Dict[str, float] = Field(
        default_factory=dict, description="Language detection scores"
    )

    # Complexity indicators (measured from actual content)
    lexical_diversity: float = Field(
        ..., ge=0.0, le=1.0, description="Type-token ratio"
    )
    vocabulary_complexity_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Complex vs simple vocabulary ratio (domain-agnostic)",
    )
    structural_consistency: float = Field(
        ..., ge=0.0, le=1.0, description="Document structure consistency"
    )
    
    # Backward compatibility aliases for test compatibility
    @property
    def vocabulary_complexity(self) -> float:
        """Alias for vocabulary_complexity_ratio to maintain API compatibility"""
        return self.vocabulary_complexity_ratio
    
    @property
    def concept_density(self) -> float:
        """Calculated concept density based on vocabulary richness and lexical diversity"""
        return (self.vocabulary_richness + self.lexical_diversity) / 2.0
    
    @property
    def structural_patterns(self) -> List[str]:
        """Alias for content_patterns to maintain API compatibility"""
        return self.content_patterns
    
    @property
    def content_signature(self) -> str:
        """Generate content signature from characteristics for API compatibility"""
        return f"vc{self.vocabulary_complexity:.2f}_cd{self.concept_density:.2f}_sc{self.structural_consistency:.2f}"


class UniversalProcessingConfiguration(BaseModel):
    """Processing configuration generated from content characteristics"""

    # Adaptive chunking (based on discovered content patterns)
    optimal_chunk_size: int = Field(
        ..., ge=100, le=4000, description="Optimal chunk size for this content"
    )
    chunk_overlap_ratio: float = Field(
        ..., ge=0.0, le=0.5, description="Overlap ratio based on content coherence"
    )

    # Adaptive extraction thresholds (learned from content distribution)
    entity_confidence_threshold: float = Field(
        ..., ge=0.5, le=1.0, description="Entity extraction threshold"
    )
    relationship_density: float = Field(
        ..., ge=0.0, le=1.0, description="Expected relationship density"
    )

    # Adaptive search optimization (based on content characteristics)
    vector_search_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Vector search importance weight"
    )
    graph_search_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Graph search importance weight"
    )

    # Quality expectations (based on content analysis)
    expected_extraction_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Expected extraction quality"
    )
    processing_complexity: str = Field(
        ..., description="Processing complexity level (low/medium/high)"
    )


class UniversalDomainAnalysis(BaseModel):
    """Complete universal domain analysis without predetermined categories"""

    # Dynamic domain identification (signature derived from content)
    domain_signature: str = Field(
        ..., description="Unique signature generated from content characteristics"
    )
    content_type_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in content type detection"
    )

    # Discovered characteristics and adaptive configuration
    characteristics: UniversalDomainCharacteristics
    processing_config: UniversalProcessingConfiguration
    
    # Backward compatibility aliases for API compatibility
    @property
    def discovered_characteristics(self) -> UniversalDomainCharacteristics:
        """Alias for characteristics to maintain API compatibility"""
        return self.characteristics
    
    @property
    def content_signature(self) -> str:
        """Alias for domain_signature to maintain API compatibility"""
        return self.domain_signature
    
    @property
    def processing_configuration(self) -> UniversalProcessingConfiguration:
        """Alias for processing_config to maintain API compatibility"""
        return self.processing_config

    # Data-driven insights (discovered from analysis)
    key_insights: List[str] = Field(
        default_factory=list, description="Key insights about the content"
    )
    adaptation_recommendations: List[str] = Field(
        default_factory=list, description="Processing adaptation recommendations"
    )

    # Analysis metadata and quality indicators
    analysis_timestamp: str = Field(..., description="When the analysis was performed")
    processing_time: float = Field(
        ..., ge=0.0, description="Time taken for analysis in seconds"
    )
    data_source_path: str = Field(..., description="Path to the analyzed data")
    analysis_reliability: float = Field(
        ..., ge=0.0, le=1.0, description="Reliability score of this analysis"
    )


class UniversalDomainDeps(BaseModel):
    """Universal dependencies without domain assumptions"""

    data_directory: str = Field(
        default="/workspace/azure-maintie-rag/data/raw",
        description="Path to data directory",
    )
    max_files_to_analyze: int = Field(
        default=50, ge=1, le=1000, description="Maximum files to analyze"
    )
    min_content_length: int = Field(
        default=100, ge=50, le=10000, description="Minimum content length to consider"
    )
    enable_multilingual: bool = Field(
        default=True, description="Enable multilingual content support"
    )


# Orchestration models for multi-agent coordination
class UniversalOrchestrationResult(BaseModel):
    """Results from universal RAG workflow orchestration"""

    success: bool = Field(
        ..., description="Whether the workflow completed successfully"
    )
    domain_analysis: Optional[UniversalDomainAnalysis] = Field(
        None, description="Domain intelligence results"
    )
    extraction_results: Optional[Dict[str, Any]] = Field(
        None, description="Knowledge extraction results"
    )
    search_results: Optional[Dict[str, Any]] = Field(
        None, description="Universal search results"
    )

    # Workflow metadata
    total_processing_time: float = Field(
        default=0.0, ge=0.0, description="Total workflow processing time"
    )
    errors: List[str] = Field(
        default_factory=list, description="Any errors encountered"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Any warnings generated"
    )

    # Quality metrics
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall result confidence"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall quality score"
    )


# Knowledge extraction models
class ExtractedEntity(BaseModel):
    """Universal entity extracted from content."""

    text: str = Field(description="Entity text")
    type: str = Field(description="Entity type")  # Changed from entity_type to match validator
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: Optional[str] = Field(default=None, description="Surrounding context")
    positions: List[int] = Field(default_factory=list, description="Character positions in text")  # Added for validator
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ExtractedRelationship(BaseModel):
    """Universal relationship between entities."""

    source: str = Field(description="Source entity")
    target: str = Field(description="Target entity")
    relation: str = Field(description="Relationship type")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: Optional[str] = Field(default=None, description="Surrounding context")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResult(BaseModel):
    """Universal search result."""

    title: str = Field(description="Result title")
    content: str = Field(description="Result content")
    score: float = Field(ge=0.0, description="Relevance score (Azure Search can exceed 1.0)")
    source: str = Field(description="Result source")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class SearchConfiguration(BaseModel):
    """Universal search configuration."""

    max_results: int = Field(
        default=10, ge=1, le=100, description="Maximum search results"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold"
    )
    use_vector_search: bool = Field(default=True, description="Enable vector search")
    use_graph_search: bool = Field(default=True, description="Enable graph search")
    use_gnn_search: bool = Field(default=True, description="Enable GNN search")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration"
    )


# Request/Response Models for Agent Communication
class ExtractionRequest(BaseModel):
    """Request model for knowledge extraction."""

    content: str = Field(..., description="Content to extract from")
    use_domain_analysis: bool = Field(
        default=True, description="Whether to use domain analysis"
    )
    max_entities: int = Field(
        default=50, ge=1, description="Maximum entities to extract"
    )
    max_relationships: int = Field(
        default=100, ge=1, description="Maximum relationships to extract"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )


class ExtractionResult(BaseModel):
    """Results from universal extraction process."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    extraction_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall extraction confidence"
    )
    processing_signature: str = Field(description="Processing signature for tracking")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata"
    )


class SearchRequest(BaseModel):
    """Request model for universal search."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results")
    use_domain_analysis: bool = Field(
        default=True, description="Whether to use domain analysis"
    )
    search_config: Optional[SearchConfiguration] = Field(
        default=None, description="Search configuration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )


class MultiModalSearchResult(BaseModel):
    """Result from multi-modal search combining vector, graph, and GNN."""

    result_id: str = Field(..., description="Unique result identifier")
    content: str = Field(..., description="Result content")
    title: Optional[str] = Field(default=None, description="Result title")
    score: float = Field(ge=0.0, le=1.0, description="Combined relevance score")
    vector_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Vector similarity score"
    )
    graph_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Graph relevance score"
    )
    gnn_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="GNN prediction score"
    )
    source: str = Field(..., description="Result source")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResponse(BaseModel):
    """Response model for universal search."""

    unified_results: List[SearchResult] = Field(default_factory=list)
    total_results_found: int = Field(ge=0, description="Total results found")
    search_confidence: float = Field(ge=0.0, le=1.0, description="Search confidence")
    search_strategy_used: str = Field(description="Search strategy employed")
    processing_time_ms: float = Field(
        ge=0.0, description="Processing time in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )


# Agent communication models
class AgentHandoffData(BaseModel):
    """Data structure for passing information between agents"""

    source_agent: str = Field(..., description="Agent that generated this data")
    target_agent: str = Field(..., description="Intended recipient agent")
    data_type: str = Field(..., description="Type of data being passed")
    payload: Dict[str, Any] = Field(..., description="The actual data payload")

    # Handoff metadata
    timestamp: str = Field(..., description="When this handoff was created")
    priority: str = Field(
        default="normal", description="Processing priority (low/normal/high)"
    )
    requires_validation: bool = Field(
        default=False, description="Whether this data needs validation"
    )


__all__ = [
    "UniversalDomainCharacteristics",
    "UniversalProcessingConfiguration",
    "UniversalDomainAnalysis",
    "UniversalDomainDeps",
    "UniversalOrchestrationResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "SearchResult",
    "SearchConfiguration",
    "AgentHandoffData",
]
# Aliases for backward compatibility
UniversalEntity = ExtractedEntity
UniversalRelation = ExtractedRelationship
