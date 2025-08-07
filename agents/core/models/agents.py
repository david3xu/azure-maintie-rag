"""
Agent Contract and Dependency Models
===================================

Agent-specific contracts, dependencies, and statistical models that define
the interfaces and requirements for the three core agents in the system:
Domain Intelligence, Knowledge Extraction, and Universal Search.

This module provides:
- Agent contract specifications
- Agent dependency models for PydanticAI RunContext
- Statistical pattern and domain analysis models
- Configuration models for agent operations
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from agents.core.constants import (
    DomainIntelligenceConstants,
    ExtractionQualityConstants,
    InfrastructureConstants,
    MathematicalFoundationConstants,
    PerformanceAdaptiveConstants,
    StatisticalConstants,
    SystemPerformanceConstants,
    UniversalSearchConstants,
)

from .base import ConfidenceMethod, PydanticAIContextualModel, SearchType

# =============================================================================
# STATISTICAL AND DOMAIN MODELS
# =============================================================================


class StatisticalPattern(BaseModel):
    """Statistical pattern recognition results"""

    pattern_type: str = Field(description="Type of statistical pattern")
    pattern_value: str = Field(description="Pattern value or expression")
    frequency: int = Field(ge=0, description="Pattern occurrence frequency")
    confidence: float = Field(ge=0.0, le=1.0, description="Pattern confidence score")
    context: List[str] = Field(description="Pattern context examples")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional pattern metadata"
    )


class DomainStatistics(BaseModel):
    """Domain-specific statistical analysis"""

    domain_name: str = Field(description="Domain identifier")
    document_count: int = Field(ge=0, description="Total documents analyzed")
    token_count: int = Field(ge=0, description="Total tokens processed")
    vocabulary_size: int = Field(ge=0, description="Unique vocabulary size")
    average_document_length: float = Field(
        ge=0.0, description="Average document length"
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0, description="Domain complexity score"
    )
    technical_density: float = Field(
        ge=0.0, le=1.0, description="Technical content density"
    )
    patterns_identified: int = Field(ge=0, description="Number of patterns identified")
    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")


# =============================================================================
# AGENT CONTRACT SPECIFICATIONS
# =============================================================================


class DomainAnalysisContract(BaseModel):
    """Domain Intelligence Agent contract specification"""

    agent_id: str = Field(description="Agent identifier")
    capabilities: List[str] = Field(description="Agent capabilities")
    input_requirements: Dict[str, Any] = Field(description="Input data requirements")
    output_format: Dict[str, Any] = Field(description="Output format specification")
    performance_guarantees: Dict[str, Any] = Field(
        description="Performance SLA guarantees"
    )
    dependencies: List[str] = Field(description="Required dependencies")
    configuration_requirements: Dict[str, Any] = Field(
        description="Configuration requirements"
    )
    supported_domains: List[str] = Field(description="Supported domain types")
    statistical_patterns: List[StatisticalPattern] = Field(
        description="Identified statistical patterns"
    )
    domain_statistics: DomainStatistics = Field(
        description="Domain analysis statistics"
    )


class KnowledgeExtractionContract(BaseModel):
    """Knowledge Extraction Agent contract specification"""

    agent_id: str = Field(description="Agent identifier")
    extraction_capabilities: List[str] = Field(description="Extraction capabilities")
    supported_entity_types: List[str] = Field(description="Supported entity types")
    supported_relationship_types: List[str] = Field(
        description="Supported relationship types"
    )
    quality_thresholds: Dict[str, float] = Field(
        description="Quality threshold requirements"
    )
    performance_metrics: Dict[str, float] = Field(
        description="Expected performance metrics"
    )
    validation_requirements: Dict[str, Any] = Field(
        description="Validation requirements"
    )
    output_schema: Dict[str, Any] = Field(description="Output schema specification")
    dependencies: List[str] = Field(description="Required dependencies")


class UniversalSearchContract(BaseModel):
    """Universal Search Agent contract specification"""

    agent_id: str = Field(description="Agent identifier")
    search_modalities: List[SearchType] = Field(description="Supported search types")
    performance_guarantees: Dict[str, float] = Field(
        description="Performance SLA guarantees"
    )
    quality_metrics: Dict[str, float] = Field(description="Quality metric requirements")
    supported_query_types: List[str] = Field(description="Supported query types")
    result_synthesis_methods: List[str] = Field(description="Result synthesis methods")
    caching_strategy: Dict[str, Any] = Field(
        description="Caching strategy specification"
    )
    dependencies: List[str] = Field(description="Required dependencies")
    configuration_schema: Dict[str, Any] = Field(description="Configuration schema")


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class CompleteDomainConfig(BaseModel):
    """Complete domain configuration combining infrastructure, ML, and patterns"""

    domain: str = Field(description="Domain identifier")
    infrastructure: "InfrastructureConfig" = Field(
        description="Infrastructure configuration"
    )
    ml_model: "MLModelConfig" = Field(description="ML model configuration")
    patterns: Any = Field(description="Extracted patterns used for configuration")
    generation_confidence: float = Field(
        ge=0.0, le=1.0, description="Configuration generation confidence"
    )


class SynthesisWeights(BaseModel):
    """Result synthesis weight configuration"""

    confidence: float = Field(ge=0.0, le=1.0, description="Confidence weight")
    agreement: float = Field(ge=0.0, le=1.0, description="Cross-modal agreement weight")
    quality: float = Field(ge=0.0, le=1.0, description="Quality score weight")


class DomainConfig(BaseModel):
    """Domain-specific configuration"""

    domain_name: str = Field(description="Domain identifier")
    similarity_threshold: float = Field(
        ge=0.0, le=1.0, description="Similarity threshold"
    )
    processing_patterns: List[str] = Field(description="Processing pattern list")
    synthesis_weights: SynthesisWeights = Field(description="Result synthesis weights")
    routing_rules: List[Dict[str, Any]] = Field(description="Query routing rules")
    cache_ttl_seconds: int = Field(
        default=PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL, ge=0
    )
    performance_targets: Dict[str, float] = Field(
        description="Performance target metrics"
    )


class ExtractionConfiguration(BaseModel):
    """Complete extraction configuration with learned parameters"""

    domain_name: str = Field(description="Domain name (learned from subdirectory)")
    entity_confidence_threshold: float = Field(
        ge=0.0, le=1.0, description="Learned entity confidence threshold"
    )
    relationship_confidence_threshold: float = Field(
        ge=0.0, le=1.0, description="Learned relationship confidence threshold"
    )
    chunk_size: int = Field(ge=50, le=10000, description="Learned optimal chunk size")
    chunk_overlap: int = Field(ge=0, le=1000, description="Learned chunk overlap")
    expected_entity_types: List[str] = Field(
        description="Learned entity types from corpus"
    )
    target_response_time_seconds: float = Field(
        ge=0.1, description="Learned response SLA from complexity"
    )
    technical_vocabulary: List[str] = Field(description="Learned technical vocabulary")
    key_concepts: List[str] = Field(description="Learned key concepts")
    cache_ttl_seconds: int = Field(
        default=PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL, ge=0
    )
    parallel_processing_threshold: int = Field(
        ge=1, description="Parallel processing threshold"
    )
    max_concurrent_chunks: int = Field(
        default=PerformanceAdaptiveConstants.MAX_CONCURRENT_CHUNKS, ge=1
    )
    generation_confidence: float = Field(
        ge=0.0, le=1.0, description="Configuration generation confidence"
    )
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    generation_timestamp: str = Field(description="When configuration was generated")


class VectorSearchConfig(BaseModel):
    """Vector search configuration"""

    top_k: int = Field(
        default=UniversalSearchConstants.DEFAULT_VECTOR_TOP_K, ge=1, le=50
    )
    similarity_threshold: float = Field(
        default=UniversalSearchConstants.VECTOR_SIMILARITY_THRESHOLD, ge=0.0, le=1.0
    )
    include_metadata: bool = Field(default=True, description="Include search metadata")
    embedding_model: str = Field(
        default=InfrastructureConstants.DEFAULT_EMBEDDING_MODEL,
        description="Embedding model name",
    )


class GraphSearchConfig(BaseModel):
    """Graph search configuration"""

    max_depth: int = Field(
        default=UniversalSearchConstants.DEFAULT_MAX_DEPTH, ge=1, le=5
    )
    max_entities: int = Field(
        default=UniversalSearchConstants.DEFAULT_MAX_ENTITIES, ge=1, le=100
    )
    relationship_threshold: float = Field(
        default=UniversalSearchConstants.DEFAULT_RELATIONSHIP_THRESHOLD, ge=0.0, le=1.0
    )
    include_relationship_weights: bool = Field(
        default=True, description="Include relationship weights in results"
    )


class GNNSearchConfig(BaseModel):
    """Graph Neural Network search configuration"""

    model_endpoint: str = Field(description="GNN model endpoint URL")
    embedding_dimension: int = Field(
        default=UniversalSearchConstants.DEFAULT_GNN_NODE_EMBEDDINGS, ge=50, le=1024
    )
    max_hops: int = Field(
        default=UniversalSearchConstants.DEFAULT_GNN_MAX_HOPS,
        ge=1,
        le=3,
        description="Maximum GNN hops",
    )
    confidence_threshold: float = Field(
        default=UniversalSearchConstants.GNN_MIN_PREDICTION_CONFIDENCE, ge=0.0, le=1.0
    )
    batch_size: int = Field(
        default=SystemPerformanceConstants.ML_BATCH_SIZE,
        ge=1,
        le=128,
        description="Batch size for inference",
    )


# =============================================================================
# AGENT DEPENDENCY MODELS
# =============================================================================


class AzureServicesDeps(BaseModel):
    """Direct Azure Services dependency for RunContext[AzureServicesDeps]"""

    # This should be the actual ConsolidatedAzureServices instance
    service_container: Any = Field(description="ConsolidatedAzureServices instance")

    class Config:
        arbitrary_types_allowed = True

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status directly from the container"""
        return self.service_container.get_service_status()


class DomainIntelligenceDeps(BaseModel):
    """Domain Intelligence Agent dependencies"""

    azure_services: Optional[Any] = Field(
        default=None, description="Azure services container"
    )
    cache_manager: Optional[Any] = Field(
        default=None, description="Cache manager instance"
    )
    hybrid_analyzer: Optional[Any] = Field(
        default=None, description="Hybrid domain analyzer"
    )
    pattern_engine: Optional[Any] = Field(
        default=None, description="Pattern extraction engine"
    )
    config_generator: Optional[Any] = Field(
        default=None, description="Configuration generator"
    )

    class Config:
        arbitrary_types_allowed = True


class KnowledgeExtractionDeps(BaseModel):
    """Knowledge Extraction Agent dependencies"""

    azure_services: Optional[Any] = Field(
        default=None, description="Azure services container"
    )
    cache_manager: Optional[Any] = Field(
        default=None, description="Cache manager instance"
    )
    extraction_processor: Optional[Any] = Field(
        default=None, description="Unified extraction processor"
    )
    validation_processor: Optional[Any] = Field(
        default=None, description="Validation processor"
    )

    class Config:
        arbitrary_types_allowed = True


class UniversalSearchDeps(BaseModel):
    """Universal Search Agent dependencies"""

    azure_services: Optional[Any] = Field(
        default=None, description="Azure services container"
    )
    cache_manager: Optional[Any] = Field(
        default=None, description="Cache manager instance"
    )
    search_orchestrator: Optional[Any] = Field(
        default=None, description="Search orchestrator"
    )
    domain_detector: Optional[Any] = Field(
        default=None, description="Domain detection service"
    )

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# SERVICE CONTAINER CONFIGURATION
# =============================================================================


class ServiceContainerConfig(BaseModel):
    """Configuration for agent service container initialization"""

    # Azure service endpoints
    openai_endpoint: str = Field(description="Azure OpenAI endpoint")
    search_endpoint: str = Field(description="Azure Search endpoint")
    cosmos_endpoint: str = Field(description="Azure Cosmos DB endpoint")
    ml_endpoint: Optional[str] = Field(
        default=None, description="Azure ML endpoint (optional)"
    )

    # Service-specific settings
    openai_deployment: str = Field(description="OpenAI model deployment name")
    search_index: str = Field(description="Search index name")
    cosmos_database: str = Field(description="Cosmos database name")
    cosmos_container: str = Field(description="Cosmos container name")

    # Authentication settings
    use_managed_identity: bool = Field(
        default=True, description="Use managed identity for authentication"
    )
    subscription_id: Optional[str] = Field(
        default=None, description="Azure subscription ID"
    )
    resource_group: Optional[str] = Field(
        default=None, description="Azure resource group"
    )

    # Performance settings
    cache_enabled: bool = Field(default=True, description="Enable caching")
    max_concurrent_requests: int = Field(
        default=SystemPerformanceConstants.DEFAULT_CONCURRENT_REQUESTS,
        ge=1,
        le=50,
        description="Maximum concurrent requests",
    )
    request_timeout_seconds: int = Field(
        default=SystemPerformanceConstants.DEFAULT_TIMEOUT_SECONDS,
        ge=5,
        le=300,
        description="Request timeout",
    )

    # Health monitoring
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval_seconds: int = Field(
        default=SystemPerformanceConstants.HEALTH_CHECK_INTERVAL_SECONDS,
        ge=10,
        le=300,
        description="Health check interval",
    )


# =============================================================================
# DOMAIN INTELLIGENCE MODELS
# =============================================================================


class DomainIntelligenceConfig(PydanticAIContextualModel):
    """Configuration for domain intelligence analysis"""

    # Analysis depth
    enable_advanced_analytics: bool = Field(
        default=True, description="Enable TF-IDF and clustering"
    )
    enable_pattern_detection: bool = Field(
        default=True, description="Enable domain pattern detection"
    )
    enable_quality_assessment: bool = Field(
        default=True, description="Enable quality assessment"
    )

    # Pattern detection parameters
    min_pattern_frequency: int = Field(
        default=DomainIntelligenceConstants.MIN_PATTERN_FREQUENCY,
        ge=1,
        description="Minimum frequency for pattern detection",
    )
    max_patterns_per_type: int = Field(
        default=DomainIntelligenceConstants.MAX_PATTERNS_PER_TYPE,
        ge=1,
        description="Maximum patterns per type",
    )

    # TF-IDF parameters
    tfidf_max_features: int = Field(
        default=DomainIntelligenceConstants.TFIDF_MAX_FEATURES,
        ge=100,
        description="Maximum TF-IDF features",
    )
    tfidf_min_df: int = Field(
        default=DomainIntelligenceConstants.TFIDF_MIN_DF,
        ge=1,
        description="Minimum document frequency",
    )
    tfidf_max_df: float = Field(
        default=DomainIntelligenceConstants.TFIDF_MAX_DF,
        gt=0.0,
        lt=1.0,
        description="Maximum document frequency ratio",
    )

    # Clustering parameters
    n_clusters: int = Field(
        default=DomainIntelligenceConstants.N_SEMANTIC_CLUSTERS,
        ge=2,
        le=20,
        description="Number of semantic clusters",
    )
    cluster_random_state: int = Field(
        default=MathematicalFoundationConstants.RANDOM_SEED,
        description="Random state for clustering",
    )

    # Confidence calculation
    confidence_method: ConfidenceMethod = Field(
        default=ConfidenceMethod.ADAPTIVE, description="Confidence calculation method"
    )
    min_confidence_threshold: float = Field(
        default=StatisticalConstants.MIN_DOMAIN_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI domain intelligence configuration"""
        return {
            "domain_intelligence_config": {
                "analysis_settings": {
                    "enable_advanced_analytics": self.enable_advanced_analytics,
                    "enable_pattern_detection": self.enable_pattern_detection,
                    "enable_quality_assessment": self.enable_quality_assessment,
                },
                "pattern_detection": {
                    "min_pattern_frequency": self.min_pattern_frequency,
                    "max_patterns_per_type": self.max_patterns_per_type,
                },
                "tfidf_settings": {
                    "max_features": self.tfidf_max_features,
                    "min_df": self.tfidf_min_df,
                    "max_df": self.tfidf_max_df,
                },
                "clustering_settings": {
                    "n_clusters": self.n_clusters,
                    "random_state": self.cluster_random_state,
                },
                "confidence_settings": {
                    "method": self.confidence_method.value,
                    "min_threshold": self.min_confidence_threshold,
                },
            }
        }


class DomainAnalysisOutput(BaseModel):
    """Structured output from Domain Intelligence Agent for PydanticAI"""

    domain_classification: str = Field(description="Detected domain type")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Classification confidence score"
    )
    generated_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Generated configuration parameters"
    )
    patterns_identified: Optional[Dict[str, List[str]]] = Field(
        default_factory=dict, description="Identified domain patterns"
    )
    technical_vocabulary: Optional[List[str]] = Field(
        default_factory=list, description="Technical terms found"
    )
    processing_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Processing metadata"
    )


class DomainAnalysisResult(PydanticAIContextualModel):
    """
    PydanticAI-enhanced domain analysis result following output validator patterns

    Replaces the previous UnifiedAnalysis dataclass with proper Pydantic validation
    and cross-agent compatibility.
    """

    # Core domain intelligence metrics
    # Forward references to avoid circular imports
    text_statistics: Any = Field(..., description="Statistical text analysis")
    cleaning_result: Any = Field(..., description="Text preprocessing results")

    # Domain-specific intelligence
    domain_patterns: Dict[str, List[str]] = Field(
        default_factory=dict, description="Detected domain patterns"
    )
    technical_vocabulary: List[str] = Field(
        default_factory=list, description="Identified technical terms"
    )
    concept_hierarchy: Dict[str, float] = Field(
        default_factory=dict, description="Concept importance hierarchy"
    )

    # Quality and confidence metrics
    analysis_confidence: Any = Field(..., description="Overall analysis confidence")
    domain_fit_score: float = Field(
        ge=0.0, le=1.0, description="Fit to detected domain"
    )
    processing_quality: str = Field(..., description="Processing quality tier")

    # Advanced analytics
    tfidf_features: Dict[str, float] = Field(
        default_factory=dict, description="TF-IDF feature importance"
    )
    semantic_clusters: Dict[str, Any] = Field(
        default_factory=dict, description="Semantic clustering results"
    )

    # Metadata
    source_file: Optional[str] = Field(default=None, description="Source file path")
    processing_time_ms: float = Field(
        ge=0.0, description="Processing time in milliseconds"
    )
    analysis_timestamp: str = Field(..., description="Analysis completion timestamp")

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI domain analysis operations"""
        return {
            "domain_analysis": {
                "domain_patterns": {
                    "pattern_types": list(self.domain_patterns.keys()),
                    "total_patterns": sum(
                        len(patterns) for patterns in self.domain_patterns.values()
                    ),
                },
                "vocabulary": {
                    "technical_terms": len(self.technical_vocabulary),
                    "concept_hierarchy_size": len(self.concept_hierarchy),
                },
                "quality_metrics": {
                    "domain_fit_score": self.domain_fit_score,
                    "processing_quality": self.processing_quality,
                },
                "analytics": {
                    "tfidf_features": len(self.tfidf_features),
                    "semantic_clusters": len(self.semantic_clusters),
                },
                "metadata": {
                    "source_file": self.source_file,
                    "processing_time_ms": self.processing_time_ms,
                    "analysis_timestamp": self.analysis_timestamp,
                },
            }
        }
