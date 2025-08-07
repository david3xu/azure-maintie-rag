"""
Centralized Data Models for Azure Universal RAG Agents
=====================================================

This module centralizes all data types, models, and schemas used across agents
to support the zero-hardcoded-values philosophy and improve maintainability.

Complete centralization of all Pydantic models, dataclasses, and TypedDicts
eliminates duplicate model definitions and enforces consistent type usage.

Categories:
- Core Base Models & Enums
- Agent Contract Models
- Configuration Models
- Request/Response Models
- Workflow State Models
- Analysis & Processing Models
- Validation & Error Models
- Dependency Models
- Azure Service Models
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Import centralized configuration for dynamic value resolution
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, computed_field, validator
from pydantic_ai import RunContext

# Note: PydanticAI output_validator is used as a method decorator on Agent instances, not imported directly


if TYPE_CHECKING:
    from config.centralized_config import (
        ExtractionConfiguration,
        SearchConfiguration,
        SystemConfiguration,
    )

# Import centralized constants
from agents.core.constants import (
    DomainIntelligenceConstants,
    ErrorHandlingCoordinatedConstants,
    FileSystemConstants,
    InfrastructureConstants,
    KnowledgeExtractionConstants,
    PerformanceAdaptiveConstants,
    StatisticalConstants,
    UniversalSearchConstants,
    WorkflowConstants,
)

# =============================================================================
# FOUNDATION BASE CLASS FOR PYDANTIC AI INTEGRATION
# =============================================================================


class PydanticAIContextualModel(BaseModel):
    """Base model with PydanticAI RunContext integration for agent communication"""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @computed_field
    @property
    def run_context_data(self) -> Optional[Dict[str, Any]]:
        """Extract relevant data for PydanticAI RunContext"""
        # Override in subclasses to provide context-specific data
        return None

    def to_run_context(self) -> Dict[str, Any]:
        """Convert model to RunContext-compatible dictionary"""
        context_data = self.run_context_data or {}
        return {
            "model_type": self.__class__.__name__,
            "model_data": self.model_dump(exclude={"run_context_data"}),
            **context_data,
        }

    def model_dump_json(self, **kwargs) -> str:
        """Override to exclude computed fields from JSON serialization"""
        kwargs.setdefault("exclude", set()).add("run_context_data")
        return super().model_dump_json(**kwargs)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to exclude computed fields from dict serialization"""
        exclude = kwargs.get("exclude", set())
        if isinstance(exclude, set):
            exclude.add("run_context_data")
        elif isinstance(exclude, dict):
            exclude["run_context_data"] = True
        else:
            kwargs["exclude"] = {"run_context_data"}
        return super().model_dump(**kwargs)


# =============================================================================
# TIER 1: FOUNDATION MODELS & UNIFIED CONFIGURATION ARCHITECTURE
# =============================================================================

# UnifiedConfigurationResolver deleted - over-abstracted config system with zero actual usage
# This complex configuration resolution system was never actually called in production
# Agents use dynamic_config_manager directly which is simpler and actually used


class UnifiedAgentConfiguration(PydanticAIContextualModel):
    """
    Unified agent configuration model that consolidates all configuration patterns.
    Replaces fragmented configuration resolution with a single, predictable interface.
    """

    agent_type: str = Field(description="Type of agent this configuration is for")
    domain_name: str = Field(
        description="Domain name for domain-specific configuration"
    )
    configuration_data: Dict[str, Any] = Field(
        description="Resolved configuration parameters"
    )
    resolved_at: datetime = Field(description="When this configuration was resolved")
    resolver_context: Dict[str, Any] = Field(
        default_factory=dict, description="Context used for resolution"
    )

    # Performance and quality tracking
    resolution_time_ms: Optional[float] = Field(
        default=None, description="Time taken to resolve configuration"
    )
    configuration_source: str = Field(
        default="unified_resolver", description="Source of configuration"
    )
    cache_hit: bool = Field(
        default=False, description="Whether this was served from cache"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI agent operations"""
        return {
            "agent_type": self.agent_type,
            "domain_name": self.domain_name,
            "configuration": self.configuration_data,
            "resolved_at": self.resolved_at.isoformat(),
            "resolver_context": self.resolver_context,
        }

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration parameter with default fallback"""
        return self.configuration_data.get(key, default)

    def update_parameter(self, key: str, value: Any, reason: str = None):
        """Update a configuration parameter with tracking"""
        old_value = self.configuration_data.get(key)
        self.configuration_data[key] = value

        # Log parameter change for audit trail
        if reason:
            change_log = self.resolver_context.setdefault("parameter_changes", [])
            change_log.append(
                {
                    "parameter": key,
                    "old_value": old_value,
                    "new_value": value,
                    "reason": reason,
                    "updated_at": datetime.now().isoformat(),
                }
            )

    def validate_required_parameters(self, required_params: List[str]) -> List[str]:
        """Validate that required parameters are present and return missing ones"""
        missing = []
        for param in required_params:
            if param not in self.configuration_data:
                missing.append(param)
        return missing


# UnifiedConfigurationResolver global instance deleted - system was unused


# Legacy compatibility functions
class ConfigurationResolver:
    """Legacy configuration resolver for backward compatibility"""

    @staticmethod
    async def resolve_extraction_config(domain_name: str = None) -> Dict[str, Any]:
        """Legacy compatibility - use unified resolver"""
        resolver = get_unified_configuration_resolver()
        config = await resolver.resolve_agent_configuration(
            "knowledge_extraction", domain_name
        )
        return config.configuration_data

    @staticmethod
    async def resolve_search_config(domain_name: str = None) -> Dict[str, Any]:
        """Legacy compatibility - use unified resolver"""
        resolver = get_unified_configuration_resolver()
        config = await resolver.resolve_agent_configuration(
            "universal_search", domain_name
        )
        return config.configuration_data

    @staticmethod
    async def resolve_azure_config() -> Dict[str, Any]:
        """Legacy compatibility - Azure configuration resolution"""
        try:
            from config.centralized_config import get_azure_configuration

            return get_azure_configuration()
        except ImportError:
            return {
                "openai_endpoint": "",
                "search_endpoint": "",
                "cosmos_endpoint": "",
                "openai_timeout": PerformanceAdaptiveConstants.DEFAULT_TIMEOUT,
                "search_timeout": PerformanceAdaptiveConstants.DEFAULT_TIMEOUT,
                "max_retries": PerformanceAdaptiveConstants.MAX_RETRIES,
            }


# This class has been moved to the top of the file to resolve import order dependencies


# =============================================================================
# TIER 1: CORE BASE MODELS & ENUMS (Unified from all model files)
# =============================================================================


class HealthStatus(str, Enum):
    """Standard health status classifications across all agents"""

    HEALTHY = "healthy"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    ERROR = "error"
    NOT_INITIALIZED = "not_initialized"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status for agent operations"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowState(str, Enum):
    """Workflow execution states"""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeState(str, Enum):
    """Individual node execution states"""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorSeverity(str, Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error category classifications"""

    AZURE_SERVICE = "azure_service"
    CONFIGURATION = "configuration"
    PROCESSING = "processing"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    CACHE_MEMORY = "cache_memory"
    DOMAIN_INTELLIGENCE = "domain_intelligence"
    AGENT_PROCESSING = "agent_processing"
    UNKNOWN = "unknown"


class SearchType(str, Enum):
    """Supported search types for tri-modal search"""

    VECTOR = "vector"
    GRAPH = "graph"
    GNN = "gnn"
    ALL = "all"


class MessageType(str, Enum):
    """Graph communication message types"""

    QUERY = "query"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"


class ConfidenceMethod(str, Enum):
    """Confidence calculation methods for domain intelligence and knowledge extraction"""

    ADAPTIVE = "adaptive"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


class StateTransferType(str, Enum):
    """Types of state transfers between workflow graphs"""

    CONFIG_GENERATION = "config_generation"
    DOMAIN_ANALYSIS = "domain_analysis"
    PATTERN_LEARNING = "pattern_learning"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_CONTEXT = "error_context"


# ModelCapability and QueryComplexity enums removed - using static model selection


# =============================================================================
# BASE REQUEST/RESPONSE MODELS
# =============================================================================


class BaseRequest(BaseModel):
    """Base request model with common fields"""

    request_id: Optional[str] = Field(
        default=None, description="Unique request identifier"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now, description="Request timestamp"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )

    class Config:
        use_enum_values = True
        extra = "forbid"
        str_strip_whitespace = True


class BaseResponse(BaseModel):
    """Base response model with common fields"""

    request_id: Optional[str] = Field(
        default=None, description="Matching request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
    processing_time_seconds: Optional[float] = Field(
        default=None, description="Processing time"
    )
    success: bool = Field(default=True, description="Operation success status")

    class Config:
        use_enum_values = True


class BaseAnalysisResult(BaseModel):
    """Base analysis result model"""

    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence score")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# =============================================================================
# ENHANCED AZURE SERVICE INTEGRATION MODELS
# =============================================================================


class AzureServiceConfiguration(PydanticAIContextualModel):
    """
    Unified Azure service configuration for all service types.
    Supports DefaultAzureCredential patterns and environment-based configuration.
    """

    # Core service identification
    service_type: str = Field(
        description="Type of Azure service (openai, search, cosmos, storage, ml)"
    )
    endpoint_url: str = Field(description="Azure service endpoint URL")
    resource_group: Optional[str] = Field(
        default=None, description="Azure resource group"
    )
    subscription_id: Optional[str] = Field(
        default=None, description="Azure subscription ID"
    )

    # Authentication configuration
    use_managed_identity: bool = Field(
        default=True, description="Use managed identity for authentication"
    )
    credential_scope: Optional[str] = Field(
        default=None, description="Azure credential scope"
    )

    # Service-specific parameters
    api_version: Optional[str] = Field(default=None, description="Service API version")
    deployment_name: Optional[str] = Field(
        default=None, description="Model deployment name (for OpenAI)"
    )
    index_name: Optional[str] = Field(
        default=None, description="Search index name (for Cognitive Search)"
    )
    database_name: Optional[str] = Field(
        default=None, description="Database name (for Cosmos DB)"
    )
    container_name: Optional[str] = Field(default=None, description="Container name")

    # Performance and reliability
    timeout_seconds: int = Field(
        default=60, ge=1, le=300, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    retry_backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )

    # Health check configuration
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval_seconds: int = Field(
        default=30, ge=5, le=300, description="Health check interval"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for Azure service operations"""
        return {
            "service_type": self.service_type,
            "endpoint_url": self.endpoint_url,
            "authentication": {
                "use_managed_identity": self.use_managed_identity,
                "credential_scope": self.credential_scope,
            },
            "service_params": {
                "api_version": self.api_version,
                "deployment_name": self.deployment_name,
                "index_name": self.index_name,
                "database_name": self.database_name,
                "container_name": self.container_name,
            },
            "reliability": {
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
                "retry_backoff_factor": self.retry_backoff_factor,
            },
        }

    @classmethod
    def create_openai_config(
        cls,
        endpoint_url: str,
        deployment_name: str,
        api_version: str = "2024-08-01-preview",
    ) -> "AzureServiceConfiguration":
        """Create Azure OpenAI service configuration"""
        return cls(
            service_type="openai",
            endpoint_url=endpoint_url,
            deployment_name=deployment_name,
            api_version=api_version,
            credential_scope="https://cognitiveservices.azure.com/.default",
        )

    @classmethod
    def create_search_config(
        cls, endpoint_url: str, index_name: str, api_version: str = "2023-07-01-Preview"
    ) -> "AzureServiceConfiguration":
        """Create Azure Cognitive Search service configuration"""
        return cls(
            service_type="search",
            endpoint_url=endpoint_url,
            index_name=index_name,
            api_version=api_version,
            credential_scope="https://search.azure.com/.default",
        )

    @classmethod
    def create_cosmos_config(
        cls, endpoint_url: str, database_name: str, container_name: str
    ) -> "AzureServiceConfiguration":
        """Create Azure Cosmos DB service configuration"""
        return cls(
            service_type="cosmos",
            endpoint_url=endpoint_url,
            database_name=database_name,
            container_name=container_name,
            credential_scope="https://cosmos.azure.com/.default",
        )


class AzureServiceMetrics(BaseModel):
    """Enhanced Azure service performance and health metrics"""

    service_name: str = Field(description="Azure service name")
    service_type: str = Field(
        description="Service type (openai, search, cosmos, storage, ml)"
    )
    health_status: HealthStatus = Field(description="Current health status")

    # Performance metrics
    response_time_ms: float = Field(ge=0.0, description="Average response time")
    p95_response_time_ms: Optional[float] = Field(
        default=None, ge=0.0, description="95th percentile response time"
    )
    p99_response_time_ms: Optional[float] = Field(
        default=None, ge=0.0, description="99th percentile response time"
    )
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate percentage")
    throughput_requests_per_second: float = Field(
        ge=0.0, description="Requests per second"
    )
    availability_percentage: float = Field(
        ge=0.0, le=100.0, description="Service availability"
    )

    # Cost and resource metrics
    cost_per_request: Optional[float] = Field(
        default=None, ge=0.0, description="Cost per request"
    )
    token_usage: Optional[int] = Field(
        default=None, ge=0, description="Token usage (for LLM services)"
    )
    quota_usage_percentage: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Quota usage"
    )

    # Authentication and connectivity
    auth_success_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Authentication success rate"
    )
    connection_pool_size: Optional[int] = Field(
        default=None, ge=0, description="Connection pool size"
    )
    active_connections: Optional[int] = Field(
        default=None, ge=0, description="Active connections"
    )

    # Temporal tracking
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last metrics update"
    )
    measurement_window_minutes: int = Field(
        default=5, ge=1, le=60, description="Measurement window"
    )

    def calculate_sla_compliance(self, target_availability: float = 99.9) -> bool:
        """Check if service meets SLA availability target"""
        return self.availability_percentage >= target_availability

    def is_performance_degraded(
        self,
        response_time_threshold_ms: float = 3000.0,
        error_rate_threshold: float = 0.05,
    ) -> bool:
        """Check if service performance is degraded"""
        return (
            self.response_time_ms > response_time_threshold_ms
            or self.error_rate > error_rate_threshold
        )


# AzureServiceHealthCheck deleted - over-engineered health check system with zero instantiation
# Elaborate 41-line health monitoring (connectivity, auth, performance, recommendations) never used
# Azure health checks can be implemented through simpler patterns when actually needed


class AzureMLModelMetadata(BaseModel):
    """Azure ML model metadata and performance"""

    model_name: str = Field(description="Model deployment name")
    model_version: str = Field(description="Model version")
    endpoint_url: str = Field(description="Model endpoint URL")
    input_schema: Dict[str, Any] = Field(description="Input schema definition")
    output_schema: Dict[str, Any] = Field(description="Output schema definition")
    performance_metrics: Dict[str, float] = Field(
        description="Model performance metrics"
    )
    deployment_status: str = Field(description="Deployment status")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


class AzureSearchIndexSchema(BaseModel):
    """Azure Cognitive Search index schema"""

    index_name: str = Field(description="Search index name")
    fields: List[Dict[str, Any]] = Field(description="Index field definitions")
    scoring_profiles: List[Dict[str, Any]] = Field(
        default_factory=list, description="Scoring profiles"
    )
    analyzers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Custom analyzers"
    )
    vector_search_config: Dict[str, Any] = Field(
        description="Vector search configuration"
    )
    document_count: int = Field(ge=0, description="Number of documents in index")
    storage_size_bytes: int = Field(ge=0, description="Index storage size")


class AzureCosmosGraphSchema(BaseModel):
    """Azure Cosmos DB graph database schema"""

    database_name: str = Field(description="Cosmos database name")
    container_name: str = Field(description="Graph container name")
    vertex_types: List[str] = Field(description="Defined vertex types")
    edge_types: List[str] = Field(description="Defined edge types")
    partition_key: str = Field(description="Partition key property")
    vertex_count: int = Field(ge=0, description="Total vertex count")
    edge_count: int = Field(ge=0, description="Total edge count")
    throughput_ru: int = Field(ge=400, description="Provisioned throughput RU/s")


# =============================================================================
# AGENT CONTRACT MODELS
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


class SynthesisWeights(BaseModel):
    """Result synthesis weight configuration"""

    confidence: float = Field(ge=0.0, le=1.0, description="Confidence weight")
    agreement: float = Field(ge=0.0, le=1.0, description="Cross-modal agreement weight")
    quality: float = Field(ge=0.0, le=1.0, description="Quality score weight")

    # Replaced custom validator with built-in Field constraints (ge=0.0, le=1.0)


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
    relationship_types: List[str] = Field(
        default_factory=list, description="Supported relationship types"
    )


class GNNSearchConfig(BaseModel):
    """Graph Neural Network search configuration"""

    max_predictions: int = Field(
        default=UniversalSearchConstants.DEFAULT_MAX_PREDICTIONS, ge=1, le=50
    )
    pattern_threshold: float = Field(
        default=UniversalSearchConstants.DEFAULT_PATTERN_THRESHOLD, ge=0.0, le=1.0
    )
    node_embeddings: int = Field(
        default=UniversalSearchConstants.DEFAULT_GNN_NODE_EMBEDDINGS, ge=32, le=512
    )
    min_training_examples: int = Field(
        default=UniversalSearchConstants.DEFAULT_MIN_TRAINING_EXAMPLES, ge=10
    )


# =============================================================================
# REQUEST/RESPONSE MODELS
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

    # Replaced custom validator with built-in Field constraints (min_length=1)


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
# RESPONSE MODELS
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
# PERFORMANCE FEEDBACK INTEGRATION MODELS
# =============================================================================


class PerformanceFeedbackPoint(BaseModel):
    """Individual performance feedback data point for learning and optimization"""

    # Context identification
    agent_type: str = Field(description="Agent that generated this performance data")
    domain_name: str = Field(description="Domain context for the operation")
    operation_type: str = Field(
        description="Type of operation (extraction, search, analysis)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this performance was recorded"
    )

    # Configuration used
    configuration_used: Dict[str, Any] = Field(
        description="Configuration parameters that were used"
    )
    configuration_source: str = Field(
        description="Source of the configuration (dynamic, static, fallback)"
    )

    # Performance metrics
    execution_time_seconds: float = Field(ge=0.0, description="Total execution time")
    success: bool = Field(description="Whether the operation succeeded")
    quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Quality assessment score"
    )

    # Input characteristics
    input_size: Optional[int] = Field(
        default=None, ge=0, description="Size of input data"
    )
    input_complexity: Optional[str] = Field(
        default=None, description="Assessed complexity of input"
    )

    # Output characteristics
    output_size: Optional[int] = Field(
        default=None, ge=0, description="Size of output data"
    )
    output_quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Detailed quality metrics"
    )

    # Resource utilization
    memory_usage_mb: Optional[float] = Field(
        default=None, ge=0.0, description="Memory usage during operation"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="CPU usage during operation"
    )

    # Error information (if applicable)
    error_message: Optional[str] = Field(
        default=None, description="Error message if operation failed"
    )
    error_category: Optional[ErrorCategory] = Field(
        default=None, description="Error category classification"
    )

    # Context for learning
    user_satisfaction: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="User satisfaction score"
    )
    downstream_impact: Optional[Dict[str, float]] = Field(
        default=None, description="Impact on downstream operations"
    )


# PerformanceFeedbackAggregate deleted in Phase 2 - was unused performance optimization model


# ConfigurationOptimizationRequest deleted in Phase 2 - was unused optimization request model


# OptimizedConfiguration deleted in Phase 2 - was unused optimization result model


# PerformanceFeedbackCollector deleted - over-engineered feature with zero actual usage
# This complex performance monitoring system was never actually called in the codebase
# Performance monitoring can be handled through simpler logging mechanisms


# =============================================================================
# WORKFLOW STATE MODELS
# =============================================================================


@dataclass
class WorkflowExecutionState:
    """Workflow execution state management"""

    workflow_id: str
    current_state: WorkflowState
    nodes_completed: int
    total_nodes: int
    start_time: datetime
    current_node: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NodeExecutionResult:
    """Individual node execution result"""

    node_id: str
    state: NodeState
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None

    def __post_init__(self):
        if self.end_time and self.execution_time_seconds is None:
            self.execution_time_seconds = (
                self.end_time - self.start_time
            ).total_seconds()


class WorkflowResultContract(PydanticAIContextualModel):
    """Enhanced workflow execution result contract with performance feedback integration"""

    # Basic workflow information
    workflow_id: str = Field(description="Workflow identifier")
    workflow_type: str = Field(
        description="Type of workflow (config_extraction, search, analysis)"
    )
    execution_state: WorkflowState = Field(description="Final execution state")

    # Execution results
    results: Dict[str, Any] = Field(description="Workflow results")
    node_results: List[Dict[str, Any]] = Field(description="Individual node results")

    # Performance and quality metrics
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    quality_scores: Dict[str, float] = Field(description="Quality assessment scores")
    total_execution_time: float = Field(ge=0.0, description="Total execution time")

    # Configuration and context tracking
    configurations_used: Dict[str, Dict[str, Any]] = Field(
        description="Configurations used by each agent"
    )
    domain_context: Optional[str] = Field(
        default=None, description="Domain context for this workflow"
    )

    # Error and warning handling
    error_log: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    recovery_actions: List[str] = Field(
        default_factory=list, description="Recovery actions taken"
    )

    # Performance feedback integration
    performance_feedback_points: List[PerformanceFeedbackPoint] = Field(
        default_factory=list,
        description="Performance feedback points generated during execution",
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list, description="Identified optimization opportunities"
    )

    # Workflow learning and improvement
    lessons_learned: List[str] = Field(
        default_factory=list, description="Lessons learned from this execution"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for workflow result processing"""
        return {
            "workflow_info": {
                "workflow_id": self.workflow_id,
                "workflow_type": self.workflow_type,
                "execution_state": self.execution_state.value,
                "domain_context": self.domain_context,
            },
            "performance_summary": {
                "total_execution_time": self.total_execution_time,
                "average_quality_score": (
                    sum(self.quality_scores.values()) / len(self.quality_scores)
                    if self.quality_scores
                    else 0.0
                ),
                "success_rate": (
                    1.0 if self.execution_state == WorkflowState.COMPLETED else 0.0
                ),
                "error_count": len(self.error_log),
            },
            "feedback_summary": {
                "feedback_points": len(self.performance_feedback_points),
                "optimization_opportunities": len(self.optimization_opportunities),
                "improvement_suggestions": len(self.improvement_suggestions),
            },
        }

    def is_successful(self) -> bool:
        """Check if workflow completed successfully"""
        return (
            self.execution_state == WorkflowState.COMPLETED and len(self.error_log) == 0
        )

    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score from all quality metrics"""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores.values()) / len(self.quality_scores)

    def generate_performance_feedback(self) -> List[PerformanceFeedbackPoint]:
        """Generate performance feedback points from workflow execution"""
        feedback_points = []

        # Generate feedback for overall workflow
        overall_feedback = PerformanceFeedbackPoint(
            agent_type="workflow",
            domain_name=self.domain_context or "general",
            operation_type=self.workflow_type,
            configuration_used=self.configurations_used,
            configuration_source="dynamic",
            execution_time_seconds=self.total_execution_time,
            success=self.is_successful(),
            quality_score=self.get_overall_quality_score(),
            output_quality_metrics=self.quality_scores,
        )
        feedback_points.append(overall_feedback)

        # Add any existing feedback points
        feedback_points.extend(self.performance_feedback_points)

        return feedback_points


# PersistedWorkflowState model deleted - was only in exports, no actual usage


# =============================================================================
# ANALYSIS & PROCESSING MODELS
# =============================================================================


class StatisticalAnalysis(BaseModel):
    """Statistical corpus analysis results"""

    corpus_path: Optional[str] = Field(
        default=None, description="Path to analyzed corpus"
    )
    total_documents: int = Field(ge=0, description="Total documents processed")
    total_tokens: int = Field(ge=0, description="Total tokens analyzed")
    total_characters: int = Field(ge=0, description="Total characters analyzed")
    token_frequencies: Dict[str, int] = Field(
        description="Token frequency distribution"
    )
    n_gram_patterns: Dict[str, int] = Field(description="N-gram pattern frequencies")
    vocabulary_size: int = Field(ge=0, description="Unique vocabulary size")
    document_structures: Dict[str, int] = Field(
        default_factory=dict, description="Document structure patterns"
    )
    average_document_length: float = Field(
        ge=0.0, description="Average document length"
    )
    document_count: int = Field(ge=0, description="Number of documents analyzed")
    length_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Document length distribution"
    )
    technical_term_density: float = Field(
        ge=0.0, le=1.0, description="Technical terminology density"
    )
    domain_specificity_score: float = Field(
        ge=0.0, le=1.0, description="Domain specificity indicator"
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0, description="Content complexity score"
    )
    analysis_confidence: float = Field(
        default=StatisticalConstants.TECHNICAL_CONTENT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    processing_time_seconds: float = Field(
        ge=0.0, description="Analysis processing time"
    )


# SemanticPatterns deleted in Phase 2 - was unused semantic analysis model

# CombinedPatterns deleted in Phase 2 - was unused pattern combination model


class ExtractedKnowledge(BaseModel):
    """Extracted knowledge representation"""

    entities: List[Dict[str, Any]] = Field(description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(description="Extracted relationships")
    confidence_scores: Dict[str, float] = Field(
        description="Extraction confidence scores"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extraction metadata"
    )
    quality_metrics: Dict[str, float] = Field(description="Quality assessment metrics")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")


class ExtractionResults(BaseModel):
    """Knowledge extraction results"""

    domain: str = Field(description="Source domain")
    extracted_knowledge: ExtractedKnowledge = Field(description="Extracted knowledge")
    quality_assessment: Dict[str, float] = Field(
        description="Quality assessment scores"
    )
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    validation_results: Dict[str, Any] = Field(description="Validation results")


class UnifiedExtractionResult(BaseModel):
    """Unified extraction processor result"""

    extraction_id: str = Field(default="", description="Unique extraction identifier")
    document_id: str = Field(default="", description="Source document identifier")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted relationships"
    )
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Processing time in seconds"
    )
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )
    extraction_method: str = Field(
        default="unified", description="Extraction method used"
    )
    confidence_distribution: Dict[str, float] = Field(
        default_factory=dict, description="Confidence score distribution"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )
    performance_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Performance statistics"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


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


# =============================================================================
# VALIDATION & ERROR MODELS
# =============================================================================


class ValidationResult(BaseModel):
    """Unified validation result model - eliminates duplicates"""

    domain: str = Field(description="Validation domain")
    valid: bool = Field(default=True, description="Validation success status")
    missing_keys: List[str] = Field(
        default_factory=list, description="Missing required keys"
    )
    invalid_values: List[str] = Field(
        default_factory=list, description="Invalid value descriptions"
    )
    source_validation: List[Dict[str, Any]] = Field(
        default_factory=list, description="Source-specific validation results"
    )
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Validation confidence"
    )
    validation_time: float = Field(
        default=0.0, ge=0.0, description="Validation processing time"
    )


# QualityMetrics deleted - over-engineered quality assessment with 15+ complex metrics never used
# This elaborate quality scoring system (entity/relationship precision/recall estimates, predictions) had zero instantiation
# Simple quality metrics can be calculated inline when actually needed


# ErrorContext moved to error handling models section


class ErrorHandlingContract(BaseModel):
    """Error handling contract specification"""

    supported_error_types: List[ErrorCategory] = Field(
        description="Supported error categories"
    )
    error_recovery_strategies: Dict[str, str] = Field(
        description="Error recovery strategies"
    )
    escalation_rules: Dict[str, str] = Field(description="Error escalation rules")
    monitoring_requirements: Dict[str, Any] = Field(
        description="Error monitoring requirements"
    )
    notification_channels: List[str] = Field(description="Error notification channels")


# =============================================================================
# DEPENDENCY MODELS
# =============================================================================

# =============================================================================
# CORRECTED PYDANTIC AI DEPENDENCIES
# =============================================================================

# According to PydanticAI docs, RunContext.deps should contain the actual dependencies
# not a wrapper object. Each agent should have its own specific dependency type.


class AzureServicesDeps(BaseModel):
    """Direct Azure Services dependency for RunContext[AzureServicesDeps]"""

    # This should be the actual ConsolidatedAzureServices instance
    service_container: Any = Field(description="ConsolidatedAzureServices instance")

    class Config:
        arbitrary_types_allowed = True

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status directly from the container"""
        return self.service_container.get_service_status()


# CacheManagerDeps deleted - over-engineered dependency wrapper with unnecessary abstraction
# This PydanticAI dependency wrapper duplicated cache_manager functionality
# Agents can access cache_manager directly without this abstraction layer

# SharedDeps deleted in Phase 4 - was DEPRECATED legacy dependency model


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
# COMMUNICATION MODELS
# =============================================================================

# GraphMessage model deleted - was only in exports, no actual usage


# GraphStatus model deleted - was only in exports, no actual usage


# =============================================================================
# CACHE AND MONITORING MODELS
# =============================================================================

# CacheContract model deleted - was only in exports, no actual usage


# MonitoringContract deleted - over-engineered monitoring config with zero actual usage
# Complex monitoring infrastructure (metrics collection, health checks, trace sampling) never instantiated
# Observability can be handled through existing Azure monitoring when actually needed


# =============================================================================
# SPECIALIZED PROCESSING MODELS
# =============================================================================


@dataclass
# ConfigurationMetadata deleted in Phase 2 - was unused metadata tracking model

# GeneratedConfiguration deleted in Phase 2 - was unused generated config model


# CacheEntry moved to cache models section


@dataclass
class CacheMetrics:
    """Cache performance metrics"""

    hit_rate: float
    miss_rate: float
    total_requests: int
    cache_size: int
    memory_usage_mb: float
    eviction_count: int
    last_updated: datetime


@dataclass
class ServiceHealth:
    """Service health status"""

    service_name: str
    status: HealthStatus
    response_time_ms: float
    error_rate: float
    last_check: datetime
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# =============================================================================
# ERROR HANDLING MODELS
# =============================================================================


@dataclass
class ErrorContext:
    """Comprehensive error context for analysis and recovery"""

    error: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    component: str
    parameters: Dict[str, Any] = None
    timestamp: float = None
    attempt_count: int = 0
    max_retries: int = 3
    recovery_strategy: Optional[str] = None
    user_message: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.timestamp is None:
            import time

            self.timestamp = time.time()

    @property
    def should_retry(self) -> bool:
        """Determine if error should be retried"""
        return self.attempt_count < self.max_retries and self.category in [
            ErrorCategory.AZURE_SERVICE,
            ErrorCategory.TIMEOUT,
        ]

    @property
    def backoff_delay(self) -> float:
        """Calculate exponential backoff delay"""
        base_delay = 1.0
        return min(30.0, base_delay * (2**self.attempt_count))


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""

    total_errors: int = 0
    errors_by_category: Dict[str, int] = None
    errors_by_severity: Dict[str, int] = None
    errors_by_component: Dict[str, int] = None
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_recovery_time: float = 0.0
    recent_errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors_by_category is None:
            self.errors_by_category = {}
        if self.errors_by_severity is None:
            self.errors_by_severity = {}
        if self.errors_by_component is None:
            self.errors_by_component = {}
        if self.recent_errors is None:
            self.recent_errors = []

    @property
    def recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        total_recoveries = self.successful_recoveries + self.failed_recoveries
        if total_recoveries == 0:
            return 0.0
        return (self.successful_recoveries / total_recoveries) * 100


# =============================================================================
# CACHE MODELS
# =============================================================================


@dataclass
class CacheEntry:
    """Unified cache entry with comprehensive metadata"""

    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: int
    cache_type: str = "general"

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        import time

        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access statistics for LRU and performance tracking"""
        import time

        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CachePerformanceMetrics:
    """Comprehensive performance metrics for monitoring"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fast_lookups: int = 0
    pattern_index_hits: int = 0
    query_cache_hits: int = 0
    domain_signature_hits: int = 0
    evictions: int = 0
    average_lookup_time: float = 0.0

    @property
    def hit_rate_percent(self) -> float:
        """Calculate cache hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        from agents.core.math_expressions import EXPR

        return EXPR.calculate_percentage(self.cache_hits, self.total_requests)

    @property
    def fast_lookup_percent(self) -> float:
        """Calculate percentage of fast lookups"""
        if self.total_requests == 0:
            return 0.0
        from agents.core.math_expressions import EXPR

        return EXPR.calculate_percentage(self.fast_lookups, self.total_requests)


# =============================================================================
# MEMORY MANAGEMENT MODELS
# =============================================================================


@dataclass
class MemoryStatus:
    """Memory management status and metrics"""

    total_items: int = 0
    memory_limit_mb: float = 200.0
    estimated_usage_mb: float = 0.0
    evictions: int = 0
    last_cleanup: float = 0.0
    health_status: str = "healthy"

    @property
    def utilization_percent(self) -> float:
        """Calculate memory utilization percentage"""
        if self.memory_limit_mb == 0:
            return 0.0
        from agents.core.math_expressions import EXPR

        return EXPR.calculate_utilization_percentage(
            self.estimated_usage_mb, self.memory_limit_mb
        )

    def update_health_status(self):
        """Update health status based on utilization"""
        if self.utilization_percent > 90:
            self.health_status = "critical"
        elif self.utilization_percent > 75:
            self.health_status = "warning"
        else:
            self.health_status = "healthy"


# =============================================================================
# CONSOLIDATED EXTRACTION BASE MODELS (from agents/shared/extraction_base.py)
# =============================================================================

# ExtractionType enum deleted - was only in exports, no actual usage


class ExtractionStatus(str, Enum):
    """Status of extraction operations"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ExtractionContext(PydanticAIContextualModel):
    """Context information for extraction operations"""

    document_id: Optional[str] = Field(default=None, description="Document identifier")
    text_segment: str = Field(description="Text segment being processed")
    segment_start: int = Field(ge=0, description="Start position in full document")
    segment_end: int = Field(ge=0, description="End position in full document")

    # Context metadata
    domain_type: Optional[str] = Field(default=None, description="Detected domain type")
    language: str = Field(default="en", description="Text language")
    processing_hints: Dict[str, Any] = Field(
        default_factory=dict, description="Processing hints and parameters"
    )

    # Quality indicators
    text_quality_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Quality score of input text"
    )
    preprocessing_applied: List[str] = Field(
        default_factory=list, description="List of preprocessing steps applied"
    )

    @validator("segment_end")
    def validate_segment_bounds(cls, v, values):
        if "segment_start" in values and v < values["segment_start"]:
            raise ValueError("segment_end must be >= segment_start")
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for extraction operations"""
        return {
            "document_context": {
                "document_id": self.document_id,
                "segment_bounds": (self.segment_start, self.segment_end),
                "domain_type": self.domain_type,
                "language": self.language,
            },
            "extraction_context": {
                "text_quality": self.text_quality_score,
                "processing_hints": self.processing_hints,
                "preprocessing_applied": self.preprocessing_applied,
            },
        }


# BaseExtractionResult class deleted - was not exported and unused


# =============================================================================
# CONSOLIDATED TEXT STATISTICS MODELS (from agents/shared/text_statistics.py)
# =============================================================================


class TextStatistics(BaseModel):
    """Comprehensive text statistics for document analysis"""

    char_count: int = Field(ge=0, description="Total character count")
    word_count: int = Field(ge=0, description="Total word count")
    sentence_count: int = Field(ge=0, description="Total sentence count")
    paragraph_count: int = Field(ge=0, description="Total paragraph count")

    avg_words_per_sentence: float = Field(
        ge=0.0, description="Average words per sentence"
    )
    avg_chars_per_word: float = Field(ge=0.0, description="Average characters per word")
    lexical_diversity: float = Field(
        ge=0.0, le=1.0, description="Lexical diversity score"
    )

    readability_score: float = Field(
        ge=0.0, le=100.0, description="Flesch Reading Ease score"
    )

    def calculate_readability(self) -> float:
        """Calculate Flesch Reading Ease score"""
        if self.sentence_count == 0 or self.word_count == 0:
            return 0.0
        return min(
            100.0,
            max(
                0.0,
                206.835
                - (1.015 * self.avg_words_per_sentence)
                - (84.6 * (self.avg_chars_per_word / 4.7)),
            ),
        )


# DocumentComplexityProfile deleted - over-engineered complexity analysis system with zero actual usage
# This elaborate document complexity scoring system was never instantiated in practice
# Simple text statistics and basic chunking strategies are sufficient for current needs


# =============================================================================
# CONSOLIDATED CONFIDENCE MODELS (from agents/shared/confidence_calculator.py)
# =============================================================================


class ConfidenceScore(BaseModel):
    """Individual confidence score with metadata"""

    value: float = Field(ge=0.0, le=1.0, description="Confidence score value")
    source: str = Field(description="Source of confidence score")
    weight: float = Field(
        ge=0.0, le=1.0, description="Weight of this score in aggregation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# AggregatedConfidence deleted - over-engineered confidence aggregation with statistical measures never used
# Only .final_confidence was accessed, which can be replaced with simple float confidence
# Complex statistical measures (mean, std, min, max, consensus_strength) were never utilized


# EntityConfidenceFactors deleted - over-engineered confidence factor system with 7 separate metrics
# Only instantiated once for complex calculation that can be simplified
# Individual factors (context_clarity, type_consistency, etc.) can be calculated inline


class RelationshipConfidenceFactors(BaseModel):
    """Factors affecting relationship extraction confidence"""

    entity_confidence_product: float = Field(
        ge=0.0, le=1.0, description="Product of related entity confidences"
    )
    relationship_type_clarity: float = Field(
        ge=0.0, le=1.0, description="Relationship type clarity"
    )
    contextual_support: float = Field(
        ge=0.0, le=1.0, description="Contextual evidence support"
    )
    semantic_coherence: float = Field(
        ge=0.0, le=1.0, description="Semantic coherence score"
    )
    distance_penalty: float = Field(
        ge=0.0, le=1.0, description="Distance-based penalty factor"
    )
    domain_pattern_match: float = Field(
        ge=0.0, le=1.0, description="Domain pattern matching score"
    )


# =============================================================================
# CONSOLIDATED CONTENT PREPROCESSING MODELS (from agents/shared/content_preprocessing.py)
# =============================================================================


class TextCleaningOptions(BaseModel):
    """Configuration for text cleaning operations"""

    remove_extra_whitespace: bool = Field(
        default=True, description="Remove extra whitespace"
    )
    normalize_unicode: bool = Field(
        default=True, description="Normalize Unicode characters"
    )
    remove_special_chars: bool = Field(
        default=False, description="Remove special characters"
    )
    lowercase: bool = Field(default=False, description="Convert to lowercase")
    remove_numbers: bool = Field(default=False, description="Remove numeric characters")
    remove_punctuation: bool = Field(default=False, description="Remove punctuation")

    # Advanced cleaning options
    remove_urls: bool = Field(default=True, description="Remove URLs")
    remove_emails: bool = Field(default=True, description="Remove email addresses")
    remove_phone_numbers: bool = Field(default=True, description="Remove phone numbers")
    preserve_sentence_structure: bool = Field(
        default=True, description="Preserve sentence boundaries"
    )


class CleanedContent(BaseModel):
    """Result of content cleaning operation"""

    original_text: str = Field(description="Original input text")
    cleaned_text: str = Field(description="Cleaned output text")
    cleaning_options: TextCleaningOptions = Field(
        description="Cleaning options applied"
    )

    # Cleaning statistics
    characters_removed: int = Field(ge=0, description="Number of characters removed")
    words_removed: int = Field(ge=0, description="Number of words removed")
    sentences_removed: int = Field(ge=0, description="Number of sentences removed")

    # Quality metrics
    cleaning_quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality of cleaning operation"
    )
    content_preservation_score: float = Field(
        ge=0.0, le=1.0, description="How well original content was preserved"
    )

    cleaning_warnings: List[str] = Field(
        default_factory=list, description="Warnings from cleaning process"
    )


class ContentChunker(BaseModel):
    """Configuration for content chunking operations"""

    chunk_size: int = Field(
        ge=50, le=5000, description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(
        ge=0, le=1000, description="Overlap between chunks in characters"
    )
    respect_sentence_boundaries: bool = Field(
        default=True, description="Respect sentence boundaries"
    )
    respect_paragraph_boundaries: bool = Field(
        default=True, description="Respect paragraph boundaries"
    )

    min_chunk_size: int = Field(ge=10, description="Minimum chunk size")
    max_chunk_size: int = Field(le=10000, description="Maximum chunk size")


class ContentChunk(BaseModel):
    """Individual content chunk with metadata"""

    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Chunk content")
    start_position: int = Field(ge=0, description="Start position in original document")
    end_position: int = Field(ge=0, description="End position in original document")

    # Chunk metadata
    chunk_index: int = Field(ge=0, description="Chunk index in sequence")
    word_count: int = Field(ge=0, description="Word count in chunk")
    character_count: int = Field(ge=0, description="Character count in chunk")

    # Overlap information
    overlap_with_previous: int = Field(
        ge=0, description="Characters overlapping with previous chunk"
    )
    overlap_with_next: int = Field(
        ge=0, description="Characters overlapping with next chunk"
    )

    # Quality indicators
    boundary_quality: float = Field(
        ge=0.0, le=1.0, description="Quality of chunk boundaries"
    )
    content_completeness: float = Field(
        ge=0.0, le=1.0, description="Completeness of content in chunk"
    )


# =============================================================================
# CONSOLIDATED KNOWLEDGE EXTRACTION MODELS
# =============================================================================


class KnowledgeExtractionResult(PydanticAIContextualModel):
    """
    PydanticAI-enhanced knowledge extraction result following output validator patterns

    Replaces the previous UnifiedExtractionResult with proper Pydantic validation
    and cross-agent compatibility.
    """

    # Core extraction results
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted and validated entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted and validated relationships"
    )

    # Extraction statistics
    entity_count: int = Field(ge=0, description="Total number of entities extracted")
    relationship_count: int = Field(
        ge=0, description="Total number of relationships extracted"
    )
    unique_entity_types: int = Field(ge=0, description="Number of unique entity types")
    unique_relationship_types: int = Field(
        ge=0, description="Number of unique relationship types"
    )

    # Quality metrics
    avg_entity_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across entities"
    )
    avg_relationship_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across relationships"
    )
    extraction_quality_score: float = Field(
        ge=0.0, le=1.0, description="Overall extraction quality"
    )

    # Graph metrics
    entity_pairs_connected: int = Field(
        ge=0, description="Number of entity pairs connected by relationships"
    )
    graph_density: float = Field(
        ge=0.0, le=1.0, description="Density of the relationship graph"
    )
    connected_components: int = Field(
        ge=0, description="Number of disconnected graph components"
    )

    # Processing metadata
    extraction_method: str = Field(description="Primary extraction method used")
    processing_time_ms: float = Field(
        ge=0.0, description="Processing time in milliseconds"
    )
    text_length: int = Field(ge=0, description="Length of processed text")
    strategies_used: List[str] = Field(
        default_factory=list, description="List of extraction strategies applied"
    )

    # Validation results
    validation_passed: bool = Field(
        default=True, description="Whether PydanticAI validation passed"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )

    @validator("entity_count")
    def validate_entity_count(cls, v, values):
        if "entities" in values and v != len(values["entities"]):
            raise ValueError("entity_count must match length of entities list")
        return v

    @validator("relationship_count")
    def validate_relationship_count(cls, v, values):
        if "relationships" in values and v != len(values["relationships"]):
            raise ValueError(
                "relationship_count must match length of relationships list"
            )
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for knowledge extraction results"""
        return {
            "extraction_summary": {
                "entity_count": self.entity_count,
                "relationship_count": self.relationship_count,
                "quality_score": self.extraction_quality_score,
                "validation_passed": self.validation_passed,
            },
            "graph_metrics": {
                "entity_pairs_connected": self.entity_pairs_connected,
                "graph_density": self.graph_density,
                "connected_components": self.connected_components,
            },
            "performance": {
                "processing_time_ms": self.processing_time_ms,
                "text_length": self.text_length,
                "extraction_method": self.extraction_method,
                "strategies_used": self.strategies_used,
            },
        }


class KnowledgeValidationResult(BaseModel):
    """Simple validation result for knowledge extraction using PydanticAI validation"""

    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Validation warning messages"
    )
    entity_count: int = Field(ge=0, description="Number of entities validated")
    relationship_count: int = Field(
        ge=0, description="Number of relationships validated"
    )


# =============================================================================
# ADDITIONAL CORE MODELS
# =============================================================================

# ConfigurationMetadata moved to specialized processing models section


# ModelSelectionCriteria deleted in Phase 5 - was unused model selection criteria


@dataclass
class WorkflowStateBridge:
    """Bridge data between workflow states"""

    source_state: str
    target_state: str
    transition_data: Dict[str, Any]
    timestamp: float

    def __post_init__(self):
        if self.transition_data is None:
            self.transition_data = {}


@dataclass
class ServiceContainerConfig:
    """Service container configuration"""

    container_name: str
    service_mappings: Dict[str, Any]
    initialization_order: List[str]
    health_check_config: Dict[str, Any]
    retry_config: Dict[str, Any]

    def __post_init__(self):
        if self.service_mappings is None:
            self.service_mappings = {}
        if self.initialization_order is None:
            self.initialization_order = []
        if self.health_check_config is None:
            self.health_check_config = {}
        if self.retry_config is None:
            self.retry_config = {}


# =============================================================================
# DOMAIN INTELLIGENCE ANALYZER MODELS
# =============================================================================


@dataclass
class UnifiedAnalysis:
    """Unified content analysis result"""

    complexity_score: float
    technical_terms: List[str]
    domain_indicators: List[str]
    quality_metrics: Dict[str, float]
    processing_recommendations: Dict[str, Any] = None

    def __post_init__(self):
        if self.processing_recommendations is None:
            self.processing_recommendations = {}


# ContentQuality deleted - over-structured quality assessment with zero actual usage
# Complex quality scoring system (readability, technical accuracy, completeness) never instantiated
# Quality metrics can be calculated inline when actually needed


@dataclass
class BackgroundProcessingConfig:
    """Background processing configuration"""

    processing_enabled: bool
    batch_size: int
    processing_interval: float
    max_retries: int
    timeout_seconds: float


@dataclass
class BackgroundProcessingResult:
    """Background processing execution result"""

    processing_id: str
    success: bool
    items_processed: int
    processing_time: float
    error_message: Optional[str] = None


@dataclass
# ConfigurationRecommendations deleted - over-engineered recommendation system with minimal usage
# Only instantiated once in hybrid_configuration_generator but never actually used in production
# Simple dictionary can replace this when/if recommendations are actually needed


@dataclass
class LLMExtraction:
    """LLM extraction results for configuration generation"""

    domain_characteristics: List[str]
    key_concepts: List[str]
    entity_types: List[str]
    relationship_patterns: List[str]
    processing_complexity: str
    reasoning_quality: float = 0.8


# PatternStatistics deleted - simple dataclass that can be replaced with Dict[str, Any]
# Only used for basic statistics tracking in pattern_engine
# Dict with same fields provides identical functionality with less complexity


@dataclass
class ProcessingStats:
    """Background processing statistics"""

    start_time: float = 0.0
    end_time: float = 0.0
    domains_processed: int = 0
    files_processed: int = 0
    processing_errors: int = 0

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    @property
    def files_per_second(self) -> float:
        return self.files_processed / self.total_time if self.total_time > 0 else 0.0


@dataclass
class DomainSignature:
    """Domain signature containing patterns and analysis for a specific domain"""

    domain: str
    patterns: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    content_analysis: Optional[Any] = None  # DomainAnalysisResult type
    processing_timestamp: float = 0.0
    cache_key: str = ""


@dataclass
class DomainConfig:
    """Domain-specific configuration"""

    domain_name: str
    similarity_thresholds: Dict[str, float]
    processing_parameters: Dict[str, Any]
    model_configurations: Dict[str, Any]
    validation_rules: List[str] = None

    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []


@dataclass
class LearnedPattern:
    """Pattern learned from domain analysis"""

    pattern_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence_score: float
    usage_count: int = 0


@dataclass
class ExtractedPatterns:
    """Collection of extracted patterns"""

    domain: str
    entity_patterns: List[LearnedPattern]
    relationship_patterns: List[LearnedPattern]
    linguistic_patterns: List[LearnedPattern]
    extraction_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.extraction_metadata is None:
            self.extraction_metadata = {}


@dataclass
class DomainGenerationConfig:
    """Domain generation configuration"""

    generation_strategy: str
    quality_thresholds: Dict[str, float]
    output_format: str
    validation_requirements: List[str] = None

    def __post_init__(self):
        if self.validation_requirements is None:
            self.validation_requirements = []


@dataclass
class DomainPatternResult:
    """Domain pattern extraction result"""

    domain: str
    patterns_extracted: int
    pattern_confidence: float
    processing_time: float
    patterns: List[LearnedPattern] = None

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []


@dataclass
class DomainConfigResult:
    """Domain configuration generation result"""

    domain: str
    config_generated: bool
    config_data: Dict[str, Any]
    generation_confidence: float
    validation_passed: bool


# =============================================================================
# DYNAMIC MANAGER MODELS
# =============================================================================

# RuntimeConfigurationData deleted in Phase 4 - was unused runtime config model


# =============================================================================
# SEARCH ORCHESTRATOR MODELS
# =============================================================================


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


# =============================================================================
# ARCHITECTURE COMPLIANCE MODELS
# =============================================================================

# DataDrivenExtractionConfiguration deleted in Phase 3 - was unused data-driven config model
# ArchitectureComplianceValidator deleted in Phase 3 - was unused architecture validation model


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Utility functions deleted in Phase 4 - were unused factory functions:
# create_base_request, create_validation_result, create_error_context


# =============================================================================
# PYDANTIC AI OUTPUT VALIDATOR MODELS
# =============================================================================


class ExtractionQualityOutput(BaseModel):
    """PydanticAI output validator for extraction quality assessment"""

    entities_per_text: float = Field(
        ge=1.0,
        le=20.0,
        description="Optimal entity extraction density - replaces hardcoded thresholds",
    )
    relations_per_entity: float = Field(
        ge=0.3,
        le=5.0,
        description="Healthy relationship coverage - agent learns optimal ratio",
    )
    avg_entity_confidence: float = Field(
        ge=0.6,
        le=1.0,
        description="Minimum acceptable entity confidence - dynamic threshold",
    )
    avg_relation_confidence: float = Field(
        ge=0.6,
        le=1.0,
        description="Minimum acceptable relationship confidence - learned threshold",
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite quality score - agent-determined weighting",
    )
    quality_tier: str = Field(
        pattern="^(excellent|good|acceptable|needs_improvement)$",
        description="Quality classification - agent learns tier boundaries",
    )


class ValidatedEntity(BaseModel):
    """PydanticAI output validator for entity extraction results"""

    name: str = Field(min_length=1, description="Entity name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score - agent determines threshold"
    )
    entity_type: str = Field(
        min_length=1, description="Entity classification - learned from domain"
    )
    extraction_method: str = Field(
        pattern="^(pattern_based|nlp_based|hybrid)$",
        description="Method used for extraction",
    )


class ValidatedRelationship(BaseModel):
    """PydanticAI output validator for relationship extraction"""

    source_entity: str = Field(min_length=1, description="Source entity")
    target_entity: str = Field(min_length=1, description="Target entity")
    relation_type: str = Field(min_length=1, description="Relationship type")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Relationship confidence - agent learned threshold"
    )


class ContentAnalysisOutput(BaseModel):
    """PydanticAI output validator for content analysis results"""

    word_count: int = Field(
        ge=50,
        description="Minimum meaningful content - agent determines based on domain",
    )
    vocabulary_richness: float = Field(
        ge=0.1,
        le=1.0,
        description="Vocabulary diversity - learned from corpus analysis",
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0, description="Content complexity - agent-driven assessment"
    )
    is_meaningful_content: bool = Field(
        description="Content quality gate - replaces manual validation chains"
    )
    technical_density: float = Field(
        ge=0.0, le=1.0, description="Technical term density - domain-specific threshold"
    )


class DomainConfigurationOutput(BaseModel):
    """PydanticAI output validator for domain configuration generation"""

    entity_confidence_threshold: float = Field(
        ge=0.3,
        le=0.9,
        description="Domain-specific entity threshold - learned from corpus",
    )
    relationship_confidence_threshold: float = Field(
        ge=0.3,
        le=0.9,
        description="Domain-specific relationship threshold - data-driven",
    )
    similarity_threshold: float = Field(
        ge=0.3,
        le=0.9,
        description="Similarity threshold - optimized for domain characteristics",
    )
    optimal_chunk_size: int = Field(
        ge=100,
        le=2000,
        description="Optimal chunk size - based on document structure analysis",
    )
    processing_complexity: str = Field(
        pattern="^(low|medium|high)$",
        description="Processing complexity level - determined by content analysis",
    )


# =============================================================================
# TIER 2: CONSOLIDATED SERVICE MODELS (Replaces duplicate model files)
# =============================================================================


class ConsolidatedAzureConfiguration(PydanticAIContextualModel):
    """Consolidated Azure service configuration with dynamic resolution"""

    # Core Azure settings (resolved from centralized config)
    openai_endpoint: str = Field(description="Azure OpenAI endpoint")
    openai_api_version: str = Field(
        default_factory=lambda: InfrastructureConstants.OPENAI_API_VERSION
    )
    search_endpoint: str = Field(description="Azure Cognitive Search endpoint")
    cosmos_endpoint: str = Field(description="Azure Cosmos DB endpoint")

    # Service timeouts (resolved dynamically)
    openai_timeout: int = Field(
        default_factory=lambda: PerformanceAdaptiveConstants.DEFAULT_TIMEOUT
    )
    search_timeout: int = Field(
        default_factory=lambda: PerformanceAdaptiveConstants.DEFAULT_TIMEOUT
    )
    cosmos_timeout: int = Field(
        default_factory=lambda: PerformanceAdaptiveConstants.DEFAULT_TIMEOUT
    )

    # Request limits (from constants, not hardcoded)
    max_tokens: int = Field(
        default_factory=lambda: InfrastructureConstants.MAX_TOKENS_GPT4
    )
    max_search_results: int = Field(
        default_factory=lambda: InfrastructureConstants.MAX_SEARCH_RESULTS
    )
    max_retries: int = Field(
        default_factory=lambda: PerformanceAdaptiveConstants.MAX_RETRIES
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for Azure service operations"""
        return {
            "azure_config": self.model_dump(exclude={"run_context_data"}),
            "service_endpoints": {
                "openai": self.openai_endpoint,
                "search": self.search_endpoint,
                "cosmos": self.cosmos_endpoint,
            },
        }

    @classmethod
    def from_centralized_config(
        cls, domain_name: str = None
    ) -> "ConsolidatedAzureConfiguration":
        """Create configuration from centralized config system"""
        azure_config = ConfigurationResolver.resolve_azure_config()
        return cls(**azure_config)


class ConsolidatedExtractionConfiguration(PydanticAIContextualModel):
    """Consolidated knowledge extraction configuration with dynamic resolution"""

    # Core extraction parameters (resolved from Domain Intelligence Agent)
    entity_confidence_threshold: float = Field(
        description="Entity extraction confidence threshold (learned from domain analysis)"
    )
    relationship_confidence_threshold: float = Field(
        description="Relationship extraction confidence threshold (learned from domain analysis)"
    )
    chunk_size: int = Field(
        description="Document chunk size (learned from document characteristics)"
    )
    chunk_overlap: int = Field(
        description="Chunk overlap size (calculated from chunk_size)"
    )

    # Processing parameters (from centralized config)
    max_entities_per_chunk: int = Field(
        default_factory=lambda: KnowledgeExtractionConstants.MAX_ENTITIES_PER_CHUNK
    )
    min_relationship_strength: float = Field(
        default_factory=lambda: KnowledgeExtractionConstants.MIN_RELATIONSHIP_STRENGTH
    )

    # Domain context (from Domain Intelligence Agent)
    domain_name: str = Field(description="Source domain name")
    technical_vocabulary: List[str] = Field(
        default_factory=list, description="Domain-specific vocabulary"
    )
    expected_entity_types: List[str] = Field(
        default_factory=list, description="Expected entity types for domain"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for Knowledge Extraction Agent"""
        return {
            "agent_type": "knowledge_extraction",
            "extraction_config": self.model_dump(exclude={"run_context_data"}),
            "domain_context": {
                "domain_name": self.domain_name,
                "vocabulary": self.technical_vocabulary,
                "entity_types": self.expected_entity_types,
            },
        }

    @classmethod
    def from_centralized_config(
        cls, domain_name: str
    ) -> "ConsolidatedExtractionConfiguration":
        """Create extraction configuration from centralized config and Domain Intelligence Agent"""
        extraction_config = ConfigurationResolver.resolve_extraction_config(domain_name)
        # domain_name is already included in extraction_config, don't duplicate
        return cls(**extraction_config)


class ConsolidatedSearchConfiguration(PydanticAIContextualModel):
    """Consolidated tri-modal search configuration with dynamic resolution"""

    # Core search parameters (resolved from Domain Intelligence Agent)
    vector_similarity_threshold: float = Field(
        description="Vector similarity threshold (learned from domain analysis)"
    )
    vector_top_k: int = Field(
        description="Number of top vector results (query-complexity driven)"
    )
    graph_hop_count: int = Field(
        description="Graph traversal hop count (domain-specific relationship depth)"
    )
    graph_min_relationship_strength: float = Field(
        description="Minimum graph relationship strength (learned from relationship patterns)"
    )
    gnn_prediction_confidence: float = Field(
        description="GNN prediction confidence threshold (learned from GNN training performance)"
    )

    # Tri-modal weights (optimized for domain)
    tri_modal_weights: Dict[str, float] = Field(
        description="Domain-optimized weights for vector/graph/gnn search"
    )

    # Query complexity adaptation
    query_complexity_weights: Dict[str, float] = Field(
        default_factory=dict, description="Learned complexity multipliers"
    )

    # Domain context
    domain_name: str = Field(description="Source domain name")
    learned_at: datetime = Field(
        default_factory=datetime.now, description="Configuration generation timestamp"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for Universal Search Agent"""
        return {
            "agent_type": "universal_search",
            "search_config": self.model_dump(exclude={"run_context_data"}),
            "search_context": {
                "domain_name": self.domain_name,
                "tri_modal_weights": self.tri_modal_weights,
                "complexity_weights": self.query_complexity_weights,
            },
        }

    @classmethod
    def from_centralized_config(
        cls, domain_name: str
    ) -> "ConsolidatedSearchConfiguration":
        """Create search configuration from centralized config and Domain Intelligence Agent"""
        search_config = ConfigurationResolver.resolve_search_config(domain_name)
        # domain_name is already included in search_config, don't duplicate
        return cls(**search_config)

        # =============================================================================
        # TIER 3: ENHANCED AGENT CONTRACT MODELS WITH PYDANTIC AI INTEGRATION
        # =============================================================================

        # Enhanced Contract Models deleted - redundant with basic contract models
        # These "Enhanced" contracts were minimally used and duplicated functionality
        # Basic DomainAnalysisContract, KnowledgeExtractionContract, SearchContract provide same capabilities
        # The PydanticAI RunContext integration can be handled in the basic contracts when needed
        """Provide RunContext data for Universal Search Agent"""
        return {
            "agent_type": "universal_search",
            "capabilities": self.search_capabilities,
            "search_types": [st.value for st in self.supported_search_types],
        }


# =============================================================================
# PYDANTIC AI OUTPUT VALIDATOR FUNCTIONS
# =============================================================================


def validate_extraction_quality(result: Dict[str, Any]) -> ExtractionQualityOutput:
    """Replace 150+ lines of manual quality assessment with PydanticAI validation"""
    return ExtractionQualityOutput(**result)


def validate_entity_extraction(entities: List[Dict[str, Any]]) -> List[ValidatedEntity]:
    """Replace 400+ lines of entity confidence calculation with PydanticAI constraints"""
    return [ValidatedEntity(**entity) for entity in entities]


def validate_relationship_extraction(
    relationships: List[Dict[str, Any]],
) -> List[ValidatedRelationship]:
    """Replace complex relationship validation chains with PydanticAI patterns"""
    return [ValidatedRelationship(**rel) for rel in relationships]


def validate_content_analysis(analysis: Dict[str, Any]) -> ContentAnalysisOutput:
    """Replace 70+ lines of content quality validation with agent-driven constraints"""
    return ContentAnalysisOutput(**analysis)


def validate_domain_configuration(config: Dict[str, Any]) -> DomainConfigurationOutput:
    """Replace hardcoded configuration logic with agent-learned parameters"""
    return DomainConfigurationOutput(**config)


# =============================================================================
# 7. KNOWLEDGE EXTRACTION MODELS (from knowledge_extraction/ processors)
# =============================================================================


class ValidationResult(PydanticAIContextualModel):
    """Simple validation result using PydanticAI validation"""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    entity_count: int = Field(..., ge=0, description="Number of entities validated")
    relationship_count: int = Field(
        ..., ge=0, description="Number of relationships validated"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI validation operations"""
        return {
            "validation_result": {
                "is_valid": self.is_valid,
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
                "entity_count": self.entity_count,
                "relationship_count": self.relationship_count,
            },
            "validation_metrics": {
                "success_rate": 1.0 if self.is_valid else 0.0,
                "quality_indicators": {
                    "has_errors": len(self.errors) > 0,
                    "has_warnings": len(self.warnings) > 0,
                    "entity_validation": self.entity_count > 0,
                    "relationship_validation": self.relationship_count > 0,
                },
            },
        }


class KnowledgeExtractionResult(PydanticAIContextualModel):
    """
    PydanticAI-enhanced knowledge extraction result following output validator patterns

    Replaces the previous UnifiedExtractionResult with proper Pydantic validation
    and cross-agent compatibility.
    """

    # Core extraction results
    entities: List[ValidatedEntity] = Field(
        default_factory=list, description="Extracted and validated entities"
    )
    relationships: List[ValidatedRelationship] = Field(
        default_factory=list, description="Extracted and validated relationships"
    )

    # Extraction statistics
    entity_count: int = Field(ge=0, description="Total number of entities extracted")
    relationship_count: int = Field(
        ge=0, description="Total number of relationships extracted"
    )
    unique_entity_types: int = Field(ge=0, description="Number of unique entity types")
    unique_relationship_types: int = Field(
        ge=0, description="Number of unique relationship types"
    )

    # Quality metrics
    avg_entity_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across entities"
    )
    avg_relationship_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across relationships"
    )
    extraction_quality_score: float = Field(
        ge=0.0, le=1.0, description="Overall extraction quality"
    )

    # Graph metrics
    entity_pairs_connected: int = Field(
        ge=0, description="Number of entity pairs connected by relationships"
    )
    graph_density: float = Field(
        ge=0.0, le=1.0, description="Density of the relationship graph"
    )
    connected_components: int = Field(
        ge=0, description="Number of disconnected graph components"
    )

    # Processing metadata
    extraction_method: str = Field(..., description="Primary extraction method used")
    processing_time_ms: float = Field(
        ge=0.0, description="Processing time in milliseconds"
    )
    text_length: int = Field(ge=0, description="Length of processed text")
    strategies_used: List[str] = Field(
        default_factory=list, description="List of extraction strategies applied"
    )

    # Validation results
    validation_passed: bool = Field(
        default=True, description="Whether PydanticAI validation passed"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )

    @validator("entity_count")
    def validate_entity_count(cls, v, values):
        if "entities" in values and v != len(values["entities"]):
            raise ValueError("entity_count must match length of entities list")
        return v

    @validator("relationship_count")
    def validate_relationship_count(cls, v, values):
        if "relationships" in values and v != len(values["relationships"]):
            raise ValueError(
                "relationship_count must match length of relationships list"
            )
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI knowledge extraction operations"""
        return {
            "extraction_results": {
                "entity_count": self.entity_count,
                "relationship_count": self.relationship_count,
                "unique_entity_types": self.unique_entity_types,
                "unique_relationship_types": self.unique_relationship_types,
                "extraction_method": self.extraction_method,
                "strategies_used": self.strategies_used,
            },
            "quality_metrics": {
                "avg_entity_confidence": self.avg_entity_confidence,
                "avg_relationship_confidence": self.avg_relationship_confidence,
                "extraction_quality_score": self.extraction_quality_score,
                "validation_passed": self.validation_passed,
                "validation_warning_count": len(self.validation_warnings),
            },
            "graph_structure": {
                "entity_pairs_connected": self.entity_pairs_connected,
                "graph_density": self.graph_density,
                "connected_components": self.connected_components,
            },
            "processing_metadata": {
                "processing_time_ms": self.processing_time_ms,
                "text_length": self.text_length,
                "processing_efficiency": self.text_length
                / max(1, self.processing_time_ms)
                * 1000,  # chars per second
            },
        }


# =============================================================================
# 7.2. KNOWLEDGE EXTRACTION STRATEGY MODELS (from processors/unified_extraction_processor.py)
# =============================================================================


class EntityExtractionResult(BaseModel):
    """
    Result from entity extraction strategy

    Centralized from agents/knowledge_extraction/processors/unified_extraction_processor.py
    to eliminate scattered data model definitions.
    """

    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities with metadata"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )


class RelationshipExtractionResult(BaseModel):
    """
    Result from relationship extraction strategy

    Centralized from agents/knowledge_extraction/processors/unified_extraction_processor.py
    to eliminate scattered data model definitions.
    """

    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted relationships with metadata"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )


# =============================================================================
# 8. DOMAIN INTELLIGENCE ADDITIONAL MODELS (from analyzers/unified_content_analyzer.py)
# =============================================================================


class DomainAnalysisResult(PydanticAIContextualModel):
    """
    PydanticAI-enhanced domain analysis result following output validator patterns

    Replaces the previous UnifiedAnalysis dataclass with proper Pydantic validation
    and cross-agent compatibility.
    """

    # Core domain intelligence metrics
    # document_complexity removed - was over-engineered, use text_statistics for complexity analysis
    text_statistics: TextStatistics = Field(
        ..., description="Statistical text analysis"
    )
    cleaning_result: CleanedContent = Field(
        ..., description="Text preprocessing results"
    )

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
    analysis_confidence: ConfidenceScore = Field(
        ..., description="Overall analysis confidence"
    )
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

    @validator("domain_patterns")
    def validate_domain_patterns(cls, v):
        """Ensure domain patterns are meaningful"""
        if v:
            for pattern_type, patterns in v.items():
                if not isinstance(patterns, list) or len(patterns) == 0:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Empty pattern list for type: {pattern_type}")
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI domain analysis operations"""
        return {
            "domain_analysis": {
                "complexity_profile": {
                    "avg_words_per_sentence": self.document_complexity.avg_words_per_sentence,
                    "sentence_length_variance": self.document_complexity.sentence_length_variance,
                    "lexical_diversity": self.document_complexity.lexical_diversity,
                    "domain_sophistication": self.document_complexity.domain_sophistication,
                },
                "text_statistics": {
                    "total_words": self.text_statistics.total_words,
                    "unique_words": self.text_statistics.unique_words,
                    "total_sentences": self.text_statistics.total_sentences,
                    "avg_word_length": self.text_statistics.avg_word_length,
                },
                "domain_intelligence": {
                    "pattern_count": sum(
                        len(patterns) for patterns in self.domain_patterns.values()
                    ),
                    "technical_term_count": len(self.technical_vocabulary),
                    "concept_hierarchy_size": len(self.concept_hierarchy),
                    "domain_fit_score": self.domain_fit_score,
                },
            },
            "analysis_quality": {
                "confidence_score": self.analysis_confidence.value,
                "confidence_method": self.analysis_confidence.method,
                "processing_quality": self.processing_quality,
                "tfidf_feature_count": len(self.tfidf_features),
                "semantic_cluster_count": len(self.semantic_clusters),
            },
            "processing_metadata": {
                "processing_time_ms": self.processing_time_ms,
                "analysis_timestamp": self.analysis_timestamp,
                "source_file": self.source_file,
                "cleaning_applied": (
                    len(self.cleaning_result.preprocessing_steps)
                    if hasattr(self.cleaning_result, "preprocessing_steps")
                    else 0
                ),
            },
        }


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
        default=2, ge=1, description="Minimum frequency for pattern detection"
    )
    max_patterns_per_type: int = Field(
        default=20, ge=1, description="Maximum patterns per type"
    )

    # TF-IDF parameters
    tfidf_max_features: int = Field(
        default=1000, ge=100, description="Maximum TF-IDF features"
    )
    tfidf_min_df: int = Field(default=2, ge=1, description="Minimum document frequency")
    tfidf_max_df: float = Field(
        default=0.8, gt=0.0, lt=1.0, description="Maximum document frequency ratio"
    )

    # Clustering parameters
    n_clusters: int = Field(
        default=5, ge=2, le=20, description="Number of semantic clusters"
    )
    cluster_random_state: int = Field(
        default=42, description="Random state for clustering"
    )

    # Confidence calculation
    confidence_method: ConfidenceMethod = Field(
        default=ConfidenceMethod.ADAPTIVE, description="Confidence calculation method"
    )
    min_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI domain intelligence configuration"""
        return {
            "analysis_configuration": {
                "feature_flags": {
                    "advanced_analytics": self.enable_advanced_analytics,
                    "pattern_detection": self.enable_pattern_detection,
                    "quality_assessment": self.enable_quality_assessment,
                },
                "pattern_detection": {
                    "min_frequency": self.min_pattern_frequency,
                    "max_patterns_per_type": self.max_patterns_per_type,
                },
                "tfidf_settings": {
                    "max_features": self.tfidf_max_features,
                    "min_df": self.tfidf_min_df,
                    "max_df": self.tfidf_max_df,
                },
                "clustering": {
                    "n_clusters": self.n_clusters,
                    "random_state": self.cluster_random_state,
                },
            },
            "confidence_settings": {
                "method": (
                    self.confidence_method.value
                    if hasattr(self.confidence_method, "value")
                    else str(self.confidence_method)
                ),
                "min_threshold": self.min_confidence_threshold,
            },
        }


# =============================================================================
# 9. CORE SCATTERED MODELS (from agents/core/ scattered files)
# =============================================================================


class DynamicExtractionConfig(PydanticAIContextualModel):
    """Dynamically learned extraction configuration from Config-Extraction workflow"""

    entity_confidence_threshold: float = Field(
        ..., description="Dynamic entity confidence threshold"
    )
    relationship_confidence_threshold: float = Field(
        ..., description="Dynamic relationship confidence threshold"
    )
    chunk_size: int = Field(..., ge=100, le=5000, description="Dynamic chunk size")
    chunk_overlap: int = Field(..., ge=0, le=500, description="Dynamic chunk overlap")
    batch_size: int = Field(..., ge=1, le=100, description="Dynamic batch size")
    max_entities_per_chunk: int = Field(
        ..., ge=5, le=500, description="Dynamic max entities per chunk"
    )
    min_relationship_strength: float = Field(
        ..., ge=0.0, le=1.0, description="Dynamic minimum relationship strength"
    )
    quality_validation_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Dynamic quality validation threshold"
    )
    domain_name: str = Field(..., description="Domain this configuration applies to")
    learned_at: datetime = Field(..., description="When this configuration was learned")
    corpus_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Corpus statistics used for learning"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI extraction configuration"""
        return {
            "extraction_config": {
                "thresholds": {
                    "entity_confidence": self.entity_confidence_threshold,
                    "relationship_confidence": self.relationship_confidence_threshold,
                    "quality_validation": self.quality_validation_threshold,
                    "min_relationship_strength": self.min_relationship_strength,
                },
                "chunking": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "batch_size": self.batch_size,
                    "max_entities_per_chunk": self.max_entities_per_chunk,
                },
                "domain_context": {
                    "domain_name": self.domain_name,
                    "learned_at": self.learned_at.isoformat(),
                    "corpus_stats": self.corpus_stats,
                },
            }
        }


class DynamicSearchConfig(PydanticAIContextualModel):
    """Dynamically learned search configuration from domain analysis"""

    vector_similarity_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Dynamic vector similarity threshold"
    )
    vector_top_k: int = Field(
        ..., ge=1, le=100, description="Dynamic vector top-k results"
    )
    graph_hop_count: int = Field(..., ge=1, le=5, description="Dynamic graph hop count")
    graph_min_relationship_strength: float = Field(
        ..., ge=0.0, le=1.0, description="Dynamic graph minimum relationship strength"
    )
    gnn_prediction_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Dynamic GNN prediction confidence"
    )
    gnn_node_embeddings: int = Field(
        ..., ge=64, le=1024, description="Dynamic GNN node embeddings dimension"
    )
    tri_modal_weights: Dict[str, float] = Field(
        ..., description="Dynamic tri-modal weights (vector, graph, gnn)"
    )
    result_synthesis_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Dynamic result synthesis threshold"
    )
    domain_name: str = Field(..., description="Domain this configuration applies to")
    learned_at: datetime = Field(..., description="When this configuration was learned")
    query_complexity_weights: Dict[str, Any] = Field(
        default_factory=dict, description="Query complexity weighting factors"
    )

    @validator("tri_modal_weights")
    def validate_tri_modal_weights(cls, v):
        """Ensure tri-modal weights sum to 1.0"""
        if v:
            total = sum(v.values())
            if not (0.95 <= total <= 1.05):  # Allow small floating point errors
                raise ValueError(f"Tri-modal weights must sum to 1.0, got {total}")
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI search configuration"""
        return {
            "search_config": {
                "vector_settings": {
                    "similarity_threshold": self.vector_similarity_threshold,
                    "top_k": self.vector_top_k,
                },
                "graph_settings": {
                    "hop_count": self.graph_hop_count,
                    "min_relationship_strength": self.graph_min_relationship_strength,
                },
                "gnn_settings": {
                    "prediction_confidence": self.gnn_prediction_confidence,
                    "node_embeddings": self.gnn_node_embeddings,
                },
                "synthesis": {
                    "tri_modal_weights": self.tri_modal_weights,
                    "synthesis_threshold": self.result_synthesis_threshold,
                    "query_complexity_weights": self.query_complexity_weights,
                },
                "domain_context": {
                    "domain_name": self.domain_name,
                    "learned_at": self.learned_at.isoformat(),
                },
            }
        }


class ConsolidatedAzureServices(PydanticAIContextualModel):
    """
    Consolidated Azure services container combining LLM and non-LLM services.

    Consolidates functionality from:
    - azure_integration.py (Azure AI Foundry provider)
    - unified_azure_services.py (non-LLM services)

    Follows clean architecture with proper dependency injection support.
    """

    # Authentication (defaults handled in class)
    credential_initialized: bool = Field(
        default=False, description="Whether DefaultAzureCredential is initialized"
    )

    # Azure AI Foundry for LLM services
    ai_foundry_provider_initialized: bool = Field(
        default=False, description="Whether Azure AI Foundry provider is initialized"
    )

    # Non-LLM Azure services status
    search_client_initialized: bool = Field(
        default=False, description="Whether Azure Search client is initialized"
    )
    cosmos_client_initialized: bool = Field(
        default=False, description="Whether Cosmos client is initialized"
    )
    storage_client_initialized: bool = Field(
        default=False, description="Whether Storage client is initialized"
    )
    ml_client_initialized: bool = Field(
        default=False, description="Whether ML client is initialized"
    )

    # Search orchestration services
    tri_modal_orchestrator_initialized: bool = Field(
        default=False, description="Whether tri-modal orchestrator is initialized"
    )

    # Service status tracking
    initialized_services: Dict[str, bool] = Field(
        default_factory=dict, description="Service initialization status"
    )
    initialization_errors: Dict[str, str] = Field(
        default_factory=dict, description="Service initialization errors"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI Azure services operations"""
        return {
            "azure_services": {
                "authentication": {
                    "credential_initialized": self.credential_initialized
                },
                "llm_services": {
                    "ai_foundry_provider": self.ai_foundry_provider_initialized
                },
                "non_llm_services": {
                    "search_client": self.search_client_initialized,
                    "cosmos_client": self.cosmos_client_initialized,
                    "storage_client": self.storage_client_initialized,
                    "ml_client": self.ml_client_initialized,
                },
                "orchestration": {
                    "tri_modal_orchestrator": self.tri_modal_orchestrator_initialized
                },
                "status_summary": {
                    "initialized_count": sum(
                        1 for status in self.initialized_services.values() if status
                    ),
                    "total_services": len(self.initialized_services),
                    "error_count": len(self.initialization_errors),
                    "initialization_success_rate": (
                        sum(
                            1 for status in self.initialized_services.values() if status
                        )
                        / max(1, len(self.initialized_services))
                    ),
                },
            }
        }


# =============================================================================
# 10. WORKFLOW MODELS (from agents/workflows/ scattered files)
# =============================================================================


class StateTransferPacket(PydanticAIContextualModel):
    """Data packet for transferring state between workflow graphs"""

    transfer_id: str = Field(..., description="Unique transfer identifier")
    transfer_type: StateTransferType = Field(..., description="Type of state transfer")
    source_workflow: str = Field(..., description="Source workflow name")
    target_workflow: str = Field(..., description="Target workflow name")
    payload: Dict[str, Any] = Field(..., description="Transfer payload data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Transfer timestamp",
    )
    expiry: Optional[datetime] = Field(default=None, description="Transfer expiry time")
    dependencies: List[str] = Field(
        default_factory=list, description="Transfer dependencies"
    )
    version: str = Field(default="1.0", description="Transfer packet version")
    checksum: Optional[str] = Field(default=None, description="Data integrity checksum")

    @validator("expiry", pre=True, always=True)
    def set_default_expiry(cls, v, values):
        if v is None and "timestamp" in values:
            timestamp = values["timestamp"]
            return timestamp.replace(hour=23, minute=59, second=59)
        return v

    @validator("checksum", pre=True, always=True)
    def calculate_checksum(cls, v, values):
        if v is None and "payload" in values:
            import hashlib
            import json

            data_str = json.dumps(values["payload"], sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()
        return v

    def is_expired(self) -> bool:
        """Check if state transfer packet has expired"""
        return datetime.now(timezone.utc) > self.expiry

    def validate_integrity(self) -> bool:
        """Validate data integrity using checksum"""
        import hashlib
        import json

        data_str = json.dumps(self.payload, sort_keys=True, default=str)
        expected_checksum = hashlib.md5(data_str.encode()).hexdigest()
        return self.checksum == expected_checksum

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI state transfer operations"""
        return {
            "transfer_metadata": {
                "transfer_id": self.transfer_id,
                "transfer_type": (
                    self.transfer_type.value
                    if hasattr(self.transfer_type, "value")
                    else str(self.transfer_type)
                ),
                "source_workflow": self.source_workflow,
                "target_workflow": self.target_workflow,
                "version": self.version,
                "dependency_count": len(self.dependencies),
            },
            "transfer_timing": {
                "timestamp": self.timestamp.isoformat(),
                "expiry": self.expiry.isoformat() if self.expiry else None,
                "is_expired": self.is_expired(),
                "time_to_expiry": (
                    (self.expiry - datetime.now(timezone.utc)).total_seconds()
                    if self.expiry
                    else None
                ),
            },
            "data_integrity": {
                "checksum": self.checksum,
                "payload_size": len(str(self.payload)),
                "integrity_valid": self.validate_integrity(),
            },
        }


class GraphConnectionInfo(PydanticAIContextualModel):
    """Information about connections between workflow graphs"""

    source_graph: str = Field(..., description="Source graph name")
    target_graph: str = Field(..., description="Target graph name")
    connection_type: str = Field(..., description="Type of connection")
    data_flow_direction: str = Field(..., description="Data flow direction")
    last_transfer: Optional[datetime] = Field(
        default=None, description="Last successful transfer timestamp"
    )
    total_transfers: int = Field(
        default=0, ge=0, description="Total number of transfers"
    )
    failed_transfers: int = Field(
        default=0, ge=0, description="Number of failed transfers"
    )

    @property
    def success_rate(self) -> float:
        """Calculate transfer success rate"""
        if self.total_transfers == 0:
            return 1.0
        return (self.total_transfers - self.failed_transfers) / self.total_transfers

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI graph connection operations"""
        return {
            "connection_details": {
                "source_graph": self.source_graph,
                "target_graph": self.target_graph,
                "connection_type": self.connection_type,
                "data_flow_direction": self.data_flow_direction,
            },
            "transfer_statistics": {
                "total_transfers": self.total_transfers,
                "failed_transfers": self.failed_transfers,
                "successful_transfers": self.total_transfers - self.failed_transfers,
                "success_rate": self.success_rate,
                "has_transfers": self.total_transfers > 0,
            },
            "timing_info": {
                "last_transfer": (
                    self.last_transfer.isoformat() if self.last_transfer else None
                ),
                "time_since_last_transfer": (
                    (datetime.now(timezone.utc) - self.last_transfer).total_seconds()
                    if self.last_transfer
                    else None
                ),
            },
        }


# =============================================================================
# 11. ADDITIONAL DOMAIN INTELLIGENCE MODELS (from analyzers/config_generator.py)
# =============================================================================


class InfrastructureConfig(PydanticAIContextualModel):
    """Domain-specific Azure infrastructure configuration"""

    domain: str = Field(..., description="Domain name")
    search_index: str = Field(..., description="Azure Search index name")
    storage_container: str = Field(..., description="Storage container name")
    cosmos_graph: str = Field(..., description="Cosmos graph name")
    ml_endpoint: str = Field(..., description="ML endpoint URL")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Configuration confidence"
    )
    primary_concepts: List[str] = Field(
        default_factory=list, description="Primary domain concepts"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI infrastructure configuration"""
        return {
            "infrastructure_config": {
                "domain": self.domain,
                "azure_services": {
                    "search_index": self.search_index,
                    "storage_container": self.storage_container,
                    "cosmos_graph": self.cosmos_graph,
                    "ml_endpoint": self.ml_endpoint,
                },
                "domain_intelligence": {
                    "confidence": self.confidence,
                    "primary_concepts": self.primary_concepts,
                    "concept_count": len(self.primary_concepts),
                },
            }
        }


class MLModelConfig(PydanticAIContextualModel):
    """Domain-specific ML model configuration"""

    domain: str = Field(..., description="Domain name")
    node_feature_dim: int = Field(
        ..., ge=1, le=1024, description="Node feature dimension"
    )
    hidden_dim: int = Field(..., ge=1, le=1024, description="Hidden layer dimension")
    num_layers: int = Field(..., ge=1, le=10, description="Number of layers")
    learning_rate: float = Field(..., gt=0.0, lt=1.0, description="Learning rate")
    entity_types: List[str] = Field(
        default_factory=list, description="Supported entity types"
    )
    relationship_types: List[str] = Field(
        default_factory=list, description="Supported relationship types"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model configuration confidence"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI ML model configuration"""
        return {
            "ml_model_config": {
                "domain": self.domain,
                "model_architecture": {
                    "node_feature_dim": self.node_feature_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "learning_rate": self.learning_rate,
                },
                "domain_knowledge": {
                    "entity_types": self.entity_types,
                    "relationship_types": self.relationship_types,
                    "entity_type_count": len(self.entity_types),
                    "relationship_type_count": len(self.relationship_types),
                },
                "quality_metrics": {"confidence": self.confidence},
            }
        }


class CompleteDomainConfig(PydanticAIContextualModel):
    """Complete domain configuration combining infrastructure, ML, and patterns"""

    domain: str = Field(..., description="Domain name")
    infrastructure: InfrastructureConfig = Field(
        ..., description="Infrastructure configuration"
    )
    ml_model: MLModelConfig = Field(..., description="ML model configuration")
    patterns: ExtractedPatterns = Field(..., description="Extracted domain patterns")
    generation_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall generation confidence"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI complete domain configuration"""
        return {
            "complete_domain_config": {
                "domain": self.domain,
                "generation_confidence": self.generation_confidence,
                "infrastructure": self.infrastructure.run_context_data[
                    "infrastructure_config"
                ],
                "ml_model": self.ml_model.run_context_data["ml_model_config"],
                "patterns": {
                    "pattern_count": (
                        len(self.patterns.patterns)
                        if hasattr(self.patterns, "patterns")
                        else 0
                    ),
                    "statistical_count": (
                        len(self.patterns.statistical)
                        if hasattr(self.patterns, "statistical")
                        else 0
                    ),
                    "semantic_count": (
                        len(self.patterns.semantic)
                        if hasattr(self.patterns, "semantic")
                        else 0
                    ),
                },
            }
        }


# =============================================================================
# 12. UNIVERSAL SEARCH MODELS (from orchestrators/consolidated_search_orchestrator.py)
# =============================================================================


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
# 13. PYDANTIC AI AGENT OUTPUT MODELS
# =============================================================================


class DomainAnalysisOutput(PydanticAIContextualModel):
    """Structured output model for Domain Intelligence Agent"""

    domain_classification: str = Field(
        ..., description="Detected domain classification"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    generated_config: Dict[str, Any] = Field(
        default_factory=dict, description="Generated domain configuration"
    )
    processing_recommendations: List[str] = Field(
        default_factory=list, description="Processing recommendations"
    )
    domain_patterns: List[str] = Field(
        default_factory=list, description="Identified domain patterns"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis metadata"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI domain analysis operations"""
        return {
            "domain_analysis": {
                "classification": {
                    "domain": self.domain_classification,
                    "confidence": self.confidence_score,
                    "patterns_found": len(self.domain_patterns),
                },
                "configuration": {
                    "config_keys": list(self.generated_config.keys()),
                    "recommendations_count": len(self.processing_recommendations),
                },
                "metadata": self.metadata,
            }
        }


class KnowledgeExtractionOutput(PydanticAIContextualModel):
    """Structured output model for Knowledge Extraction Agent"""

    entities: List[ValidatedEntity] = Field(
        default_factory=list, description="Extracted and validated entities"
    )
    relationships: List[ValidatedRelationship] = Field(
        default_factory=list, description="Extracted and validated relationships"
    )
    extraction_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall extraction confidence"
    )
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Processing statistics"
    )
    validation_results: Dict[str, Any] = Field(
        default_factory=dict, description="Validation results"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extraction metadata"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI knowledge extraction operations"""
        return {
            "knowledge_extraction": {
                "results": {
                    "entities_count": len(self.entities),
                    "relationships_count": len(self.relationships),
                    "extraction_confidence": self.extraction_confidence,
                },
                "processing": self.processing_stats,
                "validation": self.validation_results,
                "metadata": self.metadata,
            }
        }


class UniversalSearchOutput(PydanticAIContextualModel):
    """Structured output model for Universal Search Agent"""

    search_results: List[SearchResult] = Field(
        default_factory=list, description="Tri-modal search results"
    )
    modality_results: Dict[str, ModalityResult] = Field(
        default_factory=dict, description="Results from individual modalities"
    )
    synthesis_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Result synthesis confidence"
    )
    search_strategy: Dict[str, Any] = Field(
        default_factory=dict, description="Applied search strategy"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Search metadata"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI universal search operations"""
        return {
            "universal_search": {
                "results": {
                    "total_results": len(self.search_results),
                    "modalities_used": list(self.modality_results.keys()),
                    "synthesis_confidence": self.synthesis_confidence,
                },
                "strategy": self.search_strategy,
                "performance": self.performance_metrics,
                "metadata": self.metadata,
            }
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HealthStatus",
    "ProcessingStatus",
    "WorkflowState",
    "NodeState",
    "ErrorSeverity",
    "ErrorCategory",
    "SearchType",
    "MessageType",
    "ConfidenceMethod",
    "StateTransferType",
    # "ModelCapability", "QueryComplexity",  # Removed - using static model selection
    # Base Models
    "BaseRequest",
    "BaseResponse",
    "BaseAnalysisResult",
    # Unified Configuration Architecture
    "UnifiedAgentConfiguration",  # UnifiedConfigurationResolver deleted
    # Enhanced Azure Service Models
    "AzureServiceConfiguration",
    "AzureServiceMetrics",  # AzureServiceHealthCheck deleted in Phase 10
    "AzureMLModelMetadata",
    "AzureSearchIndexSchema",
    "AzureCosmosGraphSchema",
    # Performance Feedback Integration Models
    "PerformanceFeedbackPoint",  # PerformanceFeedbackCollector deleted
    # Agent Contract Models
    "StatisticalPattern",
    "DomainStatistics",
    "DomainAnalysisContract",
    "KnowledgeExtractionContract",
    "UniversalSearchContract",
    # Configuration Models
    "SynthesisWeights",
    "DomainConfig",
    "ExtractionConfiguration",
    "VectorSearchConfig",
    "GraphSearchConfig",
    "GNNSearchConfig",
    # Request Models
    "QueryRequest",
    "VectorSearchRequest",
    "GraphSearchRequest",
    "TriModalSearchRequest",
    "DomainDetectionRequest",
    "PatternLearningRequest",
    # Response Models
    "SearchResult",
    "SearchResponse",
    "DomainDetectionResult",
    "AnalysisResult",
    # Workflow Models
    "WorkflowExecutionState",
    "NodeExecutionResult",
    "WorkflowResultContract",
    # Analysis Models
    "StatisticalAnalysis",
    "ExtractedKnowledge",
    "ExtractionResults",
    "UnifiedExtractionResult",
    "TriModalSearchResult",
    # Validation Models
    "ValidationResult",
    "QualityMetrics",
    "ErrorContext",
    "ErrorHandlingContract",
    # Dependency Models (SharedDeps deleted in Phase 4)
    "DomainIntelligenceDeps",
    "KnowledgeExtractionDeps",
    "UniversalSearchDeps",
    # Communication Models
    # GraphMessage, GraphStatus - deleted (export-only, no actual usage)
    # Cache & Monitoring Models
    # MonitoringContract deleted in Phase 9 - over-engineered monitoring config with zero usage
    # Specialized Models
    "CacheEntry",
    "CacheMetrics",
    "ServiceHealth",
    # Architecture Models (DataDrivenExtractionConfiguration, ArchitectureComplianceValidator deleted in Phase 3)
    # Tier 2: Consolidated Service Models
    "ConsolidatedAzureConfiguration",
    "ConsolidatedExtractionConfiguration",
    "ConsolidatedSearchConfiguration",
    # Tier 3: Enhanced Agent Contracts (DELETED - redundant with basic contracts)
    # Foundation Models
    "ConfigurationResolver",
    "PydanticAIContextualModel",
    # Factory Functions (deleted in Phase 4 - were unused)
    # "create_base_request", "create_validation_result", "create_error_context",
    # Additional Core Module Models
    "ErrorContext",
    "ErrorMetrics",
    "CacheEntry",
    "CachePerformanceMetrics",
    "MemoryStatus",  # ModelSelectionCriteria deleted in Phase 5
    # DomainConfigMetadata, BackgroundProcessingMetadata deleted in Phase 5 - were dead exports
    "CacheKeyMapping",
    "WorkflowStateBridge",
    "ServiceContainerConfig",
    # Domain Intelligence Analyzer Models
    "UnifiedAnalysis",
    "BackgroundProcessingConfig",
    "BackgroundProcessingResult",  # ContentQuality deleted in Phase 9
    "LLMExtraction",
    "ProcessingStats",
    "DomainSignature",
    "DomainConfig",
    "LearnedPattern",
    "ExtractedPatterns",  # PatternStatistics deleted
    "DomainGenerationConfig",
    "DomainPatternResult",
    "DomainConfigResult",
    # Dynamic Manager Models (RuntimeConfigurationData deleted in Phase 4, ModelSelectionCriteria deleted in Phase 5)
    # Search Orchestrator Models
    "SearchCoordinationResult",
    "TriModalResult",
    "TriModalSearchResult",
    # PydanticAI Output Validator Models
    "ExtractionQualityOutput",
    "ValidatedEntity",
    "ValidatedRelationship",
    "ContentAnalysisOutput",
    "DomainConfigurationOutput",
    # Knowledge Extraction Models
    "ValidationResult",
    "KnowledgeExtractionResult",
    "EntityExtractionResult",
    "RelationshipExtractionResult",
    # Domain Intelligence Additional Models
    "DomainAnalysisResult",
    "DomainIntelligenceConfig",
    # Core Scattered Models
    "DynamicExtractionConfig",
    "DynamicSearchConfig",
    "ConsolidatedAzureServices",
    # Workflow Models
    "StateTransferPacket",
    "GraphConnectionInfo",
    # Additional Domain Intelligence Models
    "InfrastructureConfig",
    "MLModelConfig",
    "CompleteDomainConfig",
    # Universal Search Models
    "ModalityResult",
    # PydanticAI Agent Output Models
    "DomainAnalysisOutput",
    "KnowledgeExtractionOutput",
    "UniversalSearchOutput",
    # PydanticAI Output Validator Functions
    "validate_extraction_quality",
    "validate_entity_extraction",
    "validate_relationship_extraction",
    "validate_content_analysis",
    "validate_domain_configuration",
]
