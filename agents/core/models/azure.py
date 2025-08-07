"""
Azure Service Models and Configurations
======================================

Azure-specific data models for service configuration, authentication,
health monitoring, and resource management across all Azure services
used by the Universal RAG system.

This module provides:
- Azure service configuration with DefaultAzureCredential support
- Performance and health metrics for Azure services
- ML model metadata and deployment tracking
- Cognitive Search index schema definitions
- Cosmos DB graph schema models
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from agents.core.constants import (
    InfrastructureConstants,
    MathematicalFoundationConstants,
    SystemPerformanceConstants,
)

from .base import HealthStatus, PydanticAIContextualModel

# =============================================================================
# AZURE SERVICE CONFIGURATION
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
        default=SystemPerformanceConstants.DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=SystemPerformanceConstants.DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )
    retry_backoff_factor: float = Field(
        default=MathematicalFoundationConstants.EXPONENTIAL_BACKOFF_BASE,
        ge=1.0,
        le=10.0,
        description="Exponential backoff factor",
    )

    # Health check configuration
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval_seconds: int = Field(
        default=SystemPerformanceConstants.HEALTH_CHECK_INTERVAL_SECONDS,
        ge=5,
        le=300,
        description="Health check interval",
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


# =============================================================================
# AZURE SERVICE METRICS
# =============================================================================


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
        default=MathematicalFoundationConstants.PERFECT_SCORE,
        ge=0.0,
        le=1.0,
        description="Authentication success rate",
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
        default=SystemPerformanceConstants.METRICS_WINDOW_MINUTES,
        ge=1,
        le=60,
        description="Measurement window",
    )

    def calculate_sla_compliance(
        self,
        target_availability: float = SystemPerformanceConstants.DEFAULT_SLA_AVAILABILITY_PERCENT,
    ) -> bool:
        """Check if service meets SLA availability target"""
        return self.availability_percentage >= target_availability

    def is_performance_degraded(
        self,
        response_time_threshold_ms: float = SystemPerformanceConstants.MAX_RESPONSE_TIME_MS,
        error_rate_threshold: float = SystemPerformanceConstants.MAX_ERROR_RATE,
    ) -> bool:
        """Check if service performance is degraded"""
        return (
            self.response_time_ms > response_time_threshold_ms
            or self.error_rate > error_rate_threshold
        )


# =============================================================================
# AZURE ML MODELS
# =============================================================================


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


class MLModelConfig(PydanticAIContextualModel):
    """Azure ML model configuration for GNN and other ML models"""

    model_name: str = Field(description="Model name in Azure ML")
    model_version: str = Field(description="Model version")
    endpoint_url: str = Field(description="Model serving endpoint")
    deployment_config: Dict[str, Any] = Field(description="Deployment configuration")
    input_schema: Dict[str, Any] = Field(description="Expected input schema")
    output_schema: Dict[str, Any] = Field(description="Expected output schema")
    performance_requirements: Dict[str, float] = Field(
        description="Performance SLA requirements"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "endpoint_url": self.endpoint_url,
            "deployment_config": self.deployment_config,
        }


# =============================================================================
# AZURE SEARCH MODELS
# =============================================================================


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


# =============================================================================
# AZURE COSMOS DB MODELS
# =============================================================================


class AzureCosmosGraphSchema(BaseModel):
    """Azure Cosmos DB graph database schema"""

    database_name: str = Field(description="Cosmos database name")
    container_name: str = Field(description="Graph container name")
    vertex_types: List[str] = Field(description="Defined vertex types")
    edge_types: List[str] = Field(description="Defined edge types")
    partition_key: str = Field(description="Partition key property")
    vertex_count: int = Field(ge=0, description="Total vertex count")
    edge_count: int = Field(ge=0, description="Total edge count")
    throughput_ru: int = Field(
        ge=InfrastructureConstants.MIN_COSMOS_THROUGHPUT_RU,
        description="Provisioned throughput RU/s",
    )


class GraphConnectionInfo(PydanticAIContextualModel):
    """Azure Cosmos DB graph connection information"""

    endpoint: str = Field(description="Cosmos DB endpoint URL")
    database_name: str = Field(description="Database name")
    container_name: str = Field(description="Graph container name")
    partition_key: str = Field(description="Partition key for queries")
    max_retry_attempts: int = Field(
        default=SystemPerformanceConstants.DEFAULT_MAX_RETRIES,
        description="Maximum retry attempts",
    )
    timeout_seconds: int = Field(
        default=SystemPerformanceConstants.DEFAULT_TIMEOUT_SECONDS,
        description="Request timeout",
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "database_name": self.database_name,
            "container_name": self.container_name,
            "partition_key": self.partition_key,
        }


# =============================================================================
# CONSOLIDATED AZURE SERVICE MODELS
# =============================================================================


class ConsolidatedAzureConfiguration(PydanticAIContextualModel):
    """Consolidated configuration for all Azure services used by the system"""

    # Azure OpenAI Configuration
    openai_endpoint: str = Field(description="Azure OpenAI endpoint")
    openai_deployment: str = Field(description="GPT model deployment name")
    openai_api_version: str = Field(description="OpenAI API version")

    # Azure Cognitive Search Configuration
    search_endpoint: str = Field(description="Azure Search endpoint")
    search_index: str = Field(description="Search index name")
    search_api_version: str = Field(description="Search API version")

    # Azure Cosmos DB Configuration
    cosmos_endpoint: str = Field(description="Cosmos DB endpoint")
    cosmos_database: str = Field(description="Cosmos database name")
    cosmos_container: str = Field(description="Cosmos container name")

    # Azure ML Configuration
    ml_endpoint: Optional[str] = Field(default=None, description="Azure ML endpoint")
    ml_deployment: Optional[str] = Field(
        default=None, description="ML model deployment"
    )

    # Authentication and Resource Management
    subscription_id: str = Field(description="Azure subscription ID")
    resource_group: str = Field(description="Resource group name")
    use_managed_identity: bool = Field(default=True, description="Use managed identity")

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "azure_openai": {
                "endpoint": self.openai_endpoint,
                "deployment": self.openai_deployment,
                "api_version": self.openai_api_version,
            },
            "azure_search": {
                "endpoint": self.search_endpoint,
                "index": self.search_index,
                "api_version": self.search_api_version,
            },
            "azure_cosmos": {
                "endpoint": self.cosmos_endpoint,
                "database": self.cosmos_database,
                "container": self.cosmos_container,
            },
            "resource_management": {
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "use_managed_identity": self.use_managed_identity,
            },
        }


class ConsolidatedAzureServices(PydanticAIContextualModel):
    """Consolidated Azure service dependencies for agent operations"""

    # Service configurations
    openai_config: AzureServiceConfiguration = Field(
        description="Azure OpenAI configuration"
    )
    search_config: AzureServiceConfiguration = Field(
        description="Azure Search configuration"
    )
    cosmos_config: AzureServiceConfiguration = Field(
        description="Azure Cosmos DB configuration"
    )
    ml_config: Optional[AzureServiceConfiguration] = Field(
        default=None, description="Azure ML configuration"
    )

    # Service health and performance
    service_metrics: Dict[str, AzureServiceMetrics] = Field(
        default_factory=dict, description="Service performance metrics"
    )
    overall_health: HealthStatus = Field(description="Overall system health")

    # Configuration metadata
    configuration_version: str = Field(description="Configuration version")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last configuration update"
    )
    environment: str = Field(description="Deployment environment")

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "azure_services": {
                "openai": self.openai_config.run_context_data,
                "search": self.search_config.run_context_data,
                "cosmos": self.cosmos_config.run_context_data,
                "ml": self.ml_config.run_context_data if self.ml_config else None,
            },
            "system_health": self.overall_health.value,
            "environment": self.environment,
            "configuration_version": self.configuration_version,
        }


# =============================================================================
# INFRASTRUCTURE CONFIGURATION
# =============================================================================


class InfrastructureConfig(PydanticAIContextualModel):
    """Infrastructure configuration for Azure services"""

    subscription_id: str = Field(description="Azure subscription ID")
    resource_group: str = Field(description="Resource group name")
    location: str = Field(description="Azure region")
    environment: str = Field(description="Environment (dev, staging, prod)")
    tags: Dict[str, str] = Field(default_factory=dict, description="Resource tags")

    # Service endpoints
    service_endpoints: Dict[str, str] = Field(description="Service endpoint mappings")

    # Authentication configuration
    managed_identity_client_id: Optional[str] = Field(
        default=None, description="Managed identity client ID"
    )
    key_vault_url: Optional[str] = Field(
        default=None, description="Key vault URL for secrets"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "infrastructure": {
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "location": self.location,
                "environment": self.environment,
            },
            "service_endpoints": self.service_endpoints,
            "authentication": {
                "managed_identity_client_id": self.managed_identity_client_id,
                "key_vault_url": self.key_vault_url,
            },
        }
