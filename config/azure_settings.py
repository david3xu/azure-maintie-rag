"""
Configuration management for Azure Universal RAG
Centralizes all application settings and environment variables for Azure services
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_azd_environment() -> str:
    """Auto-detect current azd environment"""
    try:
        result = subprocess.run(
            ["azd", "env", "get-values"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("AZURE_ENV_NAME="):
                    return line.split("=")[1].strip('"')
    except Exception as e:
        # Log the environment detection failure but continue with safe fallback
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Environment detection via azd failed: {e}. Using development fallback."
        )
        pass
    return "dynamic_environment"  # universal fallback


class Settings(BaseSettings):
    # Azure Search Enterprise Extension
    azure_search_index_prefix: str = Field(
        default="universal-rag", env="AZURE_SEARCH_INDEX_PREFIX"
    )
    azure_search_batch_size: int = Field(default=100, env="AZURE_SEARCH_BATCH_SIZE")

    # Azure ML Enterprise Extension
    azure_ml_endpoint_prefix: str = Field(
        default="gnn-model", env="AZURE_ML_ENDPOINT_PREFIX"
    )
    azure_ml_deployment_name: str = Field(
        default="default", env="AZURE_ML_DEPLOYMENT_NAME"
    )
    azure_ml_inference_timeout: int = Field(
        default=300, env="AZURE_ML_INFERENCE_TIMEOUT"
    )
    gnn_model_version: str = Field(default="latest", env="GNN_MODEL_VERSION")

    # Embedding Update Extension
    embedding_batch_size: int = Field(default=50, env="EMBEDDING_BATCH_SIZE")
    embedding_update_interval_hours: int = Field(
        default=24, env="EMBEDDING_UPDATE_INTERVAL_HOURS"
    )
    """Unified application configuration settings for Azure services - single source of truth"""

    # --- Azure Data Processing Policy Configuration ---
    skip_processing_if_data_exists: bool = Field(
        default=False,
        env="SKIP_PROCESSING_IF_DATA_EXISTS",
        description="Skip data preparation if Azure services already contain data",
    )
    force_data_reprocessing: bool = Field(
        default=False,
        env="FORCE_DATA_REPROCESSING",
        description="Force data reprocessing even if Azure services contain data",
    )
    data_state_validation_enabled: bool = Field(
        default=True,
        env="DATA_STATE_VALIDATION_ENABLED",
        description="Enable Azure data state validation before processing",
    )
    azure_data_state_cache_ttl: int = Field(
        default=300,
        env="AZURE_DATA_STATE_CACHE_TTL",
        description="Azure data state cache TTL in seconds",
    )
    azure_strict_error_handling: bool = Field(
        default=True,
        env="AZURE_STRICT_ERROR_HANDLING",
        description="Enable strict error handling across all Azure services",
    )

    # Application Settings
    app_name: str = "Azure Universal RAG"
    app_version: str = "2.0.0"
    environment: str = Field(default_factory=get_azd_environment, env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")

    # Azure Identity & Security (azd-compatible)
    azure_client_id: str = Field(default="", env="AZURE_CLIENT_ID")  # azd output
    use_managed_identity: bool = Field(default=True, env="USE_MANAGED_IDENTITY")
    cosmos_use_managed_identity: bool = Field(
        default=True, env="COSMOS_USE_MANAGED_IDENTITY"
    )
    azure_key_vault_name: str = Field(
        default="", env="AZURE_KEY_VAULT_NAME"
    )  # azd output

    # Monitoring (azd-compatible)
    applicationinsights_connection_string: str = Field(
        default="", env="APPLICATIONINSIGHTS_CONNECTION_STRING"
    )  # azd output

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # Azure OpenAI Settings (azd-managed, no keys needed)
    openai_api_type: str = "azure"
    azure_openai_endpoint: str = Field(
        default="", env="AZURE_OPENAI_ENDPOINT"
    )  # azd output
    openai_api_version: str = Field(default="2024-06-01", env="OPENAI_API_VERSION")
    openai_deployment_name: str = Field(
        default="gpt-4.1-mini", env="AZURE_OPENAI_DEPLOYMENT_NAME"
    )  # azd output
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")

    # Backward compatibility properties
    @property
    def openai_api_base(self) -> str:
        return self.azure_openai_endpoint

    @property
    def openai_api_key(self) -> str:
        # For managed identity, return empty - will be handled by Azure SDK
        return ""

    # Azure Embedding Settings (azd-managed, same endpoint as OpenAI)
    embedding_model: str = Field(
        default="text-embedding-ada-002", env="EMBEDDING_MODEL"
    )
    embedding_deployment_name: str = Field(
        default="text-embedding-ada-002", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    )  # azd output
    embedding_api_version: str = Field(
        default="2024-06-01", env="EMBEDDING_API_VERSION"
    )
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # Azure Storage Settings (azd-managed with managed identity)
    azure_storage_account: str = Field(
        default="", env="AZURE_STORAGE_ACCOUNT"
    )  # azd output
    azure_blob_container: str = Field(
        default="maintie-prod-data", env="AZURE_STORAGE_CONTAINER"
    )  # azd output
    azure_storage_container: str = Field(
        default="maintie-prod-data", env="AZURE_STORAGE_CONTAINER"
    )  # alias

    # Backward compatibility properties
    @property
    def azure_storage_connection_string(self) -> str:
        # For managed identity, return empty - will be handled by Azure SDK
        return ""

    # Azure Cognitive Search Settings (azd-managed with managed identity)
    azure_search_endpoint: str = Field(
        default="", env="AZURE_SEARCH_ENDPOINT"
    )  # azd output
    azure_search_api_version: str = Field(
        default="2023-11-01", env="AZURE_SEARCH_API_VERSION"
    )
    azure_search_index: str = Field(
        default="maintie-prod-index", env="AZURE_SEARCH_INDEX"
    )  # azd output

    # Backward compatibility properties
    @property
    def azure_search_key(self) -> str:
        # For managed identity, return empty - will be handled by Azure SDK
        return ""

    # Azure Cosmos DB Settings (azd-managed with managed identity)
    azure_cosmos_endpoint: str = Field(
        default="", env="AZURE_COSMOS_ENDPOINT"
    )  # azd output
    cosmos_database_name: str = Field(
        default="maintie-rag-prod", env="COSMOS_DATABASE_NAME"
    )  # azd output
    cosmos_graph_name: str = Field(
        default="knowledge-graph-prod", env="COSMOS_GRAPH_NAME"
    )  # azd output
    azure_cosmos_container: str = Field(
        default="", env="COSMOS_CONTAINER_NAME"
    )  # azd output
    azure_cosmos_api_version: str = Field(
        default="2023-03-01-preview", env="AZURE_COSMOS_API_VERSION"
    )

    # Backward compatibility properties
    # Cosmos DB Authentication - hybrid approach
    azure_cosmos_key: str = Field(default="", env="AZURE_COSMOS_KEY")

    @property
    def azure_cosmos_database(self) -> str:
        return self.cosmos_database_name

    # Azure ML Settings (azd-managed with managed identity)
    azure_subscription_id: str = Field(
        default="", env="AZURE_SUBSCRIPTION_ID"
    )  # azd output
    azure_resource_group: str = Field(
        default="", env="AZURE_RESOURCE_GROUP"
    )  # azd output
    azure_ml_workspace_name: str = Field(
        default="", env="AZURE_ML_WORKSPACE_NAME"
    )  # azd output
    azure_ml_api_version: str = Field(default="2023-04-01", env="AZURE_ML_API_VERSION")
    azure_tenant_id: str = Field(default="", env="AZURE_TENANT_ID")  # azd output

    # Backward compatibility properties
    @property
    def azure_ml_workspace(self) -> str:
        return self.azure_ml_workspace_name

    # Optional ML endpoints (can be empty for graceful degradation)
    azure_ml_confidence_endpoint: str = Field(
        default="", env="AZURE_ML_CONFIDENCE_ENDPOINT"
    )
    azure_ml_completeness_endpoint: str = Field(
        default="", env="AZURE_ML_COMPLETENESS_ENDPOINT"
    )

    # Knowledge Extraction Configuration
    extraction_quality_level: str = Field(
        default="adaptive", env="EXTRACTION_QUALITY_LEVEL"
    )
    extraction_confidence_threshold: float = Field(
        default=0.7, env="EXTRACTION_CONFIDENCE_THRESHOLD"
    )
    max_entities_per_document: int = Field(default=100, env="MAX_ENTITIES_PER_DOCUMENT")
    extraction_batch_size: int = Field(default=50, env="EXTRACTION_BATCH_SIZE")
    enable_text_analytics_preprocessing: bool = Field(
        default=False, env="ENABLE_TEXT_ANALYTICS_PREPROCESSING"
    )

    # Azure OpenAI Rate Limiting
    azure_openai_max_tokens_per_minute: int = Field(
        default=40000, env="AZURE_OPENAI_MAX_TOKENS_PER_MINUTE"
    )
    azure_openai_max_requests_per_minute: int = Field(
        default=60, env="AZURE_OPENAI_MAX_REQUESTS_PER_MINUTE"
    )

    # Azure Prompt Flow Integration
    enable_prompt_flow: bool = Field(default=False, env="ENABLE_PROMPT_FLOW")
    prompt_flow_fallback_enabled: bool = Field(
        default=True, env="PROMPT_FLOW_FALLBACK_ENABLED"
    )
    enable_prompt_flow_monitoring: bool = Field(
        default=True, env="ENABLE_PROMPT_FLOW_MONITORING"
    )
    azure_openai_cost_threshold_per_hour: float = Field(
        default=50.0, env="AZURE_OPENAI_COST_THRESHOLD_PER_HOUR"
    )
    azure_openai_priority_tier: str = Field(
        default="adaptive", env="AZURE_OPENAI_PRIORITY_TIER"
    )

    # Azure Key Vault Settings - Enterprise Security Enhancement
    azure_key_vault_url: str = Field(default="", env="AZURE_KEY_VAULT_URL")
    azure_use_managed_identity: bool = Field(
        default=True, env="AZURE_USE_MANAGED_IDENTITY"
    )
    azure_managed_identity_client_id: str = Field(
        default="", env="AZURE_MANAGED_IDENTITY_CLIENT_ID"
    )

    # Azure Application Insights Settings - Enterprise Monitoring
    azure_application_insights_connection_string: str = Field(
        default="", env="AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING"
    )
    azure_enable_telemetry: bool = Field(default=True, env="AZURE_ENABLE_TELEMETRY")

    # Azure Container Apps Settings - Deployment Infrastructure
    azure_container_app_name: str = Field(default="", env="AZURE_CONTAINER_APP_NAME")
    azure_container_apps_environment: str = Field(
        default="", env="AZURE_CONTAINER_APPS_ENVIRONMENT"
    )

    # Azure Container Registry Settings - Image Management
    azure_container_registry: str = Field(default="", env="AZURE_CONTAINER_REGISTRY")
    azure_container_registry_endpoint: str = Field(
        default="", env="AZURE_CONTAINER_REGISTRY_ENDPOINT"
    )

    # Azure Resource Naming Convention
    azure_resource_prefix: str = Field(default="maintie", env="AZURE_RESOURCE_PREFIX")
    azure_environment: str = Field(default="dev", env="AZURE_ENVIRONMENT")
    azure_region: str = Field(default="eastus", env="AZURE_REGION")

    # Data Paths (for local development)
    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent
    data_dir: Path = Field(default=BASE_DIR / "data", env="DATA_DIR")
    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw", env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(
        default=BASE_DIR / "data" / "processed", env="PROCESSED_DATA_DIR"
    )
    indices_dir: Path = Field(default=BASE_DIR / "data" / "indices", env="INDICES_DIR")
    config_dir: Path = Field(default=BASE_DIR / "config", env="CONFIG_DIR")

    # Azure RAG Configuration
    discovery_sample_size: int = Field(default=200, env="DISCOVERY_SAMPLE_SIZE")
    pattern_confidence_threshold: float = Field(
        default=0.7, env="PATTERN_CONFIDENCE_THRESHOLD"
    )
    discovery_min_confidence: float = Field(default=0.6, env="DISCOVERY_MIN_CONFIDENCE")
    discovery_max_patterns: int = Field(default=50, env="DISCOVERY_MAX_PATTERNS")

    # Query Analysis Settings
    max_related_entities: int = Field(default=15, env="MAX_RELATED_ENTITIES")
    max_neighbors: int = Field(default=5, env="MAX_NEIGHBORS")
    concept_expansion_limit: int = Field(default=10, env="CONCEPT_EXPANSION_LIMIT")

    # Azure Search Settings
    vector_search_top_k: int = Field(default=10, env="VECTOR_SEARCH_TOP_K")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Azure OpenAI Generation Settings
    openai_max_tokens: int = Field(default=500, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, env="LLM_TOP_P")
    llm_frequency_penalty: float = Field(default=0.1, env="LLM_FREQUENCY_PENALTY")
    llm_presence_penalty: float = Field(default=0.1, env="LLM_PRESENCE_PENALTY")

    # API Validation Settings
    query_min_length: int = Field(default=3, env="QUERY_MIN_LENGTH")
    query_max_length: int = Field(default=500, env="QUERY_MAX_LENGTH")
    max_results_limit: int = Field(default=50, env="MAX_RESULTS_LIMIT")

    # Performance Settings
    max_query_time: float = Field(default=2.0, env="MAX_QUERY_TIME")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )

    # Azure Discovery Settings
    discovery_enable_ner: bool = Field(default=True, env="DISCOVERY_ENABLE_NER")
    discovery_enable_relations: bool = Field(
        default=True, env="DISCOVERY_ENABLE_RELATIONS"
    )

    # Trusted Hosts for Security
    trusted_hosts: Optional[str] = Field(
        default="localhost,127.0.0.1", env="TRUSTED_HOSTS"
    )

    @property
    def trusted_hosts_list(self) -> List[str]:
        """Convert trusted_hosts string to list"""
        if isinstance(self.trusted_hosts, str):
            return [host.strip() for host in self.trusted_hosts.split(",")]
        return []

    # GNN Training Pipeline Configuration
    azure_ml_compute_cluster: str = Field(
        default="gnn-cluster", env="AZURE_ML_COMPUTE_CLUSTER"
    )
    gnn_training_trigger_threshold: int = Field(
        default=100, env="GNN_TRAINING_TRIGGER_THRESHOLD"
    )
    gnn_model_deployment_tier: str = Field(
        default="adaptive", env="GNN_MODEL_DEPLOYMENT_TIER"
    )
    enable_incremental_gnn_training: bool = Field(
        default=True, env="ENABLE_INCREMENTAL_GNN_TRAINING"
    )

    # Graph Embedding Configuration
    graph_embedding_dimension: int = Field(default=128, env="GRAPH_EMBEDDING_DIMENSION")
    graph_embedding_update_frequency: str = Field(
        default="daily", env="GRAPH_EMBEDDING_UPDATE_FREQUENCY"
    )

    # Cosmos DB Gremlin Configuration
    cosmos_db_database_name: str = Field(
        default="universal-rag-db", env="COSMOS_DB_DATABASE_NAME"
    )
    cosmos_db_container_name: str = Field(
        default="knowledge-graph", env="COSMOS_DB_CONTAINER_NAME"
    )
    cosmos_db_throughput: int = Field(default=400, env="COSMOS_DB_THROUGHPUT")

    # Azure ML Workspace Configuration
    ml_workspace_name: str = Field(
        default="maintie-dev-ml-1cdd8e11", env="ML_WORKSPACE_NAME"
    )
    ml_experiment_name: str = Field(
        default="universal-rag-gnn", env="ML_EXPERIMENT_NAME"
    )
    ml_environment_name: str = Field(
        default="gnn-training-env", env="ML_ENVIRONMENT_NAME"
    )

    # Azure ML GNN Training Configuration
    azure_ml_compute_cluster_name: str = Field(
        default="gnn-cluster", env="AZURE_ML_COMPUTE_CLUSTER_NAME"
    )
    azure_ml_training_environment: str = Field(
        default="gnn-training-env", env="AZURE_ML_TRAINING_ENVIRONMENT"
    )
    gnn_model_deployment_tier: str = Field(
        default="adaptive", env="GNN_MODEL_DEPLOYMENT_TIER"
    )
    gnn_batch_size: int = Field(default=32, env="GNN_BATCH_SIZE")
    gnn_learning_rate: float = Field(default=0.01, env="GNN_LEARNING_RATE")
    gnn_num_epochs: int = Field(default=100, env="GNN_NUM_EPOCHS")
    gnn_training_compute_sku: str = Field(
        default="Standard_DS3_v2", env="GNN_TRAINING_COMPUTE_SKU"
    )
    gnn_model_deployment_endpoint: str = Field(
        default="gnn-inference-dev", env="GNN_MODEL_DEPLOYMENT_ENDPOINT"
    )
    gnn_training_enabled: bool = Field(default=True, env="GNN_TRAINING_ENABLED")
    gnn_quality_threshold: float = Field(default=0.6, env="GNN_QUALITY_THRESHOLD")
    gnn_testing_mode: str = Field(default="disabled", env="GNN_TESTING_MODE")

    # Azure Service Error Handling Configuration
    fail_on_enhancement_error: bool = Field(
        default=True,
        env="FAIL_ON_ENHANCEMENT_ERROR",
        description="Raise errors instead of silent fallbacks in pipeline enhancement",
    )

    azure_strict_error_handling: bool = Field(
        default=True,
        env="AZURE_STRICT_ERROR_HANDLING",
        description="Enable strict error handling across all Azure services",
    )

    # Environment-specific service configurations
    SERVICE_CONFIGS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "dev": {
            "search_sku": "adaptive",
            "search_replicas": 1,
            "storage_sku": "Standard_LRS",
            "cosmos_throughput": 400,
            "ml_compute_instances": 1,
            "openai_tokens_per_minute": 10000,
            "telemetry_sampling_rate": 10.0,
            "retention_days": 30,
            "app_insights_sampling": 10.0,
        },
        "staging": {
            "search_sku": "adaptive",
            "search_replicas": 1,
            "storage_sku": "Standard_ZRS",
            "cosmos_throughput": 800,
            "ml_compute_instances": 2,
            "openai_tokens_per_minute": 20000,
            "telemetry_sampling_rate": 5.0,
            "retention_days": 60,
            "app_insights_sampling": 5.0,
        },
        "prod": {
            "search_sku": "adaptive",
            "search_replicas": 2,
            "storage_sku": "Standard_GRS",
            "cosmos_throughput": 1600,
            "ml_compute_instances": 4,
            "openai_tokens_per_minute": 40000,
            "telemetry_sampling_rate": 1.0,
            "retention_days": 90,
            "app_insights_sampling": 1.0,
        },
    }

    def get_service_config(self, config_key: str):
        """Get environment-specific service configuration"""
        env_config = self.SERVICE_CONFIGS.get(
            self.azure_environment, self.SERVICE_CONFIGS["dev"]
        )
        return env_config.get(config_key)

    # Cost optimization properties
    @property
    def effective_search_sku(self) -> str:
        return self.get_service_config("search_sku")

    @property
    def effective_storage_sku(self) -> str:
        return self.get_service_config("storage_sku")

    @property
    def effective_openai_tokens_per_minute(self) -> int:
        return self.get_service_config("openai_tokens_per_minute")

    @property
    def effective_cosmos_throughput(self) -> int:
        return self.get_service_config("cosmos_throughput")

    @property
    def effective_ml_compute_instances(self) -> int:
        return self.get_service_config("ml_compute_instances")

    @property
    def effective_telemetry_sampling_rate(self) -> float:
        return self.get_service_config("telemetry_sampling_rate")

    @property
    def effective_retention_days(self) -> int:
        return self.get_service_config("retention_days")

    @property
    def resource_type_mappings(self) -> Dict[str, str]:
        """Get resource type mappings from environment"""
        mappings_env = os.getenv(
            "AZURE_RESOURCE_TYPE_MAPPINGS",
            '{"storage":"stor","search":"srch","keyvault":"kv","cosmos":"cosmos","ml":"ml","appinsights":"ai","loganalytics":"law"}',
        )
        return json.loads(mappings_env)

    def get_resource_name(self, resource_type: str, suffix: str = "") -> str:
        """Generate Azure resource names from configuration"""
        mappings = self.resource_type_mappings
        short_type = mappings.get(resource_type, resource_type)
        parts = [self.azure_resource_prefix, self.azure_environment, short_type]
        if suffix:
            parts.append(suffix)
        hyphen_excluded_types = os.getenv(
            "AZURE_HYPHEN_EXCLUDED_TYPES", "storage"
        ).split(",")
        if resource_type in hyphen_excluded_types:
            return "".join(parts)
        else:
            return "-".join(parts)

    def validate_azure_config(self) -> Dict[str, Any]:
        """Validate azd-managed Azure configuration completeness"""
        return {
            "storage_configured": bool(self.azure_storage_account),
            "search_configured": bool(self.azure_search_endpoint),
            "cosmos_configured": bool(self.azure_cosmos_endpoint),
            "ml_configured": bool(
                self.azure_subscription_id and self.azure_resource_group
            ),
            "openai_configured": bool(self.azure_openai_endpoint),
            "managed_identity_enabled": self.use_managed_identity,
        }

    # Raw data processing configuration
    @property
    def raw_data_include_patterns(self) -> List[str]:
        """Get supported file patterns from environment"""
        patterns_env = os.getenv("RAW_DATA_INCLUDE_PATTERNS", "*.md,*.txt")
        return [pattern.strip() for pattern in patterns_env.split(",")]

    @property
    def supported_text_formats(self) -> List[str]:
        """Get supported file extensions from environment"""
        formats_env = os.getenv("SUPPORTED_TEXT_FORMATS", ".md,.txt")
        return [fmt.strip() for fmt in formats_env.split(",")]

    # Azure Session and Connection Management
    azure_session_refresh_minutes: int = Field(
        default=50, env="AZURE_SESSION_REFRESH_MINUTES"
    )
    azure_connection_pool_size: int = Field(
        default=10, env="AZURE_CONNECTION_POOL_SIZE"
    )
    azure_health_check_timeout_seconds: int = Field(
        default=30, env="AZURE_HEALTH_CHECK_TIMEOUT_SECONDS"
    )
    azure_circuit_breaker_failure_threshold: int = Field(
        default=5, env="AZURE_CIRCUIT_BREAKER_FAILURE_THRESHOLD"
    )

    # Knowledge Discovery Settings - Increased for intelligent chunking
    discovery_sample_size: int = Field(default=500, env="DISCOVERY_SAMPLE_SIZE")
    max_discovery_batches: int = Field(default=100, env="MAX_DISCOVERY_BATCHES")
    max_entity_types_discovery: int = Field(
        default=100, env="MAX_ENTITY_TYPES_DISCOVERY"
    )
    max_relation_types_discovery: int = Field(
        default=60, env="MAX_RELATION_TYPES_DISCOVERY"
    )
    max_triplet_extraction_batches: int = Field(
        default=500, env="MAX_TRIPLET_EXTRACTION_BATCHES"
    )

    # === AZD-Compatibility Helper Methods ===
    @property
    def effective_openai_endpoint(self) -> str:
        """Get the effective OpenAI endpoint (azd-style or legacy)"""
        return self.azure_openai_endpoint or self.openai_api_base

    @property
    def effective_embedding_endpoint(self) -> str:
        """Get the effective embedding endpoint (same as OpenAI for Azure)"""
        return (
            self.azure_openai_endpoint
            or self.embedding_api_base
            or self.openai_api_base
        )

    @property
    def effective_search_endpoint(self) -> str:
        """Get the effective search endpoint"""
        return (
            self.azure_search_endpoint
            or f"https://{self.azure_search_service_name}.search.windows.net/"
        )

    @property
    def is_azd_deployment(self) -> bool:
        """Check if this is an azd-managed deployment"""
        return bool(
            self.azure_client_id
            and self.azure_openai_endpoint
            and self.use_managed_identity
        )

    @property
    def effective_deployment_name(self) -> str:
        """Get the effective OpenAI deployment name"""
        return self.openai_deployment_name

    @property
    def effective_embedding_deployment_name(self) -> str:
        """Get the effective embedding deployment name"""
        return self.embedding_deployment_name

    class Config:
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_storage_config(self, container_name: str = None) -> Dict[str, str]:
        """Get azd-managed storage configuration with managed identity"""
        return {
            "account_name": self.azure_storage_account,
            "container_name": container_name or self.azure_blob_container,
            "use_managed_identity": True,
        }


class AzureRegionalSettings(BaseSettings):
    """Regional Azure service performance optimization"""

    azure_primary_region: str = Field(env="AZURE_PRIMARY_REGION")
    azure_secondary_regions: List[str] = Field(
        default_factory=list, env="AZURE_SECONDARY_REGIONS"
    )
    enable_regional_performance_tracking: bool = Field(default=True)

    def get_optimal_service_region(self, service_type: str) -> str:
        """Determine optimal Azure region for service type"""
        regional_mapping = {
            "azure_openai": self.azure_primary_region,  # Consistency priority
            "cognitive_search": self.azure_primary_region,  # Data locality
            "cosmos_db": "global",  # Multi-master deployment
            "blob_storage": self.azure_primary_region,  # Bandwidth optimization
        }
        return regional_mapping.get(service_type, self.azure_primary_region)


# Global settings instance
settings = Settings()

# Alias for backward compatibility
AzureSettings = Settings
azure_settings = settings
