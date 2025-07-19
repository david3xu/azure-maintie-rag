"""
Configuration management for Azure Universal RAG
Centralizes all application settings and environment variables for Azure services
"""

import os
from pathlib import Path
from typing import Optional, List, ClassVar, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Unified application configuration settings for Azure services - single source of truth"""

    # Application Settings
    app_name: str = "Azure Universal RAG"
    app_version: str = "2.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # Azure OpenAI Settings
    openai_api_type: str = Field(default="azure", env="OPENAI_API_TYPE")
    openai_api_key: str = Field(default="1234567890", env="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://clu-project-foundry-instance.openai.azure.com/", env="OPENAI_API_BASE")
    openai_api_version: str = Field(default="2025-03-01-preview", env="OPENAI_API_VERSION")
    openai_deployment_name: str = Field(default="gpt-4.1", env="OPENAI_DEPLOYMENT_NAME")
    openai_model: str = Field(default="gpt-4.1", env="OPENAI_MODEL")

    # Azure Embedding Settings
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    embedding_deployment_name: str = Field(default="text-embedding-ada-002", env="EMBEDDING_DEPLOYMENT_NAME")
    embedding_api_base: str = Field(default="https://clu-project-foundry-instance.openai.azure.com/", env="EMBEDDING_API_BASE")
    embedding_api_version: str = Field(default="2025-03-01-preview", env="EMBEDDING_API_VERSION")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # Azure Storage Settings
    azure_storage_account: str = Field(default="", env="AZURE_STORAGE_ACCOUNT")
    azure_storage_key: str = Field(default="", env="AZURE_STORAGE_KEY")
    azure_blob_container: str = Field(default="universal-rag-data", env="AZURE_BLOB_CONTAINER")
    azure_storage_connection_string: str = Field(default="", env="AZURE_STORAGE_CONNECTION_STRING")

    # Azure Cognitive Search Settings
    azure_search_service: str = Field(default="", env="AZURE_SEARCH_SERVICE")
    azure_search_key: str = Field(default="", env="AZURE_SEARCH_KEY")
    azure_search_index: str = Field(default="universal-rag-index", env="AZURE_SEARCH_INDEX")
    azure_search_api_version: str = Field(default="2023-11-01", env="AZURE_SEARCH_API_VERSION")
    azure_search_service_name: str = Field(default="", env="AZURE_SEARCH_SERVICE_NAME")
    azure_search_admin_key: str = Field(default="", env="AZURE_SEARCH_ADMIN_KEY")

    # Azure Cosmos DB Settings (Gremlin API)
    azure_cosmos_endpoint: str = Field(default="", env="AZURE_COSMOS_ENDPOINT")
    azure_cosmos_key: str = Field(default="", env="AZURE_COSMOS_KEY")
    azure_cosmos_database: str = Field(default="universal-rag-db", env="AZURE_COSMOS_DATABASE")
    azure_cosmos_container: str = Field(default="knowledge-graph", env="AZURE_COSMOS_CONTAINER")
    azure_cosmos_db_connection_string: str = Field(default="", env="AZURE_COSMOS_DB_CONNECTION_STRING")

    # Azure ML Settings
    azure_subscription_id: str = Field(default="", env="AZURE_SUBSCRIPTION_ID")
    azure_resource_group: str = Field(default="", env="AZURE_RESOURCE_GROUP")
    azure_ml_workspace: str = Field(default="", env="AZURE_ML_WORKSPACE")
    azure_ml_workspace_name: str = Field(default="", env="AZURE_ML_WORKSPACE_NAME")
    azure_tenant_id: str = Field(default="", env="AZURE_TENANT_ID")

    # Azure Resource Naming Convention
    azure_resource_prefix: str = Field(default="maintie", env="AZURE_RESOURCE_PREFIX")
    azure_environment: str = Field(default="dev", env="AZURE_ENVIRONMENT")
    azure_region: str = Field(default="eastus", env="AZURE_REGION")

    # Data Paths (for local development)
    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent
    data_dir: Path = Field(default=BASE_DIR / "data", env="DATA_DIR")
    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw", env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=BASE_DIR / "data" / "processed", env="PROCESSED_DATA_DIR")
    indices_dir: Path = Field(default=BASE_DIR / "data" / "indices", env="INDICES_DIR")
    config_dir: Path = Field(default=BASE_DIR / "config", env="CONFIG_DIR")

    # Azure RAG Configuration
    discovery_sample_size: int = Field(default=10, env="DISCOVERY_SAMPLE_SIZE")
    pattern_confidence_threshold: float = Field(default=0.7, env="PATTERN_CONFIDENCE_THRESHOLD")
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
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )

    # Azure Discovery Settings
    discovery_enable_ner: bool = Field(default=True, env="DISCOVERY_ENABLE_NER")
    discovery_enable_relations: bool = Field(default=True, env="DISCOVERY_ENABLE_RELATIONS")

    # Trusted Hosts for Security
    trusted_hosts: Optional[List[str]] = Field(default=None, env="TRUSTED_HOSTS")

    def get_resource_name(self, resource_type: str, suffix: str = "") -> str:
        """Generate Azure resource names following convention"""
        parts = [self.azure_resource_prefix, self.azure_environment, self.azure_region, resource_type]
        if suffix:
            parts.append(suffix)
        return "-".join(parts)

    def validate_azure_config(self) -> Dict[str, Any]:
        """Validate Azure configuration completeness"""
        return {
            "storage_configured": bool(self.azure_storage_account and self.azure_storage_key),
            "search_configured": bool(self.azure_search_service and self.azure_search_key),
            "cosmos_configured": bool(self.azure_cosmos_endpoint and self.azure_cosmos_key),
            "ml_configured": bool(self.azure_subscription_id and self.azure_resource_group),
            "openai_configured": bool(self.openai_api_key and self.openai_api_base),
        }

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Alias for backward compatibility
AzureSettings = Settings
azure_settings = settings
