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

    # Azure Services Settings
    azure_storage_connection_string: str = Field(default="", env="AZURE_STORAGE_CONNECTION_STRING")
    azure_search_service_name: str = Field(default="", env="AZURE_SEARCH_SERVICE_NAME")
    azure_search_admin_key: str = Field(default="", env="AZURE_SEARCH_ADMIN_KEY")
    azure_cosmos_db_connection_string: str = Field(default="", env="AZURE_COSMOS_DB_CONNECTION_STRING")
    azure_ml_workspace_name: str = Field(default="", env="AZURE_ML_WORKSPACE_NAME")

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

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
