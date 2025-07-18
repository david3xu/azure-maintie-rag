"""
Configuration management for MaintIE Enhanced RAG
Centralizes all application settings and environment variables
"""

import os
from pathlib import Path
from typing import Optional, List, ClassVar, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Unified application configuration settings - single source of truth"""

    # Application Settings
    app_name: str = "MaintIE Enhanced RAG"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # Azure OpenAI Settings (optional for testing)
    openai_api_type: str = Field(default="azure", env="OPENAI_API_TYPE")
    openai_api_key: str = Field(default="1234567890", env="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://clu-project-foundry-instance.openai.azure.com/", env="OPENAI_API_BASE")
    openai_api_version: str = Field(default="2025-03-01-preview", env="OPENAI_API_VERSION")
    openai_deployment_name: str = Field(default="gpt-4.1", env="OPENAI_DEPLOYMENT_NAME")
    openai_model: str = Field(default="gpt-4.1", env="OPENAI_MODEL")

    # Embedding Settings (Azure) - optional for testing
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    embedding_deployment_name: str = Field(default="text-embedding-ada-002", env="EMBEDDING_DEPLOYMENT_NAME")
    embedding_api_base: str = Field(default="https://clu-project-foundry-instance.openai.azure.com/", env="EMBEDDING_API_BASE")
    embedding_api_version: str = Field(default="2025-03-01-preview", env="EMBEDDING_API_VERSION")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # Data Paths
    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent
    data_dir: Path = Field(default=BASE_DIR / "data", env="DATA_DIR")
    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw", env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=BASE_DIR / "data" / "processed", env="PROCESSED_DATA_DIR")
    indices_dir: Path = Field(default=BASE_DIR / "data" / "indices", env="INDICES_DIR")
    config_dir: Path = Field(default=BASE_DIR / "config", env="CONFIG_DIR")

    # Universal RAG Settings
    enable_universal_rag: bool = Field(default=True, env="ENABLE_UNIVERSAL_RAG")
    default_domain: str = Field(default="general", env="DEFAULT_DOMAIN")
    enable_dynamic_discovery: bool = Field(default=True, env="ENABLE_DYNAMIC_DISCOVERY")
    discovery_sample_size: int = Field(default=10, env="DISCOVERY_SAMPLE_SIZE")
    pattern_confidence_threshold: float = Field(default=0.7, env="PATTERN_CONFIDENCE_THRESHOLD")
    universal_text_processing: bool = Field(default=True, env="UNIVERSAL_TEXT_PROCESSING")
    schema_free_processing: bool = Field(default=True, env="SCHEMA_FREE_PROCESSING")

    # Query Analysis Settings
    max_related_entities: int = Field(default=15, env="MAX_RELATED_ENTITIES")
    max_neighbors: int = Field(default=5, env="MAX_NEIGHBORS")
    concept_expansion_limit: int = Field(default=10, env="CONCEPT_EXPANSION_LIMIT")

    # Retrieval Settings
    vector_search_top_k: int = Field(default=10, env="VECTOR_SEARCH_TOP_K")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Generation Settings
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

    # Universal Domain Context Mappings
    universal_domain_contexts: Dict[str, str] = Field(
        default={
            "maintenance": "industrial equipment maintenance and troubleshooting",
            "medical": "medical information and healthcare guidance",
            "legal": "legal information and document analysis",
            "finance": "financial analysis and business guidance",
            "education": "educational content and learning materials",
            "general": "general knowledge and information"
        }
    )

    # Dynamic Discovery Settings
    discovery_min_confidence: float = Field(default=0.6, env="DISCOVERY_MIN_CONFIDENCE")
    discovery_max_patterns: int = Field(default=50, env="DISCOVERY_MAX_PATTERNS")
    discovery_enable_ner: bool = Field(default=True, env="DISCOVERY_ENABLE_NER")
    discovery_enable_relations: bool = Field(default=True, env="DISCOVERY_ENABLE_RELATIONS")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
