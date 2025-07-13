"""
Configuration management for MaintIE Enhanced RAG
Centralizes all application settings and environment variables
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings"""

    # Application Settings
    app_name: str = "MaintIE Enhanced RAG"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # Azure OpenAI Settings
    openai_api_type: str = Field(default="azure", env="OPENAI_API_TYPE")
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_api_base: str = Field(env="OPENAI_API_BASE")
    openai_api_version: str = Field(env="OPENAI_API_VERSION")
    openai_deployment_name: str = Field(env="OPENAI_DEPLOYMENT_NAME")
    openai_model: str = Field(env="OPENAI_MODEL")

    # Embedding Settings (Azure)
    embedding_model: str = Field(env="EMBEDDING_MODEL")
    embedding_deployment_name: str = Field(env="EMBEDDING_DEPLOYMENT_NAME")
    embedding_api_base: str = Field(env="EMBEDDING_API_BASE")
    embedding_api_version: str = Field(env="EMBEDDING_API_VERSION")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # Data Paths
    BASE_DIR = Path(__file__).parent.parent
    data_dir: Path = Field(default=BASE_DIR / "data", env="DATA_DIR")
    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw", env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=BASE_DIR / "data" / "processed", env="PROCESSED_DATA_DIR")
    indices_dir: Path = Field(default=BASE_DIR / "data" / "indices", env="INDICES_DIR")

    # Data Processing Settings
    gold_data_filename: str = Field(default="gold_release.json", env="GOLD_DATA_FILENAME")
    silver_data_filename: str = Field(default="silver_release.json", env="SILVER_DATA_FILENAME")
    gold_confidence_base: float = Field(default=0.9, env="GOLD_CONFIDENCE_BASE")
    silver_confidence_base: float = Field(default=0.7, env="SILVER_CONFIDENCE_BASE")

    # Query Analysis Settings
    max_related_entities: int = Field(default=15, env="MAX_RELATED_ENTITIES")
    max_neighbors: int = Field(default=5, env="MAX_NEIGHBORS")
    concept_expansion_limit: int = Field(default=10, env="CONCEPT_EXPANSION_LIMIT")

    # Retrieval Settings
    vector_search_top_k: int = Field(default=10, env="VECTOR_SEARCH_TOP_K")
    entity_search_top_k: int = Field(default=8, env="ENTITY_SEARCH_TOP_K")
    graph_search_top_k: int = Field(default=6, env="GRAPH_SEARCH_TOP_K")
    embedding_batch_size: int = Field(default=128, env="EMBEDDING_BATCH_SIZE")
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Fusion Weights
    vector_weight: float = Field(default=0.4, env="VECTOR_WEIGHT")
    entity_weight: float = Field(default=0.3, env="ENTITY_WEIGHT")
    graph_weight: float = Field(default=0.3, env="GRAPH_WEIGHT")

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

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
