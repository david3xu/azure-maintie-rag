"""
Configuration management for MaintIE Enhanced RAG
Centralizes all application settings and environment variables
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseSettings, Field


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

    # OpenAI Settings
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=500, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")

    # Embedding Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # Data Paths
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    raw_data_dir: Path = Field(default=Path("data/raw"), env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=Path("data/processed"), env="PROCESSED_DATA_DIR")
    indices_dir: Path = Field(default=Path("data/indices"), env="INDICES_DIR")

    # Knowledge Graph Settings
    max_entities: int = Field(default=10000, env="MAX_ENTITIES")
    max_relations: int = Field(default=50000, env="MAX_RELATIONS")
    graph_expansion_depth: int = Field(default=2, env="GRAPH_EXPANSION_DEPTH")

    # Retrieval Settings
    vector_search_top_k: int = Field(default=10, env="VECTOR_SEARCH_TOP_K")
    entity_search_top_k: int = Field(default=8, env="ENTITY_SEARCH_TOP_K")
    graph_search_top_k: int = Field(default=6, env="GRAPH_SEARCH_TOP_K")

    # Fusion Weights
    vector_weight: float = Field(default=0.4, env="VECTOR_WEIGHT")
    entity_weight: float = Field(default=0.3, env="ENTITY_WEIGHT")
    graph_weight: float = Field(default=0.3, env="GRAPH_WEIGHT")

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
