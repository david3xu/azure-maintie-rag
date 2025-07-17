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
    debug: bool = Field(default=False, env="DEBUG")

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # Azure OpenAI Settings (optional for testing)
    openai_api_type: str = Field(default="azure", env="OPENAI_API_TYPE")
    openai_api_key: str = Field(default="test-key", env="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://test.openai.azure.com/", env="OPENAI_API_BASE")
    openai_api_version: str = Field(default="2023-12-01-preview", env="OPENAI_API_VERSION")
    openai_deployment_name: str = Field(default="gpt-4", env="OPENAI_DEPLOYMENT_NAME")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")

    # Embedding Settings (Azure) - optional for testing
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    embedding_deployment_name: str = Field(default="text-embedding-ada-002", env="EMBEDDING_DEPLOYMENT_NAME")
    embedding_api_base: str = Field(default="https://test.openai.azure.com/", env="EMBEDDING_API_BASE")
    embedding_api_version: str = Field(default="2023-12-01-preview", env="EMBEDDING_API_VERSION")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # Data Paths
    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent
    data_dir: Path = Field(default=BASE_DIR / "data", env="DATA_DIR")
    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw", env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=BASE_DIR / "data" / "processed", env="PROCESSED_DATA_DIR")
    indices_dir: Path = Field(default=BASE_DIR / "data" / "indices", env="INDICES_DIR")
    config_dir: Path = Field(default=BASE_DIR / "config", env="CONFIG_DIR")

    # Data Processing Settings
    # Note: Universal RAG now works with raw text files - no domain-specific JSON format required
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
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
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

    # Domain Knowledge Settings
    troubleshooting_keywords: List[str] = Field(
        default=[
            'failure', 'problem', 'issue', 'broken', 'not working',
            'troubleshoot', 'diagnose', 'fix', 'repair', 'malfunction'
        ],
        env="TROUBLESHOOTING_KEYWORDS"
    )

    procedural_keywords: List[str] = Field(
        default=[
            'how to', 'procedure', 'steps', 'process', 'method',
            'instructions', 'guide', 'manual', 'protocol'
        ],
        env="PROCEDURAL_KEYWORDS"
    )

    preventive_keywords: List[str] = Field(
        default=[
            'preventive', 'maintenance schedule', 'inspection',
            'service', 'routine', 'periodic', 'scheduled'
        ],
        env="PREVENTIVE_KEYWORDS"
    )

    safety_keywords: List[str] = Field(
        default=[
            'safety', 'hazard', 'risk', 'dangerous', 'caution',
            'warning', 'lockout', 'ppe', 'procedure'
        ],
        env="SAFETY_KEYWORDS"
    )

    # Equipment Categories
    equipment_categories: Dict[str, List[str]] = Field(
        default={
            'rotating_equipment': ['pump', 'motor', 'compressor', 'turbine', 'fan'],
            'static_equipment': ['tank', 'vessel', 'pipe', 'valve'],
            'electrical': ['motor', 'generator', 'transformer', 'panel'],
            'hvac': ['fan', 'damper', 'coil', 'duct', 'filter'],
            'instrumentation': ['sensor', 'transmitter', 'gauge', 'indicator']
        }
    )

    # Abbreviation Expansions
    technical_abbreviations: Dict[str, str] = Field(
        default={
            'pm': 'preventive maintenance',
            'cm': 'corrective maintenance',
            'hvac': 'heating ventilation air conditioning',
            'loto': 'lockout tagout',
            'sop': 'standard operating procedure',
            'rca': 'root cause analysis'
        }
    )

    # Component Patterns
    component_patterns: Dict[str, str] = Field(
        default={
            r'\bbearing\b': 'bearing',
            r'\bseal\b': 'seal',
            r'\bgasket\b': 'gasket',
            r'\bvalve\b': 'valve',
            r'\bmotor\b': 'motor',
            r'\bfilter\b': 'filter',
            r'\bbelt\b': 'belt',
            r'\bcoupling\b': 'coupling'
        }
    )

    # Equipment Patterns
    equipment_patterns: Dict[str, str] = Field(
        default={
            r'\bpump\b': 'pump',
            r'\bmotor\b': 'motor',
            r'\bcompressor\b': 'compressor',
            r'\bturbine\b': 'turbine',
            r'\bfan\b': 'fan',
            r'\bvalve\b': 'valve',
            r'\btank\b': 'tank',
            r'\bvessel\b': 'vessel',
            r'\bpipe\b': 'pipe',
            r'\bheat exchanger\b': 'heat exchanger'
        }
    )

    # Failure Patterns
    failure_patterns: Dict[str, str] = Field(
        default={
            r'\bfailure\b': 'failure',
            r'\bleak\b': 'leak',
            r'\bvibration\b': 'vibration',
            r'\bnoise\b': 'noise',
            r'\boverheating\b': 'overheating',
            r'\bwear\b': 'wear',
            r'\bcorrosion\b': 'corrosion',
            r'\bcrack\b': 'crack',
            r'\bmisalignment\b': 'misalignment'
        }
    )

    # Procedure Patterns
    procedure_patterns: Dict[str, str] = Field(
        default={
            r'\bmaintenance\b': 'maintenance',
            r'\binspection\b': 'inspection',
            r'\brepair\b': 'repair',
            r'\breplacement\b': 'replacement',
            r'\binstallation\b': 'installation',
            r'\bcalibration\b': 'calibration',
            r'\btesting\b': 'testing',
            r'\bservicing\b': 'servicing'
        }
    )

    # Tool Mappings
    tool_mappings: Dict[str, List[str]] = Field(
        default={
            'pump': ['wrench set', 'pressure gauge', 'vibration meter'],
            'motor': ['multimeter', 'insulation tester', 'alignment tool'],
            'bearing': ['bearing puller', 'lubricant', 'dial indicator'],
            'seal': ['seal installation tool', 'torque wrench', 'gasket material']
        }
    )

    # Safety Mappings
    safety_mappings: Dict[str, List[str]] = Field(
        default={
            'electrical': ['lockout/tagout', 'PPE required', 'voltage testing'],
            'pressure': ['pressure relief', 'isolation', 'proper venting'],
            'rotating': ['guards in place', 'stop rotation', 'clear area'],
            'chemical': ['MSDS review', 'containment', 'ventilation']
        }
    )

    # Expansion Rules
    expansion_rules: Dict[str, List[str]] = Field(
        default={
            'pump': ['centrifugal pump', 'positive displacement pump', 'impeller', 'volute'],
            'seal': ['mechanical seal', 'packing', 'gasket', 'O-ring'],
            'bearing': ['ball bearing', 'roller bearing', 'thrust bearing', 'lubrication'],
            'motor': ['electric motor', 'AC motor', 'DC motor', 'stator', 'rotor'],
            'failure': ['malfunction', 'breakdown', 'defect', 'wear', 'damage']
        }
    )

    # Typical Procedures
    typical_procedures: Dict[str, List[str]] = Field(
        default={
            'troubleshooting': ['visual inspection', 'diagnostic testing', 'root cause analysis'],
            'preventive': ['scheduled inspection', 'lubrication', 'replacement'],
            'procedural': ['step-by-step guide', 'safety checklist', 'quality verification']
        }
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
