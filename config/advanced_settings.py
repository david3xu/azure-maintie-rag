"""
Extended configuration for MaintIE Enhanced RAG
Addresses all hard-coded values with environment overrides
"""

from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseSettings, Field


class AdvancedSettings(BaseSettings):
    """Extended settings covering all configurable parameters"""

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
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Generation Settings
    llm_top_p: float = Field(default=0.9, env="LLM_TOP_P")
    llm_frequency_penalty: float = Field(default=0.1, env="LLM_FREQUENCY_PENALTY")
    llm_presence_penalty: float = Field(default=0.1, env="LLM_PRESENCE_PENALTY")

    # API Validation Settings
    query_min_length: int = Field(default=3, env="QUERY_MIN_LENGTH")
    query_max_length: int = Field(default=500, env="QUERY_MAX_LENGTH")
    max_results_limit: int = Field(default=50, env="MAX_RESULTS_LIMIT")

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


# Global advanced settings instance
advanced_settings = AdvancedSettings()
