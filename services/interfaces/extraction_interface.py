"""
Extraction Configuration Interface

Defines the contract between Data-Driven Configuration System and Knowledge Extraction Pipeline.
This interface ensures clean separation and prevents overlap between components.

Following Azure Universal RAG Coding Standards:
- Data-driven configuration (no hardcoded values)
- Production-ready with comprehensive validation
- Universal design (works with any domain)
- Performance-first (async-ready structures)
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class ExtractionStrategy(str, Enum):
    """Data-driven extraction strategies discovered from domain analysis"""
    TECHNICAL_CONTENT = "technical_content"
    STRUCTURED_DATA = "structured_data"
    CONVERSATIONAL = "conversational"
    MIXED_CONTENT = "mixed_content"
    AUTO_DETECT = "auto_detect"


class ExtractionConfiguration(BaseModel):
    """
    Configuration interface passed from Config System to Extraction Pipeline.
    
    Contains all parameters needed for optimal knowledge extraction,
    derived from domain-wide data analysis.
    """
    
    # Domain context
    domain_name: str = Field(..., description="Domain name from directory structure")
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    config_version: str = Field(default="1.0.0")
    
    # Entity extraction parameters (data-driven)
    entity_confidence_threshold: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence threshold derived from domain analysis"
    )
    expected_entity_types: List[str] = Field(
        default_factory=list,
        description="Entity types discovered from domain patterns"
    )
    entity_extraction_focus: ExtractionStrategy = Field(
        default=ExtractionStrategy.AUTO_DETECT,
        description="Extraction strategy optimized for domain"
    )
    max_entities_per_chunk: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum entities per processing chunk"
    )
    
    # Relationship extraction parameters (data-driven)
    relationship_patterns: List[str] = Field(
        default_factory=list,
        description="Relationship patterns discovered from domain analysis"
    )
    relationship_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Relationship confidence derived from domain patterns"
    )
    max_relationships_per_chunk: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum relationships per processing chunk"
    )
    
    # Processing parameters (performance-optimized)
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Optimal chunk size derived from domain analysis"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Chunk overlap optimized for domain content patterns"
    )
    processing_strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.AUTO_DETECT,
        description="Processing strategy optimized for domain"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for performance"
    )
    max_concurrent_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent chunk processing"
    )
    
    # Domain-specific vocabulary (learned from data)
    technical_vocabulary: List[str] = Field(
        default_factory=list,
        max_items=500,
        description="Technical terms discovered from domain analysis"
    )
    key_concepts: List[str] = Field(
        default_factory=list,
        max_items=100,
        description="Key concepts identified in domain"
    )
    stop_words_additions: List[str] = Field(
        default_factory=list,
        max_items=50,
        description="Domain-specific stop words to ignore"
    )
    
    # Quality and validation thresholds (data-derived)
    minimum_quality_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for extracted knowledge"
    )
    validation_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation criteria derived from domain patterns"
    )
    extraction_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout for extraction operations"
    )
    
    # Performance optimization settings
    enable_caching: bool = Field(default=True, description="Enable extraction result caching")
    cache_ttl_seconds: int = Field(default=3600, ge=300, le=86400)
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    target_response_time_seconds: float = Field(
        default=3.0,
        gt=0.0,
        le=30.0,
        description="Target response time for extraction"
    )

    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is reasonable relative to chunk size"""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    @validator('expected_entity_types')
    def validate_entity_types(cls, v):
        """Ensure entity types are non-empty strings"""
        if not v:
            return v
        for entity_type in v:
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise ValueError("Entity types must be non-empty strings")
        return v

    class Config:
        """Pydantic configuration following coding standards"""
        use_enum_values = True
        extra = "forbid"  # Strict validation
        str_strip_whitespace = True
        validate_assignment = True


class ExtractionResults(BaseModel):
    """
    Results from Knowledge Extraction Pipeline.
    
    Used for feedback to improve Configuration System performance.
    """
    
    # Processing metadata
    domain_name: str = Field(..., description="Domain that was processed")
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    documents_processed: int = Field(..., ge=0)
    total_processing_time_seconds: float = Field(..., ge=0.0)
    
    # Quality metrics (for config feedback)
    extraction_accuracy: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Overall extraction accuracy score"
    )
    entity_precision: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Entity extraction precision"
    )
    entity_recall: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Entity extraction recall"
    )
    relationship_precision: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Relationship extraction precision"
    )
    relationship_recall: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Relationship extraction recall"
    )
    
    # Performance metrics (for config optimization)
    average_processing_time_per_document: float = Field(..., ge=0.0)
    memory_usage_mb: float = Field(..., ge=0.0)
    cpu_utilization_percent: float = Field(..., ge=0.0, le=100.0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Discovery insights (for config improvement)
    new_entity_types_discovered: List[str] = Field(
        default_factory=list,
        description="New entity types found during processing"
    )
    unexpected_relationship_patterns: List[str] = Field(
        default_factory=list,
        description="Unexpected relationship patterns discovered"
    )
    domain_vocabulary_gaps: List[str] = Field(
        default_factory=list,
        description="Important terms missing from domain vocabulary"
    )
    quality_issues_detected: List[str] = Field(
        default_factory=list,
        description="Quality issues that could improve config"
    )
    
    # Output statistics
    total_entities_extracted: int = Field(..., ge=0)
    total_relationships_extracted: int = Field(..., ge=0)
    unique_entity_types_found: int = Field(..., ge=0)
    unique_relationship_types_found: int = Field(..., ge=0)
    
    # Validation results
    extraction_passed_validation: bool = Field(...)
    validation_error_count: int = Field(default=0, ge=0)
    validation_warnings: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration following coding standards"""
        extra = "forbid"
        validate_assignment = True


class ConfigurationFeedback(BaseModel):
    """
    Feedback structure for Config System optimization.
    
    Enables continuous improvement of extraction configurations.
    """
    
    # Source information
    domain_name: str = Field(...)
    feedback_timestamp: datetime = Field(default_factory=datetime.now)
    extraction_results: ExtractionResults = Field(...)
    
    # Optimization recommendations
    recommended_threshold_adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="Recommended adjustments to confidence thresholds"
    )
    recommended_parameter_changes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Recommended changes to processing parameters"
    )
    recommended_vocabulary_additions: List[str] = Field(
        default_factory=list,
        description="Terms to add to domain vocabulary"
    )
    
    # Performance insights
    performance_bottlenecks: List[str] = Field(
        default_factory=list,
        description="Identified performance bottlenecks"
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list,
        description="Opportunities for further optimization"
    )
    
    # Quality improvements
    quality_improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving extraction quality"
    )
    
    class Config:
        """Pydantic configuration following coding standards"""
        extra = "forbid"
        validate_assignment = True


# Export interface components
__all__ = [
    'ExtractionStrategy',
    'ExtractionConfiguration', 
    'ExtractionResults',
    'ConfigurationFeedback'
]