"""
Detailed Agent Specifications Models

Pydantic models for the innovative Domain Intelligence Agent tools
as specified in the detailed agent specifications.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StatisticalAnalysis(BaseModel):
    """Statistical corpus analysis results"""

    corpus_path: Optional[str] = Field(default=None, description="Path to analyzed corpus")
    total_documents: Optional[int] = Field(default=0, description="Total documents processed")
    total_tokens: int = Field(description="Total tokens analyzed")
    total_characters: Optional[int] = Field(default=0, description="Total characters analyzed")

    # Token frequency analysis
    token_frequencies: Dict[str, int] = Field(
        description="Token frequency distribution"
    )
    n_gram_patterns: Dict[str, int] = Field(description="N-gram pattern frequencies")
    vocabulary_size: int = Field(description="Unique vocabulary size")

    # Document structure analysis
    document_structures: Optional[Dict[str, int]] = Field(
        default_factory=dict, description="Document structure patterns"
    )
    average_document_length: float = Field(description="Average document length")
    document_count: Optional[int] = Field(default=0, description="Number of documents analyzed")
    length_distribution: Optional[Dict[str, int]] = Field(
        default_factory=dict, description="Document length distribution"
    )

    # Domain-specific metrics
    technical_term_density: Optional[float] = Field(default=0.0, description="Technical terminology density")
    domain_specificity_score: Optional[float] = Field(default=0.0, description="Domain specificity indicator")
    complexity_score: Optional[float] = Field(default=0.0, description="Content complexity score")

    # Quality metrics
    analysis_confidence: Optional[float] = Field(default=0.8, description="Analysis confidence score")
    processing_time_seconds: Optional[float] = Field(default=0.0, description="Analysis processing time")


class SemanticPatterns(BaseModel):
    """LLM-generated semantic patterns"""

    content_sample: str = Field(description="Sample content analyzed")

    # Semantic understanding
    domain_classification: str = Field(description="LLM-identified domain")
    primary_concepts: List[str] = Field(description="Primary domain concepts")
    concept_relationships: List[Dict[str, str]] = Field(
        description="Concept relationships"
    )

    # Entity patterns
    entity_types: List[str] = Field(description="Identified entity types")
    entity_examples: Dict[str, List[str]] = Field(
        description="Examples for each entity type"
    )
    entity_confidence: Dict[str, float] = Field(
        description="Confidence per entity type"
    )

    # Relationship patterns
    relationship_types: List[str] = Field(description="Identified relationship types")
    relationship_examples: Dict[str, List[str]] = Field(
        description="Examples for each relationship"
    )
    relationship_confidence: Dict[str, float] = Field(
        description="Confidence per relationship"
    )

    # Content structure insights
    content_structure_analysis: Dict[str, Any] = Field(
        description="Content structure insights"
    )
    processing_strategy_recommendation: str = Field(
        description="Recommended processing strategy"
    )

    # Quality metrics
    semantic_confidence: float = Field(description="Semantic analysis confidence")
    llm_processing_time: float = Field(description="LLM processing time")


class CombinedPatterns(BaseModel):
    """Combined statistical and semantic patterns"""

    statistical_analysis: StatisticalAnalysis = Field(
        description="Statistical analysis results"
    )
    semantic_patterns: SemanticPatterns = Field(description="Semantic pattern analysis")

    # Pattern fusion
    validated_entities: List[Dict[str, Any]] = Field(
        description="Cross-validated entity patterns"
    )
    validated_relationships: List[Dict[str, Any]] = Field(
        description="Cross-validated relationship patterns"
    )
    confidence_scores: Dict[str, float] = Field(
        description="Combined confidence scores"
    )

    # Processing recommendations
    optimal_extraction_strategy: str = Field(description="Optimal extraction strategy")
    recommended_thresholds: Dict[str, float] = Field(
        description="Recommended confidence thresholds"
    )
    processing_parameters: Dict[str, Any] = Field(
        description="Optimal processing parameters"
    )

    # Quality metrics
    pattern_consistency_score: float = Field(
        description="Pattern consistency between approaches"
    )
    overall_confidence: float = Field(description="Overall pattern confidence")


class QualityMetrics(BaseModel):
    """Pattern and configuration quality assessment"""

    config_path: Optional[str] = Field(
        description="Path to configuration being validated"
    )

    # Entity quality metrics
    entity_coverage: float = Field(description="Entity pattern coverage score")
    entity_precision_estimate: float = Field(description="Estimated entity precision")
    entity_recall_estimate: float = Field(description="Estimated entity recall")

    # Relationship quality metrics
    relationship_coverage: float = Field(description="Relationship pattern coverage")
    relationship_precision_estimate: float = Field(
        description="Estimated relationship precision"
    )
    relationship_recall_estimate: float = Field(
        description="Estimated relationship recall"
    )

    # Configuration quality
    config_completeness: float = Field(description="Configuration completeness score")
    config_consistency: float = Field(description="Internal configuration consistency")
    threshold_appropriateness: float = Field(
        description="Threshold appropriateness score"
    )

    # Performance predictions
    predicted_processing_time: float = Field(
        description="Predicted processing time per document"
    )
    predicted_memory_usage: float = Field(description="Predicted memory usage")
    predicted_accuracy: float = Field(description="Predicted extraction accuracy")

    # Validation results
    validation_passed: bool = Field(description="Whether validation passed")
    validation_warnings: List[str] = Field(description="Validation warnings")
    validation_errors: List[str] = Field(description="Validation errors")

    # Quality score
    overall_quality_score: float = Field(description="Overall quality score (0.0-1.0)")


class ExtractionConfiguration(BaseModel):
    """
    🎯 CORE MODEL: Complete extraction configuration with learned parameters
    
    Generated by Agent 1's create_fully_learned_extraction_config tool.
    Contains 100% learned values with zero hardcoded critical parameters.
    """
    
    # Core domain identification
    domain_name: str = Field(description="Domain name (learned from subdirectory)")
    
    # ✅ LEARNED: Critical parameters from data analysis
    entity_confidence_threshold: float = Field(description="Learned entity confidence threshold")
    relationship_confidence_threshold: float = Field(description="Learned relationship confidence threshold")
    chunk_size: int = Field(description="Learned optimal chunk size")
    chunk_overlap: int = Field(description="Learned chunk overlap")
    expected_entity_types: List[str] = Field(description="Learned entity types from corpus")
    
    # ✅ LEARNED: Performance parameters
    target_response_time_seconds: float = Field(description="Learned response SLA from complexity")
    
    # ✅ LEARNED: Content-based parameters
    technical_vocabulary: List[str] = Field(description="Learned technical vocabulary")
    key_concepts: List[str] = Field(description="Learned key concepts")
    
    # ✅ ACCEPTABLE HARDCODED: Non-critical parameters
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL (acceptable default)")
    parallel_processing_threshold: int = Field(description="Parallel processing threshold")
    max_concurrent_chunks: int = Field(default=5, description="Max concurrent chunks (acceptable default)")
    
    # Metadata
    generation_confidence: float = Field(description="Configuration generation confidence")
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring") 
    generation_timestamp: str = Field(description="When configuration was generated")


class DomainDeps(BaseModel):
    """Domain Intelligence Agent dependencies"""

    azure_services: Optional[Any] = Field(
        description="Azure services container", default=None
    )
    cache_manager: Optional[Any] = Field(
        description="Cache manager instance", default=None
    )
    hybrid_analyzer: Optional[Any] = Field(
        description="Hybrid domain analyzer", default=None
    )
    pattern_engine: Optional[Any] = Field(
        description="Pattern extraction engine", default=None
    )
    config_generator: Optional[Any] = Field(
        description="Configuration generator", default=None
    )

    class Config:
        arbitrary_types_allowed = True


# ===== MODELS FOR RESTORED LEGACY TOOLS =====

class HybridAnalysis(BaseModel):
    """Hybrid analysis combining LLM and statistical methods"""
    
    llm_extraction: Any = Field(description="LLM extraction results")
    statistical_features: Any = Field(description="Statistical analysis features")
    combined_confidence: float = Field(description="Combined analysis confidence")
    analysis_method: str = Field(default="hybrid_llm_statistical")
    
    class Config:
        arbitrary_types_allowed = True


class LLMExtraction(BaseModel):
    """LLM-based extraction results"""
    
    domain_classification: str = Field(description="Classified domain")
    confidence: float = Field(description="Classification confidence")
    extracted_entities: List[str] = Field(default_factory=list)
    extracted_concepts: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ExtractedPatterns(BaseModel):
    """Statistical pattern extraction results"""
    
    entity_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    action_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    relationship_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    source_word_count: int = Field(default=0)
    pattern_confidence: float = Field(default=0.8)
    
    class Config:
        arbitrary_types_allowed = True


class DomainAnalysisResult(BaseModel):
    """Complete domain analysis result"""
    
    domain: str = Field(description="Analyzed domain")
    classification: Dict[str, Any] = Field(description="Classification results")
    patterns_extracted: int = Field(description="Number of patterns extracted")
    config_generated: bool = Field(description="Whether config was generated")
    confidence: float = Field(description="Overall analysis confidence")
    
    class Config:
        arbitrary_types_allowed = True


# Export all models
__all__ = [
    "StatisticalAnalysis",
    "SemanticPatterns", 
    "CombinedPatterns",
    "ExtractionConfiguration",
    "QualityMetrics",
    "DomainDeps",
    "HybridAnalysis",
    "LLMExtraction", 
    "ExtractedPatterns",
    "DomainAnalysisResult",
]
