"""
Detailed Agent Specifications Models

Pydantic models for the innovative Domain Intelligence Agent tools
as specified in the detailed agent specifications.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StatisticalAnalysis(BaseModel):
    """Statistical corpus analysis results"""

    corpus_path: str = Field(description="Path to analyzed corpus")
    total_documents: int = Field(description="Total documents processed")
    total_tokens: int = Field(description="Total tokens analyzed")

    # Token frequency analysis
    token_frequencies: Dict[str, int] = Field(
        description="Token frequency distribution"
    )
    n_gram_patterns: Dict[str, int] = Field(description="N-gram pattern frequencies")
    vocabulary_size: int = Field(description="Unique vocabulary size")

    # Document structure analysis
    document_structures: Dict[str, int] = Field(
        description="Document structure patterns"
    )
    average_document_length: float = Field(description="Average document length")
    length_distribution: Dict[str, int] = Field(
        description="Document length distribution"
    )

    # Domain-specific metrics
    technical_term_density: float = Field(description="Technical terminology density")
    domain_specificity_score: float = Field(description="Domain specificity indicator")

    # Quality metrics
    analysis_confidence: float = Field(description="Analysis confidence score")
    processing_time_seconds: float = Field(description="Analysis processing time")


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


# Export all models
__all__ = [
    "StatisticalAnalysis",
    "SemanticPatterns",
    "CombinedPatterns",
    "QualityMetrics",
    "DomainDeps",
]
