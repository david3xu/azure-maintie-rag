"""
Knowledge Extraction Models
==========================

Data models for knowledge extraction processes including entity and relationship
extraction, confidence scoring, text processing, and validation. These models
support the Knowledge Extraction Agent and related processing pipelines.

This module provides:
- Entity and relationship extraction models
- Text statistics and content analysis
- Confidence scoring and validation
- Content preprocessing and cleaning
- Extraction quality assessment
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, validator

from .base import ExtractionStatus, PydanticAIContextualModel

# =============================================================================
# CORE EXTRACTION MODELS
# =============================================================================


class ExtractedKnowledge(BaseModel):
    """Extracted knowledge representation"""

    entities: List[Dict[str, Any]] = Field(description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(description="Extracted relationships")
    confidence_scores: Dict[str, float] = Field(
        description="Extraction confidence scores"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extraction metadata"
    )
    quality_metrics: Dict[str, float] = Field(description="Quality assessment metrics")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")


class ExtractionResults(BaseModel):
    """Knowledge extraction results"""

    domain: str = Field(description="Source domain")
    extracted_knowledge: ExtractedKnowledge = Field(description="Extracted knowledge")
    quality_assessment: Dict[str, float] = Field(
        description="Quality assessment scores"
    )
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    validation_results: Dict[str, Any] = Field(description="Validation results")


class UnifiedExtractionResult(BaseModel):
    """Unified extraction processor result"""

    extraction_id: str = Field(default="", description="Unique extraction identifier")
    document_id: str = Field(default="", description="Source document identifier")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted relationships"
    )
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Processing time in seconds"
    )
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )
    extraction_method: str = Field(
        default="unified", description="Extraction method used"
    )
    confidence_distribution: Dict[str, float] = Field(
        default_factory=dict, description="Confidence score distribution"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )
    performance_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Performance statistics"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# =============================================================================
# EXTRACTION CONTEXT
# =============================================================================


class ExtractionContext(PydanticAIContextualModel):
    """Context information for extraction operations"""

    document_id: Optional[str] = Field(default=None, description="Document identifier")
    text_segment: str = Field(description="Text segment being processed")
    segment_start: int = Field(ge=0, description="Start position in full document")
    segment_end: int = Field(ge=0, description="End position in full document")

    # Context metadata
    domain_type: Optional[str] = Field(default=None, description="Detected domain type")
    language: str = Field(default="en", description="Text language")
    processing_hints: Dict[str, Any] = Field(
        default_factory=dict, description="Processing hints and parameters"
    )

    # Quality indicators
    text_quality_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Quality score of input text"
    )
    preprocessing_applied: List[str] = Field(
        default_factory=list, description="List of preprocessing steps applied"
    )

    @validator("segment_end")
    def validate_segment_bounds(cls, v, values):
        if "segment_start" in values and v < values["segment_start"]:
            raise ValueError("segment_end must be >= segment_start")
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for extraction operations"""
        return {
            "document_context": {
                "document_id": self.document_id,
                "segment_bounds": (self.segment_start, self.segment_end),
                "domain_type": self.domain_type,
                "language": self.language,
            },
            "extraction_context": {
                "text_quality": self.text_quality_score,
                "processing_hints": self.processing_hints,
                "preprocessing_applied": self.preprocessing_applied,
            },
        }


# =============================================================================
# KNOWLEDGE EXTRACTION RESULT (PYDANTIC AI)
# =============================================================================


class KnowledgeExtractionResult(PydanticAIContextualModel):
    """
    PydanticAI-enhanced knowledge extraction result following output validator patterns

    Replaces the previous UnifiedExtractionResult with proper Pydantic validation
    and cross-agent compatibility.
    """

    # Core extraction results
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted and validated entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted and validated relationships"
    )

    # Extraction statistics
    entity_count: int = Field(ge=0, description="Total number of entities extracted")
    relationship_count: int = Field(
        ge=0, description="Total number of relationships extracted"
    )
    unique_entity_types: int = Field(ge=0, description="Number of unique entity types")
    unique_relationship_types: int = Field(
        ge=0, description="Number of unique relationship types"
    )

    # Quality metrics
    avg_entity_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across entities"
    )
    avg_relationship_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across relationships"
    )
    extraction_quality_score: float = Field(
        ge=0.0, le=1.0, description="Overall extraction quality"
    )

    # Graph metrics
    entity_pairs_connected: int = Field(
        ge=0, description="Number of entity pairs connected by relationships"
    )
    graph_density: float = Field(
        ge=0.0, le=1.0, description="Density of the relationship graph"
    )
    connected_components: int = Field(
        ge=0, description="Number of disconnected graph components"
    )

    # Processing metadata
    extraction_method: str = Field(description="Primary extraction method used")
    processing_time_ms: float = Field(
        ge=0.0, description="Processing time in milliseconds"
    )
    text_length: int = Field(ge=0, description="Length of processed text")
    strategies_used: List[str] = Field(
        default_factory=list, description="List of extraction strategies applied"
    )

    # Validation results
    validation_passed: bool = Field(
        default=True, description="Whether PydanticAI validation passed"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )

    @validator("entity_count")
    def validate_entity_count(cls, v, values):
        if "entities" in values and v != len(values["entities"]):
            raise ValueError("entity_count must match length of entities list")
        return v

    @validator("relationship_count")
    def validate_relationship_count(cls, v, values):
        if "relationships" in values and v != len(values["relationships"]):
            raise ValueError(
                "relationship_count must match length of relationships list"
            )
        return v

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for knowledge extraction results"""
        return {
            "extraction_summary": {
                "entity_count": self.entity_count,
                "relationship_count": self.relationship_count,
                "quality_score": self.extraction_quality_score,
                "validation_passed": self.validation_passed,
            },
            "graph_metrics": {
                "entity_pairs_connected": self.entity_pairs_connected,
                "graph_density": self.graph_density,
                "connected_components": self.connected_components,
            },
            "performance": {
                "processing_time_ms": self.processing_time_ms,
                "text_length": self.text_length,
                "extraction_method": self.extraction_method,
                "strategies_used": self.strategies_used,
            },
        }


class KnowledgeValidationResult(BaseModel):
    """Simple validation result for knowledge extraction using PydanticAI validation"""

    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Validation warning messages"
    )
    entity_count: int = Field(ge=0, description="Number of entities validated")
    relationship_count: int = Field(
        ge=0, description="Number of relationships validated"
    )


# =============================================================================
# ENTITY AND RELATIONSHIP MODELS
# =============================================================================


class EntityExtractionResult(BaseModel):
    """Result of entity extraction process"""

    entity_name: str = Field(description="Extracted entity name")
    entity_type: str = Field(description="Entity type classification")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: str = Field(description="Context where entity was found")
    position_start: int = Field(ge=0, description="Start position in text")
    position_end: int = Field(ge=0, description="End position in text")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional entity attributes"
    )


class RelationshipExtractionResult(BaseModel):
    """Result of relationship extraction process"""

    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Relationship type")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: str = Field(description="Context where relationship was found")
    evidence_text: str = Field(description="Text evidence for relationship")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional relationship attributes"
    )


# =============================================================================
# TEXT STATISTICS AND ANALYSIS
# =============================================================================


class TextStatistics(BaseModel):
    """Comprehensive text statistics for document analysis"""

    char_count: int = Field(ge=0, description="Total character count")
    word_count: int = Field(ge=0, description="Total word count")
    sentence_count: int = Field(ge=0, description="Total sentence count")
    paragraph_count: int = Field(ge=0, description="Total paragraph count")

    avg_words_per_sentence: float = Field(
        ge=0.0, description="Average words per sentence"
    )
    avg_chars_per_word: float = Field(ge=0.0, description="Average characters per word")
    lexical_diversity: float = Field(
        ge=0.0, le=1.0, description="Lexical diversity score"
    )

    readability_score: float = Field(
        ge=0.0, le=100.0, description="Flesch Reading Ease score"
    )

    def calculate_readability(self) -> float:
        """Calculate Flesch Reading Ease score"""
        if self.sentence_count == 0 or self.word_count == 0:
            return 0.0
        return min(
            100.0,
            max(
                0.0,
                206.835
                - (1.015 * self.avg_words_per_sentence)
                - (84.6 * (self.avg_chars_per_word / 4.7)),
            ),
        )


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================


class ConfidenceScore(BaseModel):
    """Individual confidence score with metadata"""

    value: float = Field(ge=0.0, le=1.0, description="Confidence score value")
    source: str = Field(description="Source of confidence score")
    weight: float = Field(
        ge=0.0, le=1.0, description="Weight of this score in aggregation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class RelationshipConfidenceFactors(BaseModel):
    """Factors affecting relationship extraction confidence"""

    entity_confidence_product: float = Field(
        ge=0.0, le=1.0, description="Product of related entity confidences"
    )
    relationship_type_clarity: float = Field(
        ge=0.0, le=1.0, description="Relationship type clarity"
    )
    contextual_support: float = Field(
        ge=0.0, le=1.0, description="Contextual evidence support"
    )
    semantic_coherence: float = Field(
        ge=0.0, le=1.0, description="Semantic coherence score"
    )
    distance_penalty: float = Field(
        ge=0.0, le=1.0, description="Distance-based penalty factor"
    )
    domain_pattern_match: float = Field(
        ge=0.0, le=1.0, description="Domain pattern matching score"
    )


# =============================================================================
# CONTENT PREPROCESSING MODELS
# =============================================================================


class TextCleaningOptions(BaseModel):
    """Configuration for text cleaning operations"""

    remove_extra_whitespace: bool = Field(
        default=True, description="Remove extra whitespace"
    )
    normalize_unicode: bool = Field(
        default=True, description="Normalize Unicode characters"
    )
    remove_special_chars: bool = Field(
        default=False, description="Remove special characters"
    )
    lowercase: bool = Field(default=False, description="Convert to lowercase")
    remove_numbers: bool = Field(default=False, description="Remove numeric characters")
    remove_punctuation: bool = Field(default=False, description="Remove punctuation")

    # Advanced cleaning options
    remove_urls: bool = Field(default=True, description="Remove URLs")
    remove_emails: bool = Field(default=True, description="Remove email addresses")
    remove_phone_numbers: bool = Field(default=True, description="Remove phone numbers")
    preserve_sentence_structure: bool = Field(
        default=True, description="Preserve sentence boundaries"
    )


class CleanedContent(BaseModel):
    """Result of content cleaning operation"""

    original_text: str = Field(description="Original input text")
    cleaned_text: str = Field(description="Processed/cleaned text")
    cleaning_applied: List[str] = Field(
        description="List of cleaning operations applied"
    )
    original_length: int = Field(ge=0, description="Original text length")
    cleaned_length: int = Field(ge=0, description="Cleaned text length")
    reduction_percentage: float = Field(
        ge=0.0, le=100.0, description="Text reduction percentage"
    )


class ContentChunker(BaseModel):
    """Configuration for content chunking operations"""

    chunk_size: int = Field(ge=50, le=10000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(ge=0, le=1000, description="Overlap between chunks")
    preserve_sentences: bool = Field(
        default=True, description="Preserve sentence boundaries"
    )
    preserve_paragraphs: bool = Field(
        default=False, description="Preserve paragraph boundaries"
    )
    min_chunk_size: int = Field(ge=10, description="Minimum chunk size")


class ContentChunk(BaseModel):
    """Individual content chunk"""

    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    start_position: int = Field(ge=0, description="Start position in original text")
    end_position: int = Field(ge=0, description="End position in original text")
    token_count: int = Field(ge=0, description="Number of tokens in chunk")
    word_count: int = Field(ge=0, description="Number of words in chunk")

    # Chunk relationships
    previous_chunk_id: Optional[str] = Field(
        default=None, description="Previous chunk identifier"
    )
    next_chunk_id: Optional[str] = Field(
        default=None, description="Next chunk identifier"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional chunk metadata"
    )


# =============================================================================
# PYDANTIC AI OUTPUT VALIDATION MODELS
# =============================================================================


class ExtractionQualityOutput(BaseModel):
    """PydanticAI output validator for extraction quality assessment"""

    entities_per_text: float = Field(
        ge=1.0,
        le=20.0,
        description="Optimal entity extraction density - replaces hardcoded thresholds",
    )
    relations_per_entity: float = Field(
        ge=0.3,
        le=5.0,
        description="Healthy relationship coverage - agent learns optimal ratio",
    )
    avg_entity_confidence: float = Field(
        ge=0.6,
        le=1.0,
        description="Minimum acceptable entity confidence - dynamic threshold",
    )
    avg_relation_confidence: float = Field(
        ge=0.6,
        le=1.0,
        description="Minimum acceptable relationship confidence - learned threshold",
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite quality score - agent-determined weighting",
    )
    quality_tier: str = Field(
        pattern="^(excellent|good|acceptable|needs_improvement)$",
        description="Quality classification - agent learns tier boundaries",
    )


class ValidatedEntity(BaseModel):
    """PydanticAI output validator for entity extraction results"""

    name: str = Field(min_length=1, description="Entity name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score - agent determines threshold"
    )
    entity_type: str = Field(
        min_length=1, description="Entity classification - learned from domain"
    )
    extraction_method: str = Field(
        pattern="^(pattern_based|nlp_based|hybrid)$",
        description="Method used for extraction",
    )


class ValidatedRelationship(BaseModel):
    """PydanticAI output validator for relationship extraction"""

    source_entity: str = Field(min_length=1, description="Source entity")
    target_entity: str = Field(min_length=1, description="Target entity")
    relation_type: str = Field(min_length=1, description="Relationship type")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Relationship confidence - agent learned threshold"
    )


class ContentAnalysisOutput(BaseModel):
    """PydanticAI output validator for content analysis results"""

    word_count: int = Field(
        ge=50,
        description="Minimum meaningful content - agent determines based on domain",
    )
    vocabulary_richness: float = Field(
        ge=0.1,
        le=1.0,
        description="Vocabulary diversity - learned from corpus analysis",
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0, description="Content complexity - agent-driven assessment"
    )
    is_meaningful_content: bool = Field(
        description="Content quality gate - replaces manual validation chains"
    )
    technical_density: float = Field(
        ge=0.0, le=1.0, description="Technical term density - domain-specific threshold"
    )


# =============================================================================
# CONSOLIDATED EXTRACTION CONFIGURATION
# =============================================================================


class ConsolidatedExtractionConfiguration(PydanticAIContextualModel):
    """Consolidated configuration for all extraction processes"""

    # Entity extraction settings
    entity_confidence_threshold: float = Field(
        ge=0.0, le=1.0, description="Minimum entity confidence threshold"
    )
    max_entities_per_chunk: int = Field(
        default=50, ge=1, le=200, description="Maximum entities per chunk"
    )
    entity_types_enabled: List[str] = Field(
        default_factory=list, description="Enabled entity types"
    )

    # Relationship extraction settings
    relationship_confidence_threshold: float = Field(
        ge=0.0, le=1.0, description="Minimum relationship confidence threshold"
    )
    max_relationships_per_chunk: int = Field(
        default=100, ge=1, le=500, description="Maximum relationships per chunk"
    )
    relationship_types_enabled: List[str] = Field(
        default_factory=list, description="Enabled relationship types"
    )

    # Processing settings
    chunk_size: int = Field(
        default=1000, ge=100, le=5000, description="Text chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=100, ge=0, le=500, description="Overlap between chunks"
    )
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing"
    )
    max_concurrent_chunks: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent chunks"
    )

    # Quality and validation
    quality_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum quality threshold"
    )
    enable_validation: bool = Field(
        default=True, description="Enable result validation"
    )
    enable_confidence_filtering: bool = Field(
        default=True, description="Enable confidence-based filtering"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        return {
            "extraction_config": {
                "entity_settings": {
                    "confidence_threshold": self.entity_confidence_threshold,
                    "max_entities_per_chunk": self.max_entities_per_chunk,
                    "enabled_types": len(self.entity_types_enabled),
                },
                "relationship_settings": {
                    "confidence_threshold": self.relationship_confidence_threshold,
                    "max_relationships_per_chunk": self.max_relationships_per_chunk,
                    "enabled_types": len(self.relationship_types_enabled),
                },
                "processing_settings": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "parallel_processing": self.parallel_processing,
                    "max_concurrent_chunks": self.max_concurrent_chunks,
                },
                "quality_settings": {
                    "quality_threshold": self.quality_threshold,
                    "enable_validation": self.enable_validation,
                    "enable_confidence_filtering": self.enable_confidence_filtering,
                },
            }
        }
