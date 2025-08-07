"""
Workflow State and Execution Models
==================================

Workflow orchestration, state management, and execution tracking models
for the multi-agent coordination system. These models handle the complex
state transitions and data flows between agents.

This module provides:
- Workflow execution state tracking
- Node execution results and coordination
- State transfer between workflow graphs
- Background processing configuration
- Performance feedback integration
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, validator

from agents.core.constants import (
    ExtractionQualityConstants,
    MathematicalFoundationConstants,
    StatisticalConstants,
    SystemPerformanceConstants,
    WorkflowExecutionConstants,
)

from .base import (
    NodeState,
    PydanticAIContextualModel,
    StateTransferType,
    WorkflowState,
)

# =============================================================================
# BACKGROUND PROCESSING MODELS
# =============================================================================


@dataclass
class ProcessingStats:
    """Statistics for background processing operations"""

    total_domains: int = 0
    successful_processes: int = 0
    failed_processes: int = 0
    total_documents: int = 0
    total_processing_time: float = 0.0
    average_confidence: float = 0.0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate processing success rate"""
        if self.total_domains == 0:
            return 0.0
        return self.successful_processes / self.total_domains

    @property
    def processing_duration(self) -> Optional[timedelta]:
        """Calculate total processing duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class DomainSignature:
    """Domain signature containing essential patterns and configuration"""

    domain: str
    patterns: Dict[str, Any] = field(default_factory=dict)
    config: Optional[Any] = None  # CompleteDomainConfig
    signature_confidence: float = 0.0
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "domain": self.domain,
            "patterns": self.patterns,
            "config": self.config.dict() if self.config else None,
            "signature_confidence": self.signature_confidence,
            "creation_timestamp": self.creation_timestamp,
        }


# =============================================================================
# WORKFLOW EXECUTION STATE MODELS
# =============================================================================


@dataclass
class WorkflowExecutionState:
    """Workflow execution state management"""

    workflow_id: str
    current_state: WorkflowState
    nodes_completed: int
    total_nodes: int
    start_time: datetime
    current_node: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NodeExecutionResult:
    """Individual node execution result"""

    node_id: str
    state: NodeState
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None

    def __post_init__(self):
        if self.end_time and self.execution_time_seconds is None:
            self.execution_time_seconds = (
                self.end_time - self.start_time
            ).total_seconds()


@dataclass
class WorkflowStateBridge:
    """Bridge data between workflow states"""

    source_state: str
    target_state: str
    transition_data: Dict[str, Any]
    timestamp: float

    def __post_init__(self):
        if self.transition_data is None:
            self.transition_data = {}


# =============================================================================
# WORKFLOW RESULT CONTRACT
# =============================================================================


class WorkflowResultContract(PydanticAIContextualModel):
    """Enhanced workflow execution result contract with performance feedback integration"""

    # Basic workflow information
    workflow_id: str = Field(description="Workflow identifier")
    workflow_type: str = Field(
        description="Type of workflow (config_extraction, search, analysis)"
    )
    execution_state: WorkflowState = Field(description="Final execution state")

    # Execution results
    results: Dict[str, Any] = Field(description="Workflow results")
    node_results: List[Dict[str, Any]] = Field(description="Individual node results")

    # Performance and quality metrics
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    quality_scores: Dict[str, float] = Field(description="Quality assessment scores")
    total_execution_time: float = Field(ge=0.0, description="Total execution time")

    # Configuration and context tracking
    configurations_used: Dict[str, Dict[str, Any]] = Field(
        description="Configurations used by each agent"
    )
    domain_context: Optional[str] = Field(
        default=None, description="Domain context for this workflow"
    )

    # Error and warning handling
    error_log: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    recovery_actions: List[str] = Field(
        default_factory=list, description="Recovery actions taken"
    )

    # Performance feedback integration (forward reference)
    performance_feedback_points: List[Any] = Field(
        default_factory=list,
        description="Performance feedback points generated during execution",
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list, description="Identified optimization opportunities"
    )

    # Workflow learning and improvement
    lessons_learned: List[str] = Field(
        default_factory=list, description="Lessons learned from this execution"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for workflow result processing"""
        return {
            "workflow_info": {
                "workflow_id": self.workflow_id,
                "workflow_type": self.workflow_type,
                "execution_state": self.execution_state.value,
                "domain_context": self.domain_context,
            },
            "performance_summary": {
                "total_execution_time": self.total_execution_time,
                "average_quality_score": (
                    sum(self.quality_scores.values()) / len(self.quality_scores)
                    if self.quality_scores
                    else 0.0
                ),
                "success_rate": (
                    1.0 if self.execution_state == WorkflowState.COMPLETED else 0.0
                ),
                "error_count": len(self.error_log),
            },
            "feedback_summary": {
                "feedback_points": len(self.performance_feedback_points),
                "optimization_opportunities": len(self.optimization_opportunities),
                "improvement_suggestions": len(self.improvement_suggestions),
            },
        }

    def is_successful(self) -> bool:
        """Check if workflow completed successfully"""
        return (
            self.execution_state == WorkflowState.COMPLETED and len(self.error_log) == 0
        )

    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score from all quality metrics"""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores.values()) / len(self.quality_scores)


# =============================================================================
# STATE TRANSFER MODELS
# =============================================================================


class StateTransferPacket(PydanticAIContextualModel):
    """Data packet for transferring state between workflow graphs"""

    transfer_id: str = Field(..., description="Unique transfer identifier")
    transfer_type: StateTransferType = Field(..., description="Type of state transfer")
    source_workflow: str = Field(..., description="Source workflow name")
    target_workflow: str = Field(..., description="Target workflow name")
    payload: Dict[str, Any] = Field(..., description="Transfer payload data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Transfer timestamp",
    )
    expiry: Optional[datetime] = Field(default=None, description="Transfer expiry time")
    dependencies: List[str] = Field(
        default_factory=list, description="Transfer dependencies"
    )
    version: str = Field(default="1.0", description="Transfer packet version")
    checksum: Optional[str] = Field(default=None, description="Data integrity checksum")

    @validator("expiry", pre=True, always=True)
    def set_default_expiry(cls, v, values):
        if v is None and "timestamp" in values:
            timestamp = values["timestamp"]
            return timestamp.replace(hour=23, minute=59, second=59)
        return v

    @validator("checksum", pre=True, always=True)
    def calculate_checksum(cls, v, values):
        if v is None and "payload" in values:
            data_str = json.dumps(values["payload"], sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()
        return v

    def is_expired(self) -> bool:
        """Check if state transfer packet has expired"""
        return datetime.now(timezone.utc) > self.expiry

    def validate_integrity(self) -> bool:
        """Validate data integrity using checksum"""
        data_str = json.dumps(self.payload, sort_keys=True, default=str)
        expected_checksum = hashlib.md5(data_str.encode()).hexdigest()
        return self.checksum == expected_checksum

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI state transfer operations"""
        return {
            "transfer_metadata": {
                "transfer_id": self.transfer_id,
                "transfer_type": (
                    self.transfer_type.value
                    if hasattr(self.transfer_type, "value")
                    else str(self.transfer_type)
                ),
                "source_workflow": self.source_workflow,
                "target_workflow": self.target_workflow,
                "version": self.version,
                "dependency_count": len(self.dependencies),
            },
            "transfer_timing": {
                "timestamp": self.timestamp.isoformat(),
                "expiry": self.expiry.isoformat() if self.expiry else None,
                "is_expired": self.is_expired(),
                "time_to_expiry": (
                    (self.expiry - datetime.now(timezone.utc)).total_seconds()
                    if self.expiry
                    else None
                ),
            },
            "data_integrity": {
                "checksum": self.checksum,
                "payload_size": len(str(self.payload)),
                "integrity_valid": self.validate_integrity(),
            },
        }


# =============================================================================
# BACKGROUND PROCESSING MODELS
# =============================================================================


@dataclass
class BackgroundProcessingConfig:
    """Background processing configuration"""

    processing_enabled: bool
    batch_size: int
    processing_interval: float
    max_retries: int
    timeout_seconds: float


@dataclass
class BackgroundProcessingResult:
    """Background processing execution result"""

    processing_id: str
    success: bool
    items_processed: int
    processing_time: float
    error_message: Optional[str] = None


# =============================================================================
# STATISTICAL ANALYSIS MODELS
# =============================================================================


class StatisticalAnalysis(BaseModel):
    """Statistical corpus analysis results"""

    corpus_path: Optional[str] = Field(
        default=None, description="Path to analyzed corpus"
    )
    total_documents: int = Field(ge=0, description="Total documents processed")
    total_tokens: int = Field(ge=0, description="Total tokens analyzed")
    total_characters: int = Field(ge=0, description="Total characters analyzed")
    token_frequencies: Dict[str, int] = Field(
        description="Token frequency distribution"
    )
    n_gram_patterns: Dict[str, int] = Field(description="N-gram pattern frequencies")
    vocabulary_size: int = Field(ge=0, description="Unique vocabulary size")
    document_structures: Dict[str, int] = Field(
        default_factory=dict, description="Document structure patterns"
    )
    average_document_length: float = Field(
        ge=0.0, description="Average document length"
    )
    document_count: int = Field(ge=0, description="Number of documents analyzed")
    length_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Document length distribution"
    )
    technical_term_density: float = Field(
        ge=0.0, le=1.0, description="Technical terminology density"
    )
    domain_specificity_score: float = Field(
        ge=0.0, le=1.0, description="Domain specificity indicator"
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0, description="Content complexity score"
    )
    analysis_confidence: float = Field(
        default=StatisticalConstants.TECHNICAL_CONTENT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence in analysis results",
    )
    processing_time: float = Field(ge=0.0, description="Analysis processing time")

    def get_top_tokens(
        self, limit: int = StatisticalConstants.DEFAULT_TOP_ITEMS_LIMIT
    ) -> List[tuple]:
        """Get most frequent tokens"""
        sorted_tokens = sorted(
            self.token_frequencies.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_tokens[:limit]

    def calculate_readability_score(self) -> float:
        """Calculate approximate readability score"""
        if self.average_document_length == 0:
            return MathematicalFoundationConstants.ZERO_THRESHOLD

        # Simple heuristic based on document length and vocabulary diversity
        vocab_diversity = self.vocabulary_size / max(self.total_tokens, 1)
        length_factor = min(
            self.average_document_length
            / StatisticalConstants.REFERENCE_DOCUMENT_LENGTH,
            MathematicalFoundationConstants.PERFECT_SCORE,
        )

        return vocab_diversity * (1 - self.technical_term_density) * length_factor


# =============================================================================
# PATTERN LEARNING MODELS
# =============================================================================


@dataclass
class LearnedPattern:
    """Pattern learned from domain analysis"""

    pattern_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence_score: float
    usage_count: int = 0


@dataclass
class ExtractedPatterns:
    """Collection of extracted patterns"""

    domain: str
    entity_patterns: List[LearnedPattern]
    relationship_patterns: List[LearnedPattern]
    linguistic_patterns: List[LearnedPattern]
    extraction_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.extraction_metadata is None:
            self.extraction_metadata = {}


# =============================================================================
# DOMAIN INTELLIGENCE MODELS
# =============================================================================


@dataclass
class UnifiedAnalysis:
    """Unified content analysis result"""

    complexity_score: float
    technical_terms: List[str]
    domain_indicators: List[str]
    quality_metrics: Dict[str, float]
    processing_recommendations: Dict[str, Any] = None

    def __post_init__(self):
        if self.processing_recommendations is None:
            self.processing_recommendations = {}


@dataclass
class LLMExtraction:
    """LLM extraction results for configuration generation"""

    domain_characteristics: List[str]
    key_concepts: List[str]
    entity_types: List[str]
    relationship_patterns: List[str]
    processing_complexity: str
    reasoning_quality: float = ExtractionQualityConstants.DEFAULT_REASONING_QUALITY
