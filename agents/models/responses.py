"""
PydanticAI Agent Response Models

Structured output models for Azure RAG system using Pydantic V2 BaseModel
with comprehensive validation and SLA enforcement.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

logger = logging.getLogger(__name__)


class SearchResultType(str, Enum):
    """Type of search result"""

    VECTOR = "vector"
    GRAPH = "graph"
    GNN = "gnn"
    HYBRID = "hybrid"


class ConfidenceLevel(str, Enum):
    """Confidence level categories"""

    HIGH = "high"  # >= 0.8
    MEDIUM = "medium"  # 0.6-0.8
    LOW = "low"  # < 0.6


class SearchDocument(BaseModel):
    """Individual search result document with V2 validation"""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="allow",  # Allow extra metadata fields
        validate_assignment=True,
    )

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content text")
    title: Optional[str] = Field(None, description="Document title")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    source_type: SearchResultType = Field(
        ..., description="Type of search that found this document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional document metadata"
    )

    @computed_field
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Calculate confidence level from score"""
        if self.score >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class GraphEntity(BaseModel):
    """Graph search entity result"""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    id: str = Field(..., description="Entity identifier")
    label: str = Field(..., description="Entity label/name")
    type: str = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(..., ge=0.0, le=1.0)


class GraphRelationship(BaseModel):
    """Graph search relationship result"""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    id: str = Field(..., description="Relationship identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0, description="Relationship weight")


class TriModalSearchResult(BaseModel):
    """
    Structured tri-modal search result with V2 validation and competitive advantage tracking.

    This is the main output model for our competitive advantage:
    Vector + Graph + GNN search with sub-3s response time guarantee.
    Enhanced with Pydantic V2 for superior validation and performance.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    query: str = Field(..., description="Original query text")
    domain: str = Field(..., description="Detected or specified domain")

    # Vector search results
    vector_results: List[SearchDocument] = Field(
        default_factory=list, description="Vector similarity search results"
    )

    # Graph search results
    graph_entities: List[GraphEntity] = Field(
        default_factory=list, description="Graph entities found"
    )
    graph_relationships: List[GraphRelationship] = Field(
        default_factory=list, description="Graph relationships found"
    )

    # GNN search results (enhanced by graph neural networks)
    gnn_results: List[SearchDocument] = Field(
        default_factory=list, description="GNN-enhanced search results"
    )

    # Consolidated results (our competitive advantage)
    consolidated_results: List[SearchDocument] = Field(
        default_factory=list,
        description="Intelligently merged results from all search types",
    )

    # Performance metrics (SLA compliance)
    execution_time: float = Field(
        ..., ge=0.0, description="Total execution time in seconds"
    )
    search_types_used: List[SearchResultType] = Field(
        ..., description="Search types that were executed"
    )

    # Quality metrics
    total_results: int = Field(..., ge=0, description="Total number of results found")
    confidence_distribution: Dict[ConfidenceLevel, int] = Field(
        default_factory=dict, description="Distribution of results by confidence level"
    )

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(..., description="Whether the search completed successfully")
    error_message: Optional[str] = Field(
        None, description="Error message if search failed"
    )

    # V2 SLA and competitive advantage validation
    @model_validator(mode="after")
    def validate_competitive_advantages(self) -> "TriModalSearchResult":
        """V2 validation for SLA compliance and competitive advantage metrics"""

        # SLA Compliance Validation (sub-3s response time)
        if self.success and self.execution_time > 3.0:
            logger.warning(
                f"SLA VIOLATION: Tri-modal search took {self.execution_time:.2f}s (target: <3.0s)",
                extra={
                    "query": self.query[:100],
                    "domain": self.domain,
                    "execution_time": self.execution_time,
                },
            )

        # Competitive Advantage Validation
        modalities_used = len(self.search_types_used)
        if self.success and modalities_used < 2:
            logger.warning(
                f"Competitive advantage compromised: Only {modalities_used} search modalities used"
            )

        # Auto-calculate confidence distribution for competitive metrics
        if self.consolidated_results and not self.confidence_distribution:
            self.confidence_distribution = self._calculate_confidence_distribution()

        # Auto-calculate total results if not set
        if self.total_results == 0 and self.consolidated_results:
            self.total_results = len(self.consolidated_results)

        return self

    def _calculate_confidence_distribution(self) -> Dict[ConfidenceLevel, int]:
        """Calculate confidence distribution from consolidated results"""
        distribution = {
            ConfidenceLevel.HIGH: 0,
            ConfidenceLevel.MEDIUM: 0,
            ConfidenceLevel.LOW: 0,
        }

        for doc in self.consolidated_results:
            distribution[doc.confidence_level] += 1

        return distribution

    @computed_field
    @property
    def meets_sla(self) -> bool:
        """Check if result meets SLA requirements"""
        return self.execution_time < 3.0 and self.success

    @computed_field
    @property
    def high_confidence_count(self) -> int:
        """Count of high confidence results"""
        return sum(1 for doc in self.consolidated_results if doc.score >= 0.8)

    @computed_field
    @property
    def competitive_advantage_score(self) -> float:
        """Calculate competitive advantage score based on tri-modal performance"""
        if not self.success:
            return 0.0

        # Base score from modality usage (max 0.4)
        modality_score = min(len(self.search_types_used) / 3.0, 1.0) * 0.4

        # Performance score (max 0.3)
        performance_score = max(0, (3.0 - self.execution_time) / 3.0) * 0.3

        # Quality score from high confidence results (max 0.3)
        quality_score = 0.0
        if self.total_results > 0:
            quality_score = (self.high_confidence_count / self.total_results) * 0.3

        return modality_score + performance_score + quality_score

    @computed_field
    @property
    def tri_modal_unity_achieved(self) -> bool:
        """Check if true tri-modal unity was achieved"""
        required_modalities = {"vector", "graph", "gnn"}
        used_modalities = {str(modality).lower() for modality in self.search_types_used}
        return required_modalities.issubset(used_modalities)


class DomainDetectionResult(BaseModel):
    """Structured domain detection result with confidence scoring"""

    text: str = Field(..., description="Input text that was analyzed")
    detected_domain: str = Field(..., description="Primary detected domain")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )

    # Detailed domain probabilities
    domain_probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Probability distribution across all domains"
    )

    # Analysis metadata
    features_used: List[str] = Field(
        default_factory=list, description="Features used for domain detection"
    )
    execution_time: float = Field(..., ge=0.0, description="Detection time in seconds")
    success: bool = Field(..., description="Whether detection completed successfully")

    @model_validator(mode="after")
    def validate_probabilities(self) -> "DomainDetectionResult":
        """V2 validation for probability distribution"""
        if self.domain_probabilities:
            total = sum(self.domain_probabilities.values())
            if not (0.99 <= total <= 1.01):  # Allow for float precision
                raise ValueError(f"Domain probabilities must sum to 1.0, got {total}")
        return self

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", validate_assignment=True
    )


class PatternLearningResult(BaseModel):
    """Structured pattern learning result with V2 validation"""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", validate_assignment=True
    )

    patterns_learned: int = Field(
        ..., ge=0, description="Number of new patterns learned"
    )
    patterns_updated: int = Field(
        ..., ge=0, description="Number of existing patterns updated"
    )
    confidence_scores: List[float] = Field(
        default_factory=list, description="Confidence scores for learned patterns"
    )

    # Pattern details
    pattern_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of learned patterns"
    )

    # Performance metrics
    learning_time: float = Field(..., ge=0.0, description="Learning time in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")

    success: bool = Field(..., description="Whether learning completed successfully")
    error_message: Optional[str] = Field(
        None, description="Error message if learning failed"
    )

    @computed_field
    @property
    def total_patterns_affected(self) -> int:
        """Total patterns learned or updated"""
        return self.patterns_learned + self.patterns_updated

    @computed_field
    @property
    def average_confidence(self) -> float:
        """Average confidence score of learned patterns"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


class AgentHealthStatus(BaseModel):
    """Agent health and status information with V2 validation"""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", validate_assignment=True
    )

    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Current agent status")
    azure_services_connected: bool = Field(
        ..., description="Azure services connectivity"
    )

    # Performance metrics
    average_response_time: float = Field(
        ..., ge=0.0, description="Average response time in seconds"
    )
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Success rate over recent queries"
    )

    # Service health
    service_health: Dict[str, bool] = Field(
        default_factory=dict, description="Health status of individual services"
    )

    last_health_check: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def meets_performance_sla(self) -> bool:
        """Check if agent meets performance SLA"""
        return self.average_response_time < 3.0 and self.success_rate >= 0.95

    @computed_field
    @property
    def overall_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        if not self.service_health:
            base_score = 0.5
        else:
            healthy_services = sum(
                1 for healthy in self.service_health.values() if healthy
            )
            base_score = healthy_services / len(self.service_health)

        # Factor in performance metrics
        performance_score = min(self.success_rate, 1.0)
        sla_bonus = 0.1 if self.meets_performance_sla else 0.0

        return min((base_score * 0.6) + (performance_score * 0.3) + sla_bonus, 1.0)


# ================================
# UNIVERSAL ENTITY/RELATION MODELS
# ================================
# Consolidated from infra/models/universal_rag_models.py


class UniversalEntity(BaseModel):
    """Universal entity model that works with any domain - no hardcoded types"""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # Allow numpy arrays
    )

    entity_id: str = Field(..., description="Unique entity identifier")
    text: str = Field(..., description="Entity text content")
    entity_type: str = Field(..., description="Dynamic entity type (domain-specific)")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Entity confidence score"
    )
    context: Optional[str] = Field(None, description="Contextual information")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional entity metadata"
    )
    embedding: Optional[np.ndarray] = Field(None, description="Entity embedding vector")

    @model_validator(mode="after")
    def normalize_entity_type(self):
        """Normalize entity type to lowercase with underscores"""
        self.entity_type = self.entity_type.lower().replace(" ", "_")
        return self

    @computed_field
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Calculate confidence level from score"""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class UniversalRelation(BaseModel):
    """Universal relation model that works with any domain - no hardcoded types"""

    model_config = ConfigDict(
        use_enum_values=True, extra="allow", validate_assignment=True
    )

    relation_id: str = Field(..., description="Unique relation identifier")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relation_type: str = Field(
        ..., description="Dynamic relation type (domain-specific)"
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Relation confidence score"
    )
    context: Optional[str] = Field(None, description="Contextual information")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional relation metadata"
    )

    @model_validator(mode="after")
    def normalize_relation_type(self):
        """Normalize relation type to lowercase with underscores"""
        self.relation_type = self.relation_type.lower().replace(" ", "_")
        return self

    @computed_field
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Calculate confidence level from score"""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction operations"""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", validate_assignment=True
    )

    entities: List[UniversalEntity] = Field(
        default_factory=list, description="List of extracted entities"
    )
    execution_time: float = Field(..., ge=0.0, description="Extraction time in seconds")
    source_content: str = Field(..., description="Original content that was processed")
    extraction_method: str = Field(..., description="Method used for extraction")
    success: bool = Field(..., description="Whether extraction completed successfully")
    error_message: Optional[str] = Field(
        None, description="Error message if extraction failed"
    )

    @computed_field
    @property
    def entity_count(self) -> int:
        """Total number of entities extracted"""
        return len(self.entities)

    @computed_field
    @property
    def high_confidence_entities(self) -> List[UniversalEntity]:
        """Entities with high confidence scores"""
        return [entity for entity in self.entities if entity.confidence >= 0.8]


class RelationshipExtractionResponse(BaseModel):
    """Response model for relationship extraction operations"""

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", validate_assignment=True
    )

    relationships: List[UniversalRelation] = Field(
        default_factory=list, description="List of extracted relationships"
    )
    execution_time: float = Field(..., ge=0.0, description="Extraction time in seconds")
    source_content: str = Field(..., description="Original content that was processed")
    extraction_method: str = Field(..., description="Method used for extraction")
    success: bool = Field(..., description="Whether extraction completed successfully")
    error_message: Optional[str] = Field(
        None, description="Error message if extraction failed"
    )

    @computed_field
    @property
    def relationship_count(self) -> int:
        """Total number of relationships extracted"""
        return len(self.relationships)

    @computed_field
    @property
    def high_confidence_relationships(self) -> List[UniversalRelation]:
        """Relationships with high confidence scores"""
        return [rel for rel in self.relationships if rel.confidence >= 0.8]


class AgentResponse(BaseModel):
    """Generic agent response model for PydanticAI integration"""

    model_config = ConfigDict(
        use_enum_values=True, extra="allow", validate_assignment=True
    )

    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique response identifier",
    )
    agent_id: str = Field(..., description="Agent that generated the response")
    content: Any = Field(..., description="Response content (flexible type)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    success: bool = Field(
        ..., description="Whether the operation completed successfully"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if operation failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    @computed_field
    @property
    def meets_sla(self) -> bool:
        """Check if response meets SLA requirements"""
        return self.execution_time < 3.0 and self.success
