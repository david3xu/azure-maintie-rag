"""
Core data models for MaintIE Enhanced RAG system
Defines fundamental structures for maintenance entities, relations, and documents
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np


class EntityType(str, Enum):
    """MaintIE entity types based on schema"""
    PHYSICAL_OBJECT = "PhysicalObject"
    STATE = "State"
    PROCESS = "Process"
    ACTIVITY = "Activity"
    PROPERTY = "Property"
    PROBLEM = "Problem"
    # Add hierarchy types from scheme.json
    SUBSTANCE = "PhysicalObject/Substance"
    GAS = "PhysicalObject/Substance/Gas"
    LIQUID = "PhysicalObject/Substance/Liquid"
    SOLID = "PhysicalObject/Substance/Solid"
    MIXTURE = "PhysicalObject/Substance/Mixture"
    ORGANISM = "PhysicalObject/Organism"
    PERSON = "PhysicalObject/Organism/Person"
    SENSING_OBJECT = "PhysicalObject/SensingObject"
    STORING_OBJECT = "PhysicalObject/StoringObject"
    EMITTING_OBJECT = "PhysicalObject/EmittingObject"
    PROTECTING_OBJECT = "PhysicalObject/ProtectingObject"
    GENERATING_OBJECT = "PhysicalObject/GeneratingObject"
    MATTER_PROCESSING_OBJECT = "PhysicalObject/MatterProcessingObject"
    INFORMATION_PROCESSING_OBJECT = "PhysicalObject/InformationProcessingObject"
    DRIVING_OBJECT = "PhysicalObject/DrivingObject"
    COVERING_OBJECT = "PhysicalObject/CoveringObject"
    PRESENTING_OBJECT = "PhysicalObject/PresentingObject"
    CONTROLLING_OBJECT = "PhysicalObject/ControllingObject"
    RESTRICTING_OBJECT = "PhysicalObject/RestrictingObject"
    HUMAN_INTERACTION_OBJECT = "PhysicalObject/HumanInteractionObject"
    TRANSFORMING_OBJECT = "PhysicalObject/TransformingObject"
    HOLDING_OBJECT = "PhysicalObject/HoldingObject"
    GUIDING_OBJECT = "PhysicalObject/GuidingObject"
    INTERFACING_OBJECT = "PhysicalObject/InterfacingObject"
    DESIRABLE_STATE = "State/DesirableState"
    NORMAL_STATE = "State/DesirableState/NormalState"
    UNDESIRABLE_STATE = "State/UndesirableState"
    DEGRADED_STATE = "State/UndesirableState/DegradedState"
    FAILED_STATE = "State/UndesirableState/FailedState"
    DESIRABLE_PROCESS = "Process/DesirableProcess"
    UNDESIRABLE_PROCESS = "Process/UndesirableProcess"
    DESIRABLE_PROPERTY = "Property/DesirableProperty"
    UNDESIRABLE_PROPERTY = "Property/UndesirableProperty"
    MAINTENANCE_ACTIVITY = "Activity/MaintenanceActivity"
    ADJUST = "Activity/MaintenanceActivity/Adjust"
    CALIBRATE = "Activity/MaintenanceActivity/Calibrate"
    DIAGNOSE = "Activity/MaintenanceActivity/Diagnose"
    INSPECT = "Activity/MaintenanceActivity/Inspect"
    REPLACE = "Activity/MaintenanceActivity/Replace"
    REPAIR = "Activity/MaintenanceActivity/Repair"
    SERVICE = "Activity/MaintenanceActivity/Service"
    SUPPORTING_ACTIVITY = "Activity/SupportingActivity"
    ADMIN = "Activity/SupportingActivity/Admin"
    ASSEMBLE = "Activity/SupportingActivity/Assemble"
    ISOLATE = "Activity/SupportingActivity/Isolate"
    MEASURE = "Activity/SupportingActivity/Measure"
    MODIFY = "Activity/SupportingActivity/Modify"
    MOVE = "Activity/SupportingActivity/Move"
    OPERATE = "Activity/SupportingActivity/Operate"
    PERFORM = "Activity/SupportingActivity/Perform"
    TEAMWORK = "Activity/SupportingActivity/Teamwork"


class RelationType(str, Enum):
    """MaintIE relation types based on schema"""
    HAS_PART = "hasPart"
    HAS_PROPERTY = "hasProperty"
    # Add types from scheme.json
    IS_A = "isA"
    CONTAINS = "contains"
    HAS_PARTICIPANT = "hasParticipant"
    HAS_PATIENT = "hasParticipant/hasPatient"
    HAS_AGENT = "hasParticipant/hasAgent"


class QueryType(str, Enum):
    """Maintenance query categories"""
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURAL = "procedural"
    PREVENTIVE = "preventive"
    INFORMATIONAL = "informational"
    SAFETY = "safety"


@dataclass
class MaintenanceEntity:
    """Core maintenance entity from MaintIE annotations"""

    entity_id: str
    text: str
    entity_type: EntityType
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate entity after creation"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaintenanceEntity':
        """Create entity from dictionary"""
        return cls(
            entity_id=data["entity_id"],
            text=data["text"],
            entity_type=EntityType(data["entity_type"]),
            confidence=data.get("confidence", 1.0),
            context=data.get("context"),
            metadata=data.get("metadata", {})
        )


@dataclass
class MaintenanceRelation:
    """Relationship between maintenance entities"""

    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: RelationType
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary"""
        return {
            "relation_id": self.relation_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaintenanceRelation':
        """Create relation from dictionary"""
        return cls(
            relation_id=data["relation_id"],
            source_entity=data["source_entity"],
            target_entity=data["target_entity"],
            relation_type=RelationType(data["relation_type"]),
            confidence=data.get("confidence", 1.0),
            context=data.get("context"),
            metadata=data.get("metadata", {})
        )


@dataclass
class MaintenanceDocument:
    """Single maintenance work order or document"""

    doc_id: str
    text: str
    title: Optional[str] = None
    entities: List[MaintenanceEntity] = field(default_factory=list)
    relations: List[MaintenanceRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_entity(self, entity: MaintenanceEntity) -> None:
        """Add entity to document"""
        self.entities.append(entity)

    def add_relation(self, relation: MaintenanceRelation) -> None:
        """Add relation to document"""
        self.relations.append(relation)

    def get_entity_texts(self) -> List[str]:
        """Get all entity texts in document"""
        return [entity.text for entity in self.entities]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class QueryAnalysis:
    """Results of query analysis"""

    original_query: str
    query_type: QueryType
    entities: List[str]
    intent: str
    complexity: str
    urgency: str = "medium"
    equipment_category: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "entities": self.entities,
            "intent": self.intent,
            "complexity": self.complexity,
            "urgency": self.urgency,
            "equipment_category": self.equipment_category,
            "confidence": self.confidence
        }


@dataclass
class EnhancedQuery:
    """Enhanced query with expanded concepts"""

    analysis: QueryAnalysis
    expanded_concepts: List[str]
    related_entities: List[str]
    domain_context: Dict[str, Any]
    structured_search: str
    safety_considerations: List[str] = field(default_factory=list)
    safety_critical: bool = False
    safety_warnings: List[str] = field(default_factory=list)
    equipment_category: Optional[str] = None
    maintenance_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert enhanced query to dictionary"""
        return {
            "analysis": self.analysis.to_dict(),
            "expanded_concepts": self.expanded_concepts,
            "related_entities": self.related_entities,
            "domain_context": self.domain_context,
            "structured_search": self.structured_search,
            "safety_considerations": self.safety_considerations,
            "safety_critical": self.safety_critical,
            "safety_warnings": self.safety_warnings,
            "equipment_category": self.equipment_category,
            "maintenance_context": self.maintenance_context
        }


@dataclass
class SearchResult:
    """Individual search result"""

    doc_id: str
    title: str
    content: str
    score: float
    source: str  # 'vector', 'entity', 'graph'
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "entities": self.entities
        }


@dataclass
class RAGResponse:
    """Complete RAG system response"""

    query: str
    enhanced_query: EnhancedQuery
    search_results: List[SearchResult]
    generated_response: str
    confidence_score: float
    processing_time: float
    sources: List[str]
    safety_warnings: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RAG response to dictionary"""
        return {
            "query": self.query,
            "enhanced_query": self.enhanced_query.to_dict(),
            "search_results": [result.to_dict() for result in self.search_results],
            "generated_response": self.generated_response,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "sources": self.sources,
            "safety_warnings": self.safety_warnings,
            "citations": self.citations
        }
