"""
Universal RAG Models - Clean Implementation
Single source of truth for all Universal RAG data models.
Works with any domain without hardcoded assumptions.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np


# ================================
# CORE UNIVERSAL MODELS
# ================================

@dataclass
class UniversalEntity:
    """Universal entity that works with any domain - no hardcoded types"""

    entity_id: str
    text: str
    entity_type: str  # Dynamic string instead of hardcoded enum
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate entity after creation"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")

        # Normalize entity type to lowercase with underscores
        self.entity_type = self.entity_type.lower().replace(' ', '_')

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalEntity':
        """Create entity from dictionary"""
        return cls(
            entity_id=data["entity_id"],
            text=data["text"],
            entity_type=data["entity_type"],
            confidence=data.get("confidence", 1.0),
            context=data.get("context"),
            metadata=data.get("metadata", {})
        )


@dataclass
class UniversalRelation:
    """Universal relation that works with any domain - no hardcoded types"""

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str  # Dynamic string instead of hardcoded enum
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate relation after creation"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")

        # Normalize relation type to lowercase with underscores
        self.relation_type = self.relation_type.lower().replace(' ', '_')

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary"""
        return {
            "relation_id": self.relation_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalRelation':
        """Create relation from dictionary"""
        return cls(
            relation_id=data["relation_id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            relation_type=data["relation_type"],
            confidence=data.get("confidence", 1.0),
            context=data.get("context"),
            metadata=data.get("metadata", {})
        )


@dataclass
class UniversalDocument:
    """Universal document that works with any domain and content type"""

    doc_id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, UniversalEntity] = field(default_factory=dict)
    relations: List[UniversalRelation] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def add_entity(self, entity: UniversalEntity):
        """Add entity to document"""
        self.entities[entity.entity_id] = entity

    def add_relation(self, relation: UniversalRelation):
        """Add relation to document"""
        self.relations.append(relation)

    def get_entity_texts(self) -> List[str]:
        """Get all entity texts in document"""
        return [entity.text for entity in self.entities.values()]

    def get_relation_summary(self) -> Dict[str, int]:
        """Get summary of relation types in document"""
        relation_counts = {}
        for relation in self.relations:
            relation_type = relation.relation_type
            relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1
        return relation_counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "metadata": self.metadata,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relations": [relation.to_dict() for relation in self.relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalDocument':
        """Create document from dictionary"""
        doc = cls(
            doc_id=data["doc_id"],
            text=data["text"],
            title=data.get("title"),
            metadata=data.get("metadata", {})
        )

        # Load entities
        for entity_data in data.get("entities", []):
            entity = UniversalEntity.from_dict(entity_data)
            doc.add_entity(entity)

        # Load relations
        for relation_data in data.get("relations", []):
            relation = UniversalRelation.from_dict(relation_data)
            doc.add_relation(relation)

        return doc


# ================================
# QUERY PROCESSING MODELS
# ================================

class QueryType(str, Enum):
    """Universal query types that work across domains"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    CLASSIFICATION = "classification"
    PREVENTIVE = "preventive"
    SAFETY = "safety"
    INFORMATIONAL = "informational"
    UNKNOWN = "unknown"


@dataclass
class UniversalQueryAnalysis:
    """Universal query analysis - no domain assumptions"""

    query_text: str
    query_type: QueryType
    confidence: float
    entities_detected: List[str] = field(default_factory=list)
    concepts_detected: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    complexity: str = "medium"  # simple, medium, complex
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        return {
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "entities_detected": self.entities_detected,
            "concepts_detected": self.concepts_detected,
            "intent": self.intent,
            "complexity": self.complexity,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalQueryAnalysis':
        """Create analysis from dictionary"""
        return cls(
            query_text=data["query_text"],
            query_type=QueryType(data["query_type"]),
            confidence=data["confidence"],
            entities_detected=data.get("entities_detected", []),
            concepts_detected=data.get("concepts_detected", []),
            intent=data.get("intent"),
            complexity=data.get("complexity", "medium"),
            metadata=data.get("metadata", {})
        )


@dataclass
class UniversalEnhancedQuery:
    """Universal enhanced query with expanded concepts"""

    original_query: str
    expanded_concepts: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    query_analysis: Optional[UniversalQueryAnalysis] = None
    search_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert enhanced query to dictionary"""
        return {
            "original_query": self.original_query,
            "expanded_concepts": self.expanded_concepts,
            "related_entities": self.related_entities,
            "query_analysis": self.query_analysis.to_dict() if self.query_analysis else None,
            "search_terms": self.search_terms,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalEnhancedQuery':
        """Create enhanced query from dictionary"""
        query_analysis = None
        if data.get("query_analysis"):
            query_analysis = UniversalQueryAnalysis.from_dict(data["query_analysis"])

        return cls(
            original_query=data["original_query"],
            expanded_concepts=data.get("expanded_concepts", []),
            related_entities=data.get("related_entities", []),
            query_analysis=query_analysis,
            search_terms=data.get("search_terms", []),
            metadata=data.get("metadata", {})
        )


# ================================
# SEARCH AND RESPONSE MODELS
# ================================

@dataclass
class UniversalSearchResult:
    """Universal search result from any retrieval method"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    source: str = "unknown"  # 'vector', 'entity', 'graph', 'universal'

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "entities": self.entities,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalSearchResult':
        """Create search result from dictionary"""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            score=data["score"],
            metadata=data.get("metadata", {}),
            entities=data.get("entities", []),
            source=data.get("source", "unknown")
        )


@dataclass
class UniversalRAGResponse:
    """Universal RAG response that works with any domain"""

    query: str
    answer: str
    confidence: float
    sources: List[UniversalSearchResult] = field(default_factory=list)
    entities_used: List[str] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    domain: str = "general"
    safety_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": [source.to_dict() for source in self.sources],
            "entities_used": self.entities_used,
            "processing_metadata": self.processing_metadata,
            "citations": self.citations,
            "domain": self.domain,
            "safety_warnings": self.safety_warnings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalRAGResponse':
        """Create response from dictionary"""
        sources = [UniversalSearchResult.from_dict(s) for s in data.get("sources", [])]

        return cls(
            query=data["query"],
            answer=data["answer"],
            confidence=data["confidence"],
            sources=sources,
            entities_used=data.get("entities_used", []),
            processing_metadata=data.get("processing_metadata", {}),
            citations=data.get("citations", []),
            domain=data.get("domain", "general"),
            safety_warnings=data.get("safety_warnings", [])
        )


# ================================
# KNOWLEDGE GRAPH MODELS
# ================================

@dataclass
class UniversalKnowledgeGraph:
    """Universal knowledge graph metadata"""

    domain: str
    entities: Dict[str, UniversalEntity] = field(default_factory=dict)
    relations: List[UniversalRelation] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_entity(self, entity: UniversalEntity):
        """Add entity to knowledge graph"""
        self.entities[entity.entity_id] = entity
        if entity.entity_type not in self.entity_types:
            self.entity_types.append(entity.entity_type)

    def add_relation(self, relation: UniversalRelation):
        """Add relation to knowledge graph"""
        self.relations.append(relation)
        if relation.relation_type not in self.relation_types:
            self.relation_types.append(relation.relation_type)

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "unique_entity_types": len(self.entity_types),
            "unique_relation_types": len(self.relation_types),
            "domain": self.domain,
            "created_at": self.created_at
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge graph to dictionary"""
        return {
            "domain": self.domain,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relations": [relation.to_dict() for relation in self.relations],
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
            "statistics": self.get_statistics(),
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalKnowledgeGraph':
        """Create knowledge graph from dictionary"""
        kg = cls(
            domain=data["domain"],
            entity_types=data.get("entity_types", []),
            relation_types=data.get("relation_types", []),
            statistics=data.get("statistics", {}),
            created_at=data.get("created_at", datetime.now().isoformat())
        )

        # Load entities
        for entity_data in data.get("entities", []):
            entity = UniversalEntity.from_dict(entity_data)
            kg.add_entity(entity)

        # Load relations
        for relation_data in data.get("relations", []):
            relation = UniversalRelation.from_dict(relation_data)
            kg.add_relation(relation)

        return kg


# ================================
# ML TRAINING MODELS
# ================================

@dataclass
class UniversalTrainingConfig:
    """Universal training configuration for ML models"""

    model_type: str  # 'gnn', 'transformer', 'hybrid'
    domain: str
    training_data_path: str
    validation_data_path: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_type": self.model_type,
            "domain": self.domain,
            "training_data_path": self.training_data_path,
            "validation_data_path": self.validation_data_path,
            "model_config": self.model_config,
            "hyperparameters": self.hyperparameters,
            "training_metadata": self.training_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalTrainingConfig':
        """Create config from dictionary"""
        return cls(
            model_type=data["model_type"],
            domain=data["domain"],
            training_data_path=data["training_data_path"],
            validation_data_path=data.get("validation_data_path"),
            model_config=data.get("model_config", {}),
            hyperparameters=data.get("hyperparameters", {}),
            training_metadata=data.get("training_metadata", {})
        )


@dataclass
class UniversalTrainingResult:
    """Universal training result"""

    model_id: str
    model_type: str
    domain: str
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    training_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "domain": self.domain,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "model_path": self.model_path,
            "training_time": self.training_time,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalTrainingResult':
        """Create result from dictionary"""
        return cls(
            model_id=data["model_id"],
            model_type=data["model_type"],
            domain=data["domain"],
            training_metrics=data.get("training_metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            model_path=data.get("model_path"),
            training_time=data.get("training_time"),
            metadata=data.get("metadata", {})
        )


# ================================
# UTILITY FUNCTIONS
# ================================

def create_entity_type(type_name: str) -> str:
    """Create dynamic entity type"""
    return type_name.lower().replace(' ', '_')

def create_relation_type(type_name: str) -> str:
    """Create dynamic relation type"""
    return type_name.lower().replace(' ', '_')

def create_entity_type(type_name: str) -> str:
    """Create dynamic entity type"""
    return type_name.lower().replace(' ', '_')

def create_relation_type(type_name: str) -> str:
    """Create dynamic relation type"""
    return type_name.lower().replace(' ', '_')