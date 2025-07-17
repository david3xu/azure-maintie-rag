"""
Universal Data Models for Universal RAG
Replaces domain-specific models with flexible, dynamic models that work with any domain
No hardcoded entity types, relation types, or domain assumptions
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np


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


class QueryType(str, Enum):
    """Universal query types that work across domains"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    CLASSIFICATION = "classification"
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


@dataclass
class UniversalSearchResult:
    """Universal search result from any retrieval method"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    source: str = "unknown"

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
            "domain": self.domain
        }


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


# Legacy compatibility aliases for existing code
# These allow gradual migration from domain-specific to universal models
MaintenanceEntity = UniversalEntity
MaintenanceRelation = UniversalRelation
MaintenanceDocument = UniversalDocument
QueryAnalysis = UniversalQueryAnalysis
EnhancedQuery = UniversalEnhancedQuery
SearchResult = UniversalSearchResult
RAGResponse = UniversalRAGResponse


def create_sample_universal_data(domain: str = "general") -> Dict[str, Any]:
    """Create sample universal data for testing"""

    # Sample entity
    entity = UniversalEntity(
        entity_id="sample_entity_1",
        text="sample concept",
        entity_type="sample_type",
        confidence=0.9,
        metadata={"domain": domain}
    )

    # Sample relation
    relation = UniversalRelation(
        relation_id="sample_relation_1",
        source_entity_id="entity_1",
        target_entity_id="entity_2",
        relation_type="related_to",
        confidence=0.8,
        metadata={"domain": domain}
    )

    # Sample document
    document = UniversalDocument(
        doc_id="sample_doc_1",
        text="This is a sample document for universal RAG testing.",
        title="Sample Document",
        metadata={"domain": domain, "source": "sample"}
    )
    document.add_entity(entity)
    document.add_relation(relation)

    return {
        "entity": entity.to_dict(),
        "relation": relation.to_dict(),
        "document": document.to_dict(),
        "domain": domain
    }