"""
Cleaned Universal RAG Models
Legacy maintenance models cleaned up for Universal RAG system.
All hardcoded enums and domain-specific classes removed.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

# Import universal models for legacy compatibility
from core.models.universal_models import (
    UniversalEntity, UniversalRelation, UniversalDocument,
    UniversalQueryAnalysis, UniversalEnhancedQuery,
    UniversalSearchResult, UniversalRAGResponse
)


class QueryType(str, Enum):
    """Universal query categories (domain-agnostic)"""
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURAL = "procedural"
    PREVENTIVE = "preventive"
    INFORMATIONAL = "informational"
    SAFETY = "safety"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    CLASSIFICATION = "classification"


@dataclass
class QueryAnalysis:
    """Query analysis result (universal)"""

    original_query: str
    query_type: QueryType
    keywords: List[str]
    concepts: List[str]
    intent: str
    complexity: str = "medium"
    domain_indicators: List[str] = field(default_factory=list)
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "keywords": self.keywords,
            "concepts": self.concepts,
            "intent": self.intent,
            "complexity": self.complexity,
            "domain_indicators": self.domain_indicators,
            "confidence": self.confidence
        }


@dataclass
class EnhancedQuery:
    """Enhanced query with universal analysis"""

    analysis: QueryAnalysis
    expanded_concepts: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    domain_context: Dict[str, Any] = field(default_factory=dict)
    structured_search: Dict[str, Any] = field(default_factory=dict)
    safety_considerations: List[str] = field(default_factory=list)
    safety_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "analysis": self.analysis.to_dict(),
            "expanded_concepts": self.expanded_concepts,
            "related_entities": self.related_entities,
            "domain_context": self.domain_context,
            "structured_search": self.structured_search,
            "safety_considerations": self.safety_considerations,
            "safety_critical": self.safety_critical
        }


@dataclass
class SearchResult:
    """Universal search result"""

    doc_id: str
    title: str
    content: str
    score: float
    source: str  # 'vector', 'entity', 'graph', 'universal'
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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
    """Universal RAG system response"""

    query: str
    enhanced_query: EnhancedQuery
    search_results: List[SearchResult]
    generated_response: str
    confidence_score: float
    processing_time: float
    sources: List[str]
    domain: str = "general"
    safety_warnings: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "enhanced_query": self.enhanced_query.to_dict(),
            "search_results": [result.to_dict() for result in self.search_results],
            "generated_response": self.generated_response,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "sources": self.sources,
            "domain": self.domain,
            "safety_warnings": self.safety_warnings,
            "citations": self.citations
        }


# ================================
# LEGACY COMPATIBILITY ALIASES
# ================================
# These aliases allow existing code to continue working while we complete the migration
# All old domain-specific classes now point to universal equivalents

# Entity and relation aliases
MaintenanceEntity = UniversalEntity
MaintenanceRelation = UniversalRelation
MaintenanceDocument = UniversalDocument

# Query analysis aliases
MaintenanceQueryAnalysis = UniversalQueryAnalysis
MaintenanceEnhancedQuery = UniversalEnhancedQuery

# Search result aliases
MaintenanceSearchResult = UniversalSearchResult
MaintenanceRAGResponse = UniversalRAGResponse

# For backward compatibility with enum usage
# Since we removed hardcoded enums, provide dynamic type creation functions
def create_entity_type(type_name: str) -> str:
    """Create dynamic entity type (replaces EntityType enum)"""
    return type_name.lower().replace(' ', '_')

def create_relation_type(type_name: str) -> str:
    """Create dynamic relation type (replaces RelationType enum)"""
    return type_name.lower().replace(' ', '_')

# Common entity types for backward compatibility (no longer hardcoded)
COMMON_ENTITY_TYPES = {
    "PHYSICAL_OBJECT": "physical_object",
    "STATE": "state",
    "PROCESS": "process",
    "ACTIVITY": "activity",
    "PROPERTY": "property",
    "CONCEPT": "concept"
}

# Common relation types for backward compatibility (no longer hardcoded)
COMMON_RELATION_TYPES = {
    "HAS_PART": "has_part",
    "HAS_PROPERTY": "has_property",
    "IS_A": "is_a",
    "CONTAINS": "contains",
    "RELATES_TO": "relates_to"
}


def get_entity_type(type_name: str) -> str:
    """Get entity type dynamically (replaces EntityType enum lookup)"""
    return COMMON_ENTITY_TYPES.get(type_name.upper(), create_entity_type(type_name))

def get_relation_type(type_name: str) -> str:
    """Get relation type dynamically (replaces RelationType enum lookup)"""
    return COMMON_RELATION_TYPES.get(type_name.upper(), create_relation_type(type_name))


# ================================
# MIGRATION HELPER FUNCTIONS
# ================================

def convert_legacy_entity(legacy_entity: Dict[str, Any]) -> UniversalEntity:
    """Convert legacy entity dict to UniversalEntity"""
    return UniversalEntity(
        entity_id=legacy_entity.get("entity_id", ""),
        text=legacy_entity.get("text", ""),
        entity_type=str(legacy_entity.get("entity_type", "unknown")),
        confidence=legacy_entity.get("confidence", 1.0),
        context=legacy_entity.get("context"),
        metadata=legacy_entity.get("metadata", {})
    )

def convert_legacy_relation(legacy_relation: Dict[str, Any]) -> UniversalRelation:
    """Convert legacy relation dict to UniversalRelation"""
    return UniversalRelation(
        relation_id=legacy_relation.get("relation_id", ""),
        source_entity_id=legacy_relation.get("source_entity", ""),
        target_entity_id=legacy_relation.get("target_entity", ""),
        relation_type=str(legacy_relation.get("relation_type", "unknown")),
        confidence=legacy_relation.get("confidence", 1.0),
        context=legacy_relation.get("context"),
        metadata=legacy_relation.get("metadata", {})
    )

def convert_legacy_document(legacy_document: Dict[str, Any]) -> UniversalDocument:
    """Convert legacy document dict to UniversalDocument"""
    universal_doc = UniversalDocument(
        document_id=legacy_document.get("doc_id", ""),
        text=legacy_document.get("text", ""),
        title=legacy_document.get("title", ""),
        metadata=legacy_document.get("metadata", {})
    )

    # Add entities and relations if present
    for entity_data in legacy_document.get("entities", []):
        if isinstance(entity_data, dict):
            entity = convert_legacy_entity(entity_data)
            universal_doc.add_entity(entity)

    for relation_data in legacy_document.get("relations", []):
        if isinstance(relation_data, dict):
            relation = convert_legacy_relation(relation_data)
            universal_doc.add_relation(relation)

    return universal_doc


# ================================
# DEPRECATION WARNINGS (OPTIONAL)
# ================================

import warnings

def deprecated_entity_type_enum_warning():
    """Warn about deprecated EntityType enum usage"""
    warnings.warn(
        "EntityType enum is deprecated. Use dynamic string types instead. "
        "See Universal RAG documentation for migration guide.",
        DeprecationWarning,
        stacklevel=3
    )

def deprecated_relation_type_enum_warning():
    """Warn about deprecated RelationType enum usage"""
    warnings.warn(
        "RelationType enum is deprecated. Use dynamic string types instead. "
        "See Universal RAG documentation for migration guide.",
        DeprecationWarning,
        stacklevel=3
    )
