"""Universal data models for RAG system.

This package contains domain-agnostic data models that work across all domains.
All models are configuration-driven and contain no hardcoded domain assumptions.
"""

from .entities import Entity, EntityType
from .relations import Relation, RelationType
from .graphs import KnowledgeGraph, GraphNode, GraphEdge
from .queries import Query, QueryType, QueryIntent
from .responses import Response, ResponseContext, ResponseMetadata

__all__ = [
    'Entity', 'EntityType',
    'Relation', 'RelationType',
    'KnowledgeGraph', 'GraphNode', 'GraphEdge',
    'Query', 'QueryType', 'QueryIntent',
    'Response', 'ResponseContext', 'ResponseMetadata'
]