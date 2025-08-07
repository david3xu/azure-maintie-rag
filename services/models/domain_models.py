"""
Simple Domain Models - CODING_STANDARDS Compliant
Clean data models without over-engineering enterprise patterns.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QueryRequest:
    """Simple query request model"""

    query: str
    domain: str = "general"
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None


@dataclass
class QueryResponse:
    """Simple query response model"""

    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class DocumentData:
    """Simple document data model"""

    id: str
    content: str
    title: str = ""
    domain: str = "general"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EntityData:
    """Simple entity data model"""

    id: str
    text: str
    entity_type: str
    confidence: float = 1.0
    domain: str = "general"


@dataclass
class RelationshipData:
    """Simple relationship data model"""

    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float = 1.0
    domain: str = "general"


@dataclass
class SearchResult:
    """Simple search result model"""

    id: str
    title: str
    content: str
    score: float
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingStatus:
    """Simple processing status model"""

    status: str  # "pending", "processing", "completed", "error"
    progress: float = 0.0
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class ServiceHealth:
    """Simple service health model"""

    service_name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str = ""
    last_check: Optional[datetime] = None


@dataclass
class CacheEntry:
    """Simple cache entry model"""

    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class AgentConfig:
    """Simple agent configuration model"""

    agent_name: str
    agent_type: str
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowState:
    """Simple workflow state model"""

    workflow_id: str
    status: str
    current_step: str
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelFactory:
    """Simple factory for creating model instances"""

    @staticmethod
    def create_query_request(query: str, **kwargs) -> QueryRequest:
        """Create query request with defaults"""
        return QueryRequest(query=query, **kwargs)

    @staticmethod
    def create_success_response(
        query: str, results: List[Dict[str, Any]], **kwargs
    ) -> QueryResponse:
        """Create successful query response"""
        return QueryResponse(
            success=True,
            query=query,
            results=results,
            total_count=len(results),
            **kwargs,
        )

    @staticmethod
    def create_error_response(query: str, error: str, **kwargs) -> QueryResponse:
        """Create error query response"""
        return QueryResponse(
            success=False, query=query, results=[], total_count=0, error=error, **kwargs
        )

    @staticmethod
    def create_document(doc_id: str, content: str, **kwargs) -> DocumentData:
        """Create document with defaults"""
        return DocumentData(id=doc_id, content=content, **kwargs)

    @staticmethod
    def create_entity(
        entity_id: str, text: str, entity_type: str, **kwargs
    ) -> EntityData:
        """Create entity with defaults"""
        return EntityData(id=entity_id, text=text, entity_type=entity_type, **kwargs)

    @staticmethod
    def create_relationship(
        rel_id: str, source: str, target: str, rel_type: str, **kwargs
    ) -> RelationshipData:
        """Create relationship with defaults"""
        return RelationshipData(
            id=rel_id,
            source_entity=source,
            target_entity=target,
            relation_type=rel_type,
            **kwargs,
        )

    @staticmethod
    def create_search_result(
        result_id: str, title: str, content: str, score: float, **kwargs
    ) -> SearchResult:
        """Create search result with defaults"""
        return SearchResult(
            id=result_id, title=title, content=content, score=score, **kwargs
        )


# Backward compatibility functions
def create_query_request(**kwargs) -> QueryRequest:
    """Backward compatibility function"""
    return ModelFactory.create_query_request(**kwargs)


def create_query_response(**kwargs) -> QueryResponse:
    """Backward compatibility function"""
    if kwargs.get("success", True) and "error" not in kwargs:
        return ModelFactory.create_success_response(**kwargs)
    else:
        return ModelFactory.create_error_response(**kwargs)


def create_document(**kwargs) -> DocumentData:
    """Backward compatibility function"""
    return ModelFactory.create_document(**kwargs)


def create_entity(**kwargs) -> EntityData:
    """Backward compatibility function"""
    return ModelFactory.create_entity(**kwargs)


def create_relationship(**kwargs) -> RelationshipData:
    """Backward compatibility function"""
    return ModelFactory.create_relationship(**kwargs)


# Backward compatibility aliases
QueryModel = QueryRequest
ResponseModel = QueryResponse
DocumentModel = DocumentData
EntityModel = EntityData
RelationshipModel = RelationshipData
