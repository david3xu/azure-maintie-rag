"""Universal query models for any domain."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class QueryType(Enum):
    """Universal query types that work across domains."""
    ENTITY_SEARCH = "entity_search"
    RELATION_SEARCH = "relation_search"
    PATH_FINDING = "path_finding"
    SUBGRAPH_EXTRACTION = "subgraph_extraction"
    SIMILARITY_SEARCH = "similarity_search"
    COMPLEX_REASONING = "complex_reasoning"
    FACTUAL_QUESTION = "factual_question"
    PROCEDURAL_QUESTION = "procedural_question"


class QueryIntent(Enum):
    """Universal query intents that work across domains."""
    FIND = "find"
    EXPLAIN = "explain"
    COMPARE = "compare"
    LIST = "list"
    DESCRIBE = "describe"
    ANALYZE = "analyze"
    TROUBLESHOOT = "troubleshoot"
    RECOMMEND = "recommend"
    PREDICT = "predict"
    CLASSIFY = "classify"


@dataclass
class Query:
    """Universal query model for any domain.

    Configuration-driven query that adapts to any domain through
    dynamic parameters and configurable processing.
    """

    # Core query properties
    id: str
    text: str
    type: QueryType
    intent: QueryIntent

    # Query metadata
    domain: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Parsed query components
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Query parameters (domain-specific)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Processing controls
    max_results: int = 10
    confidence_threshold: float = 0.5
    search_depth: int = 2

    # Context and filters
    context: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate query after creation."""
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")

        if self.max_results <= 0:
            raise ValueError("Max results must be positive")

        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        if self.search_depth < 1:
            raise ValueError("Search depth must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'type': self.type.value,
            'intent': self.intent.value,
            'domain': self.domain,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'entities': self.entities,
            'relations': self.relations,
            'keywords': self.keywords,
            'parameters': self.parameters,
            'max_results': self.max_results,
            'confidence_threshold': self.confidence_threshold,
            'search_depth': self.search_depth,
            'context': self.context,
            'filters': self.filters
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create query from dictionary."""
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()

        return cls(
            id=data['id'],
            text=data['text'],
            type=QueryType(data['type']),
            intent=QueryIntent(data['intent']),
            domain=data['domain'],
            timestamp=timestamp,
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            entities=data.get('entities', []),
            relations=data.get('relations', []),
            keywords=data.get('keywords', []),
            parameters=data.get('parameters', {}),
            max_results=data.get('max_results', 10),
            confidence_threshold=data.get('confidence_threshold', 0.5),
            search_depth=data.get('search_depth', 2),
            context=data.get('context', {}),
            filters=data.get('filters', {})
        )

    def add_entity(self, entity: str) -> None:
        """Add entity to query."""
        if entity not in self.entities:
            self.entities.append(entity)

    def add_relation(self, relation: str) -> None:
        """Add relation to query."""
        if relation not in self.relations:
            self.relations.append(relation)

    def add_keyword(self, keyword: str) -> None:
        """Add keyword to query."""
        if keyword not in self.keywords:
            self.keywords.append(keyword)

    def set_parameter(self, key: str, value: Any) -> None:
        """Set query parameter."""
        self.parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get query parameter."""
        return self.parameters.get(key, default)

    def set_filter(self, key: str, value: Any) -> None:
        """Set query filter."""
        self.filters[key] = value

    def get_filter(self, key: str, default: Any = None) -> Any:
        """Get query filter."""
        return self.filters.get(key, default)

    def is_complex(self) -> bool:
        """Check if query requires complex reasoning."""
        complex_intents = {QueryIntent.ANALYZE, QueryIntent.COMPARE, QueryIntent.PREDICT, QueryIntent.TROUBLESHOOT}
        complex_types = {QueryType.COMPLEX_REASONING, QueryType.PATH_FINDING, QueryType.SUBGRAPH_EXTRACTION}

        return (self.intent in complex_intents or
                self.type in complex_types or
                len(self.entities) > 2 or
                self.search_depth > 2)