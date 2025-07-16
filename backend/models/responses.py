"""Universal response models for any domain."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from .entities import Entity
from .relations import Relation


@dataclass
class ResponseContext:
    """Context information for responses."""

    entities_used: List[str] = field(default_factory=list)
    relations_used: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'entities_used': self.entities_used,
            'relations_used': self.relations_used,
            'sources': self.sources,
            'confidence_scores': self.confidence_scores,
            'processing_steps': self.processing_steps
        }


@dataclass
class ResponseMetadata:
    """Metadata for responses."""

    generation_time: datetime = field(default_factory=datetime.now)
    processing_duration: float = 0.0
    model_used: str = ""
    domain: str = ""
    query_id: str = ""
    response_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'generation_time': self.generation_time.isoformat(),
            'processing_duration': self.processing_duration,
            'model_used': self.model_used,
            'domain': self.domain,
            'query_id': self.query_id,
            'response_type': self.response_type
        }


@dataclass
class Response:
    """Universal response model for any domain.

    Configuration-driven response that adapts to any domain through
    dynamic content and configurable formatting.
    """

    # Core response properties
    id: str
    text: str
    confidence: float

    # Retrieved information
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    # Response structure
    sections: Dict[str, str] = field(default_factory=dict)
    bullet_points: List[str] = field(default_factory=list)

    # Supporting information
    context: ResponseContext = field(default_factory=ResponseContext)
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)

    # Additional data (domain-specific)
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate response after creation."""
        if not self.text or not self.text.strip():
            raise ValueError("Response text cannot be empty")

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'confidence': self.confidence,
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'sections': self.sections,
            'bullet_points': self.bullet_points,
            'context': self.context.to_dict(),
            'metadata': self.metadata.to_dict(),
            'additional_data': self.additional_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Response':
        """Create response from dictionary."""
        # Parse entities
        entities = []
        for entity_data in data.get('entities', []):
            from .entities import Entity
            entities.append(Entity.from_dict(entity_data))

        # Parse relations
        relations = []
        for relation_data in data.get('relations', []):
            from .relations import Relation
            relations.append(Relation.from_dict(relation_data))

        # Parse context
        context_data = data.get('context', {})
        context = ResponseContext(
            entities_used=context_data.get('entities_used', []),
            relations_used=context_data.get('relations_used', []),
            sources=context_data.get('sources', []),
            confidence_scores=context_data.get('confidence_scores', {}),
            processing_steps=context_data.get('processing_steps', [])
        )

        # Parse metadata
        metadata_data = data.get('metadata', {})
        generation_time = datetime.fromisoformat(metadata_data['generation_time']) if 'generation_time' in metadata_data else datetime.now()
        metadata = ResponseMetadata(
            generation_time=generation_time,
            processing_duration=metadata_data.get('processing_duration', 0.0),
            model_used=metadata_data.get('model_used', ''),
            domain=metadata_data.get('domain', ''),
            query_id=metadata_data.get('query_id', ''),
            response_type=metadata_data.get('response_type', '')
        )

        return cls(
            id=data['id'],
            text=data['text'],
            confidence=data['confidence'],
            entities=entities,
            relations=relations,
            sections=data.get('sections', {}),
            bullet_points=data.get('bullet_points', []),
            context=context,
            metadata=metadata,
            additional_data=data.get('additional_data', {})
        )

    def add_entity(self, entity: Entity) -> None:
        """Add entity to response."""
        if entity not in self.entities:
            self.entities.append(entity)
            self.context.entities_used.append(entity.id)

    def add_relation(self, relation: Relation) -> None:
        """Add relation to response."""
        if relation not in self.relations:
            self.relations.append(relation)
            self.context.relations_used.append(relation.id)

    def add_section(self, title: str, content: str) -> None:
        """Add section to response."""
        self.sections[title] = content

    def add_bullet_point(self, point: str) -> None:
        """Add bullet point to response."""
        if point not in self.bullet_points:
            self.bullet_points.append(point)

    def add_source(self, source: str) -> None:
        """Add source to context."""
        if source not in self.context.sources:
            self.context.sources.append(source)

    def set_confidence_score(self, component: str, score: float) -> None:
        """Set confidence score for a component."""
        self.context.confidence_scores[component] = score

    def add_processing_step(self, step: str) -> None:
        """Add processing step to context."""
        self.context.processing_steps.append(step)

    def get_summary(self) -> str:
        """Get response summary."""
        summary_parts = []

        if self.sections:
            # Use first section as summary
            first_section = next(iter(self.sections.values()))
            summary_parts.append(first_section[:200] + "..." if len(first_section) > 200 else first_section)
        else:
            # Use first 200 characters of text
            summary_parts.append(self.text[:200] + "..." if len(self.text) > 200 else self.text)

        if self.entities:
            summary_parts.append(f"Found {len(self.entities)} relevant entities.")

        if self.relations:
            summary_parts.append(f"Found {len(self.relations)} relevant relationships.")

        return " ".join(summary_parts)