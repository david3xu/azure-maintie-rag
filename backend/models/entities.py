"""Universal entity models for any domain."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    """Universal entity types that work across domains."""
    CONCEPT = "concept"
    OBJECT = "object"
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    EVENT = "event"
    PROCESS = "process"
    ATTRIBUTE = "attribute"
    VALUE = "value"
    CUSTOM = "custom"


@dataclass
class Entity:
    """Universal entity model for any domain.

    Configuration-driven entity that adapts to any domain through
    dynamic properties and configurable types.
    """

    # Core universal properties
    id: str
    name: str
    type: EntityType

    # Domain-agnostic metadata
    confidence: float
    source_text: str
    context: str

    # Configurable properties (domain-specific)
    properties: Dict[str, Any]

    # Universal relationships
    aliases: List[str]
    synonyms: List[str]

    # Processing metadata
    extracted_by: str
    extraction_timestamp: str
    validation_status: str

    def __post_init__(self):
        """Validate entity after creation."""
        if not self.name or not self.name.strip():
            raise ValueError("Entity name cannot be empty")

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'confidence': self.confidence,
            'source_text': self.source_text,
            'context': self.context,
            'properties': self.properties,
            'aliases': self.aliases,
            'synonyms': self.synonyms,
            'extracted_by': self.extracted_by,
            'extraction_timestamp': self.extraction_timestamp,
            'validation_status': self.validation_status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            type=EntityType(data['type']),
            confidence=data['confidence'],
            source_text=data['source_text'],
            context=data['context'],
            properties=data.get('properties', {}),
            aliases=data.get('aliases', []),
            synonyms=data.get('synonyms', []),
            extracted_by=data['extracted_by'],
            extraction_timestamp=data['extraction_timestamp'],
            validation_status=data['validation_status']
        )

    def matches(self, query: str) -> float:
        """Universal entity matching for queries."""
        query_lower = query.lower()

        # Direct name match
        if query_lower in self.name.lower():
            return 1.0

        # Alias matches
        for alias in self.aliases:
            if query_lower in alias.lower():
                return 0.9

        # Synonym matches
        for synonym in self.synonyms:
            if query_lower in synonym.lower():
                return 0.8

        # Context matches
        if query_lower in self.context.lower():
            return 0.5

        return 0.0