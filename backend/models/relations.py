"""Universal relation models for any domain."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class RelationType(Enum):
    """Universal relation types that work across domains."""
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    USES = "uses"
    USED_BY = "used_by"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    CONNECTED_TO = "connected_to"
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    RELATES_TO = "relates_to"
    CUSTOM = "custom"


@dataclass
class Relation:
    """Universal relation model for any domain.

    Configuration-driven relation that adapts to any domain through
    dynamic properties and configurable types.
    """

    # Core universal properties
    id: str
    source_entity_id: str
    target_entity_id: str
    type: RelationType

    # Relation metadata
    confidence: float
    source_text: str
    context: str

    # Configurable properties (domain-specific)
    properties: Dict[str, Any]

    # Universal attributes
    direction: str  # "directional", "bidirectional", "undirectional"
    strength: float  # Relationship strength (0.0 to 1.0)

    # Processing metadata
    extracted_by: str
    extraction_timestamp: str
    validation_status: str

    def __post_init__(self):
        """Validate relation after creation."""
        if not self.source_entity_id or not self.target_entity_id:
            raise ValueError("Source and target entity IDs cannot be empty")

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if self.strength < 0.0 or self.strength > 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")

        if self.direction not in ["directional", "bidirectional", "undirectional"]:
            raise ValueError("Direction must be directional, bidirectional, or undirectional")

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary for serialization."""
        return {
            'id': self.id,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'type': self.type.value,
            'confidence': self.confidence,
            'source_text': self.source_text,
            'context': self.context,
            'properties': self.properties,
            'direction': self.direction,
            'strength': self.strength,
            'extracted_by': self.extracted_by,
            'extraction_timestamp': self.extraction_timestamp,
            'validation_status': self.validation_status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """Create relation from dictionary."""
        return cls(
            id=data['id'],
            source_entity_id=data['source_entity_id'],
            target_entity_id=data['target_entity_id'],
            type=RelationType(data['type']),
            confidence=data['confidence'],
            source_text=data['source_text'],
            context=data['context'],
            properties=data.get('properties', {}),
            direction=data['direction'],
            strength=data['strength'],
            extracted_by=data['extracted_by'],
            extraction_timestamp=data['extraction_timestamp'],
            validation_status=data['validation_status']
        )

    def get_reverse_type(self) -> RelationType:
        """Get the reverse relation type for bidirectional relations."""
        reverse_mapping = {
            RelationType.IS_A: RelationType.CUSTOM,  # No direct reverse
            RelationType.PART_OF: RelationType.HAS_PART,
            RelationType.HAS_PART: RelationType.PART_OF,
            RelationType.USES: RelationType.USED_BY,
            RelationType.USED_BY: RelationType.USES,
            RelationType.CAUSES: RelationType.CAUSED_BY,
            RelationType.CAUSED_BY: RelationType.CAUSES,
            RelationType.CONNECTED_TO: RelationType.CONNECTED_TO,
            RelationType.LOCATED_IN: RelationType.CONTAINS,
            RelationType.CONTAINS: RelationType.LOCATED_IN,
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,
            RelationType.RELATES_TO: RelationType.RELATES_TO,
            RelationType.CUSTOM: RelationType.CUSTOM
        }
        return reverse_mapping.get(self.type, RelationType.RELATES_TO)