"""
Metadata management for enhanced MaintIE features
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from core.knowledge.schema_processor import SchemeProcessor

@dataclass
class TypeMetadata:
    """Metadata for entity/relation types"""
    color: str
    active: bool
    description: str
    example_terms: list
    path: list
    confidence: float = 1.0

class MetadataManager:
    """Manage type metadata for enhanced features"""

    def __init__(self, scheme_processor: SchemeProcessor):
        self.scheme_processor = scheme_processor
        self.entity_metadata = {}
        self.relation_metadata = {}
        self._build_metadata_cache()

    def _build_metadata_cache(self):
        """Build metadata cache from scheme processor"""
        # Cache entity metadata
        for fullname, node in self.scheme_processor.entity_hierarchy.items():
            self.entity_metadata[fullname] = TypeMetadata(
                color=node.metadata.get("color", "#cccccc"),
                active=node.metadata.get("active", True),
                description=node.metadata.get("description", ""),
                example_terms=node.metadata.get("example_terms", []),
                path=node.metadata.get("path", [])
            )

        # Cache relation metadata
        for fullname, node in self.scheme_processor.relation_hierarchy.items():
            self.relation_metadata[fullname] = TypeMetadata(
                color=node.metadata.get("color", "#cccccc"),
                active=node.metadata.get("active", True),
                description=node.metadata.get("description", ""),
                example_terms=node.metadata.get("example_terms", []),
                path=node.metadata.get("path", [])
            )

    def get_entity_metadata(self, entity_type: str) -> Optional[TypeMetadata]:
        """Get metadata for entity type"""
        return self.entity_metadata.get(entity_type)

    def get_relation_metadata(self, relation_type: str) -> Optional[TypeMetadata]:
        """Get metadata for relation type"""
        return self.relation_metadata.get(relation_type)

    def get_active_types(self, category: str) -> list:
        """Get only active types for category"""
        metadata_dict = self.entity_metadata if category == "entity" else self.relation_metadata
        return [type_name for type_name, metadata in metadata_dict.items() if metadata.active]