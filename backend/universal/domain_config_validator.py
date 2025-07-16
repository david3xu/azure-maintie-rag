"""
Domain Configuration Validator - validates auto-generated domain configs
"""

import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DomainConfigValidator:
    """Validates automatically generated domain configurations"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.validation_errors = []
        self.validation_warnings = []

    def validate_domain_config(self, domain_config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Comprehensive validation of domain configuration"""

        self.validation_errors = []
        self.validation_warnings = []

        # Core structure validation
        self._validate_core_structure(domain_config)

        # Entity validation
        self._validate_entities(domain_config.get("entity_types", []))

        # Relationship validation
        self._validate_relationships(domain_config.get("relationship_types", []))

        # Triplet validation
        self._validate_triplets(domain_config.get("knowledge_triplets", []))

        # Quality metrics validation
        self._validate_quality_metrics(domain_config)

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def _validate_core_structure(self, config: Dict[str, Any]) -> None:
        """Validate core configuration structure"""
        required_fields = ["domain_name", "entity_types", "relationship_types", "knowledge_triplets"]

        for field in required_fields:
            if field not in config:
                self.validation_errors.append(f"Missing required field: {field}")

        if config.get("domain_name") != self.domain_name:
            self.validation_errors.append(f"Domain name mismatch: expected {self.domain_name}, got {config.get('domain_name')}")

    def _validate_entities(self, entities: List[str]) -> None:
        """Validate entity types"""
        if len(entities) < 3:
            self.validation_errors.append(f"Too few entity types ({len(entities)}). Minimum: 3")

        if len(entities) > 50:
            self.validation_warnings.append(f"Many entity types ({len(entities)}). Consider consolidation")

        # Check for valid naming convention
        for entity in entities:
            if not isinstance(entity, str) or not entity.strip():
                self.validation_errors.append(f"Invalid entity type: {entity}")
            elif " " in entity and "_" not in entity:
                self.validation_warnings.append(f"Entity type '{entity}' should use underscores instead of spaces")

    def _validate_relationships(self, relationships: List[str]) -> None:
        """Validate relationship types"""
        if len(relationships) < 2:
            self.validation_errors.append(f"Too few relationship types ({len(relationships)}). Minimum: 2")

        # Check for common relationship patterns
        common_relations = ["has_part", "located_in", "used_for", "causes", "requires"]
        found_common = any(rel in relationships for rel in common_relations)

        if not found_common:
            self.validation_warnings.append("No common relationship patterns found. Verify extraction quality")

    def _validate_triplets(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Validate knowledge triplets"""
        if len(triplets) < 10:
            self.validation_errors.append(f"Too few knowledge triplets ({len(triplets)}). Minimum: 10")

        # Check triplet structure
        invalid_triplets = []
        for i, triplet in enumerate(triplets):
            if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                invalid_triplets.append(i)
            elif any(not isinstance(item, str) or not item.strip() for item in triplet):
                invalid_triplets.append(i)

        if invalid_triplets:
            self.validation_errors.append(f"Invalid triplet structure at indices: {invalid_triplets[:10]}")

    def _validate_quality_metrics(self, config: Dict[str, Any]) -> None:
        """Validate quality metrics and statistics"""
        stats = config.get("statistics", {})

        if "entities_discovered" not in stats:
            self.validation_warnings.append("Missing entity discovery statistics")

        if "documents_processed" in stats and stats["documents_processed"] < 10:
            self.validation_warnings.append(f"Few documents processed ({stats['documents_processed']}). Results may be unreliable")

    def generate_validation_report(self) -> str:
        """Generate human-readable validation report"""
        report = f"Domain Configuration Validation Report: {self.domain_name}\n"
        report += "=" * 60 + "\n\n"

        if not self.validation_errors and not self.validation_warnings:
            report += "✅ Configuration passed all validation checks\n"
        else:
            if self.validation_errors:
                report += f"❌ ERRORS ({len(self.validation_errors)}):\n"
                for error in self.validation_errors:
                    report += f"  - {error}\n"
                report += "\n"

            if self.validation_warnings:
                report += f"⚠️  WARNINGS ({len(self.validation_warnings)}):\n"
                for warning in self.validation_warnings:
                    report += f"  - {warning}\n"

        return report