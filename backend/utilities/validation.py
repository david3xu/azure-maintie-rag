"""Universal validation utilities for any domain."""

from typing import Any, Dict, List, Optional, Tuple
import re
from pathlib import Path


class ValidationUtils:
    """Universal validation utilities that work across all domains."""

    @staticmethod
    def validate_entity_name(name: str) -> Tuple[bool, str]:
        """Validate entity name."""
        if not name or not name.strip():
            return False, "Entity name cannot be empty"

        if len(name) > 200:
            return False, "Entity name too long (max 200 characters)"

        # Check for basic formatting
        if not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', name):
            return False, "Entity name contains invalid characters"

        return True, "Valid"

    @staticmethod
    def validate_confidence_score(score: float) -> Tuple[bool, str]:
        """Validate confidence score."""
        if not isinstance(score, (int, float)):
            return False, "Confidence score must be numeric"

        if score < 0.0 or score > 1.0:
            return False, "Confidence score must be between 0.0 and 1.0"

        return True, "Valid"

    @staticmethod
    def validate_text_quality(text: str, min_length: int = 10) -> Tuple[bool, str]:
        """Validate text quality."""
        if not text or not text.strip():
            return False, "Text cannot be empty"

        if len(text.strip()) < min_length:
            return False, f"Text too short (minimum {min_length} characters)"

        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3:
            return False, "Text has too few alphabetic characters"

        return True, "Valid"

    @staticmethod
    def validate_domain_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate domain configuration."""
        errors = []

        # Required sections
        required_sections = [
            'domain', 'entities', 'relationships',
            'processing', 'query', 'performance'
        ]

        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Validate domain info
        if 'domain' in config:
            domain_info = config['domain']
            if 'name' not in domain_info:
                errors.append("Domain name is required")
            elif not domain_info['name'].strip():
                errors.append("Domain name cannot be empty")

        # Validate entity types
        if 'entities' in config:
            entity_config = config['entities']
            if 'types' not in entity_config:
                errors.append("Entity types are required")
            elif not isinstance(entity_config['types'], list):
                errors.append("Entity types must be a list")

        # Validate relationship types
        if 'relationships' in config:
            rel_config = config['relationships']
            if 'types' not in rel_config:
                errors.append("Relationship types are required")
            elif not isinstance(rel_config['types'], list):
                errors.append("Relationship types must be a list")

        return len(errors) == 0, errors

    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> Tuple[bool, str]:
        """Validate file path."""
        if not file_path:
            return False, "File path cannot be empty"

        path = Path(file_path)

        if must_exist and not path.exists():
            return False, f"File does not exist: {file_path}"

        if must_exist and not path.is_file():
            return False, f"Path is not a file: {file_path}"

        return True, "Valid"

    @staticmethod
    def validate_query_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate query parameters."""
        errors = []

        # Validate max_results
        if 'max_results' in params:
            max_results = params['max_results']
            if not isinstance(max_results, int) or max_results <= 0:
                errors.append("max_results must be a positive integer")

        # Validate confidence_threshold
        if 'confidence_threshold' in params:
            threshold = params['confidence_threshold']
            is_valid, message = ValidationUtils.validate_confidence_score(threshold)
            if not is_valid:
                errors.append(f"confidence_threshold: {message}")

        # Validate search_depth
        if 'search_depth' in params:
            depth = params['search_depth']
            if not isinstance(depth, int) or depth < 1:
                errors.append("search_depth must be a positive integer")

        return len(errors) == 0, errors

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text for safe processing."""
        if not text:
            return ""

        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    @staticmethod
    def validate_extraction_results(entities: List[Dict], relations: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate extraction results."""
        errors = []

        # Validate entities
        entity_ids = set()
        for i, entity in enumerate(entities):
            if 'id' not in entity:
                errors.append(f"Entity {i} missing ID")
            elif entity['id'] in entity_ids:
                errors.append(f"Duplicate entity ID: {entity['id']}")
            else:
                entity_ids.add(entity['id'])

            if 'name' not in entity:
                errors.append(f"Entity {i} missing name")

            if 'confidence' in entity:
                is_valid, message = ValidationUtils.validate_confidence_score(entity['confidence'])
                if not is_valid:
                    errors.append(f"Entity {i} confidence: {message}")

        # Validate relations
        for i, relation in enumerate(relations):
            if 'source_entity_id' not in relation:
                errors.append(f"Relation {i} missing source_entity_id")
            elif relation['source_entity_id'] not in entity_ids:
                errors.append(f"Relation {i} references unknown source entity")

            if 'target_entity_id' not in relation:
                errors.append(f"Relation {i} missing target_entity_id")
            elif relation['target_entity_id'] not in entity_ids:
                errors.append(f"Relation {i} references unknown target entity")

            if 'confidence' in relation:
                is_valid, message = ValidationUtils.validate_confidence_score(relation['confidence'])
                if not is_valid:
                    errors.append(f"Relation {i} confidence: {message}")

        return len(errors) == 0, errors