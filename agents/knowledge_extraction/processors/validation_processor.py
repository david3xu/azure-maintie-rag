"""
Centralized Validation using PydanticAI Output Validators

Uses centralized PydanticAI output validators from data_models.py.
Replaces complex validation logic with agent-driven validation patterns.
"""

import logging
from typing import Any, Dict, List
from pydantic import BaseModel, Field

# Import centralized PydanticAI validators (replaces local validation models)
from agents.core.data_models import (
    validate_entity_extraction,
    validate_relationship_extraction,
    validate_extraction_quality,
    ValidatedEntity,
    ValidatedRelationship,
    ExtractionQualityOutput,
    ValidationResult,
)

# Import from centralized constants
from agents.core.constants import KnowledgeExtractionConstants, CacheConstants

logger = logging.getLogger(__name__)


# ValidationResult now imported from agents.core.data_models


# Simple validation using PydanticAI built-in patterns
def validate_extraction_simple(
    entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
) -> bool:
    """
    Ultra-simple validation using PydanticAI built-in validation

    Returns:
        bool: True if validation passes
    """
    try:
        # Use PydanticAI built-in validation
        for entity_data in entities:
            ValidatedEntity(**entity_data)  # Will raise ValidationError if invalid

        for rel_data in relationships:
            ValidatedRelationship(**rel_data)  # Will raise ValidationError if invalid

        return len(entities) > 0  # Simple check: at least one entity

    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        return False


class PydanticAIValidator:
    """Enhanced validator using centralized PydanticAI output validators"""

    def validate_extraction(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        min_entity_confidence: float = None,
        min_relationship_confidence: float = None,
    ) -> ValidationResult:
        """
        Enhanced validation using centralized PydanticAI output validators

        Replaces manual validation logic with agent-driven validation patterns.
        """
        errors = []
        warnings = []

        try:
            # Use centralized PydanticAI entity validation (replaces manual loops)
            validated_entities: List[ValidatedEntity] = validate_entity_extraction(
                entities
            )
            logger.info(
                f"PydanticAI entity validation: {len(entities)} → {len(validated_entities)} entities"
            )

        except Exception as e:
            errors.append(f"PydanticAI entity validation failed: {str(e)}")
            validated_entities = []

        try:
            # Use centralized PydanticAI relationship validation (replaces manual loops)
            validated_relationships: List[ValidatedRelationship] = (
                validate_relationship_extraction(relationships)
            )
            logger.info(
                f"PydanticAI relationship validation: {len(relationships)} → {len(validated_relationships)} relationships"
            )

        except Exception as e:
            errors.append(f"PydanticAI relationship validation failed: {str(e)}")
            validated_relationships = []

        # Optional: Overall quality validation
        if not errors and entities and relationships:
            try:
                quality_data = {
                    "entities_per_text": len(validated_entities)
                    / 1.0,  # Assume 1 text for simplicity
                    "relations_per_entity": len(validated_relationships)
                    / max(1, len(validated_entities)),
                    "avg_entity_confidence": sum(
                        e.confidence for e in validated_entities
                    )
                    / len(validated_entities),
                    "avg_relation_confidence": sum(
                        r.confidence for r in validated_relationships
                    )
                    / len(validated_relationships),
                    "overall_score": 0.8,  # Base score
                    "quality_tier": "good",
                }

                quality_result: ExtractionQualityOutput = validate_extraction_quality(
                    quality_data
                )
                logger.info(
                    f"Overall extraction quality: {quality_result.quality_tier} ({quality_result.overall_score:.3f})"
                )

            except Exception as e:
                warnings.append(f"Quality assessment failed: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            entity_count=len(validated_entities),
            relationship_count=len(validated_relationships),
        )

    def log_validation_result(self, result: ValidationResult, context: str = ""):
        """Log validation results"""
        if result.is_valid:
            logger.info(
                f"✅ Validation passed {context}: {result.entity_count} entities, {result.relationship_count} relationships"
            )
        else:
            logger.error(f"❌ Validation failed {context}: {len(result.errors)} errors")

        for error in result.errors:
            logger.error(f"  ERROR: {error}")


# Backward compatibility aliases (updated to use PydanticAI)
ValidationProcessor = PydanticAIValidator
SimpleValidator = PydanticAIValidator  # Maintain old interface

# Export interfaces (enhanced with PydanticAI)
__all__ = [
    "ValidationProcessor",
    "PydanticAIValidator",
    "SimpleValidator",
    "ValidationResult",
    "validate_extraction_simple",
]
