"""
Simple Validation - Essential Quality Checks Only

This module provides basic validation of extraction results without over-engineering.
Focus: Catch real errors, not create elaborate scoring systems.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Import centralized configuration
from config.centralized_config import get_confidence_calculation_config, get_quality_assessment_config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Simple validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    entity_count: int
    relationship_count: int


class SimpleValidator:
    """
    Simple, practical validation without over-engineering.
    
    Validates only what matters:
    - Entities and relationships exist
    - Basic data structure integrity
    - Critical confidence thresholds
    """
    
    def validate_extraction(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        min_entity_confidence: float = None,
        min_relationship_confidence: float = None
    ) -> ValidationResult:
        """
        Simple validation that checks only essential quality criteria.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships  
            min_entity_confidence: Minimum confidence for entities
            min_relationship_confidence: Minimum confidence for relationships
            
        Returns:
            ValidationResult: Simple pass/fail with basic metrics
        """
        # Use centralized configuration for defaults
        quality_config = get_quality_assessment_config()
        if min_entity_confidence is None:
            min_entity_confidence = quality_config.default_entity_confidence_threshold
        if min_relationship_confidence is None:
            min_relationship_confidence = quality_config.default_relationship_confidence_threshold
        
        errors = []
        warnings = []
        
        # 1. Check basic data presence
        if not entities:
            errors.append("No entities extracted")
        
        if not relationships:
            warnings.append("No relationships extracted")
        
        # 2. Check entity data integrity
        for i, entity in enumerate(entities):
            if not entity.get("name"):
                errors.append(f"Entity {i} missing name")
            
            confidence = entity.get("confidence", 0.0)
            if confidence < min_entity_confidence:
                warnings.append(f"Entity '{entity.get('name', 'unknown')}' low confidence: {confidence}")
        
        # 3. Check relationship data integrity
        entity_names = {e.get("name") for e in entities}
        for i, rel in enumerate(relationships):
            source = rel.get("source")
            target = rel.get("target")
            
            if not source or not target:
                errors.append(f"Relationship {i} missing source or target")
                continue
                
            # Check if entities exist
            if source not in entity_names:
                errors.append(f"Relationship source '{source}' not found in entities")
            
            if target not in entity_names:
                errors.append(f"Relationship target '{target}' not found in entities")
            
            # Check confidence
            confidence = rel.get("confidence", 0.0)
            if confidence < min_relationship_confidence:
                warnings.append(f"Relationship '{source} -> {target}' low confidence: {confidence}")
        
        # 4. Simple validation result
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            entity_count=len(entities),
            relationship_count=len(relationships)
        )
    
    def log_validation_result(self, result: ValidationResult, context: str = ""):
        """Log validation results in a simple, readable format"""
        
        if result.is_valid:
            logger.info(f"✅ Validation passed {context}: {result.entity_count} entities, {result.relationship_count} relationships")
        else:
            logger.error(f"❌ Validation failed {context}: {len(result.errors)} errors")
            
        for error in result.errors:
            logger.error(f"  ERROR: {error}")
            
        for warning in result.warnings:
            logger.warning(f"  WARNING: {warning}")


# Simple factory function
def validate_extraction_simple(
    entities: List[Dict[str, Any]], 
    relationships: List[Dict[str, Any]]
) -> bool:
    """
    Ultra-simple validation that just returns pass/fail
    
    Returns:
        bool: True if basic validation passes
    """
    validator = SimpleValidator()
    result = validator.validate_extraction(entities, relationships)
    return result.is_valid


# Backward compatibility alias - ValidationProcessor is now SimpleValidator
ValidationProcessor = SimpleValidator

# Export interfaces (maintaining backward compatibility)
__all__ = ["ValidationProcessor", "SimpleValidator", "ValidationResult", "validate_extraction_simple"]