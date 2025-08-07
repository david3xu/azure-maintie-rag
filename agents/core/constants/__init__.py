"""
Modular Constants Central Coordination
======================================

This module provides central coordination for the modular constants system while
maintaining full backward compatibility with existing imports. It re-exports all
constant classes from the modular structure and preserves utility functions.

Phase 1 Implementation Features:
- Complete modular structure with domain-organized modules
- Full backward compatibility - all existing imports continue to work
- Foundation for Phase 2 (model integration) and Phase 3 (automation)
- Mathematical base constants as foundation layer
- Interdependency group documentation preserved

Modular Structure:
- base.py: Mathematical constants and scaling factors (foundation)
- system.py: System boundary and infrastructure constants
- domain.py: Domain-adaptive constants (high automation potential)
- performance.py: Performance-adaptive constants
- search.py: Search and ML model constants
- workflow.py: Workflow coordination constants
- extraction.py: Knowledge extraction constants
- validation.py: Security and validation constants
- legacy.py: Backward compatibility aliases

Import this module using any of these patterns (all work identically):
    from agents.core.constants import SystemBoundaryConstants
    from agents.core.constants.system import SystemBoundaryConstants
    from agents.core.constants import MathematicalConstants
"""

# =============================================================================
# IMPORT ALL MODULAR CONSTANTS - CENTRAL COORDINATION
# =============================================================================

from typing import Any, Dict, List

# Foundation Layer - Mathematical Base Constants
from .base import (
    BaseScalingFactors,
    MathematicalConstants,
    derive_chunk_size,
    derive_confidence,
    derive_timeout,
)

# Domain Layer - Adaptive Domain Intelligence
from .domain import (
    ContentAnalysisAdaptiveConstants,
    DomainAdaptiveConstants,
)

# Extraction Layer - Knowledge Processing
from .extraction import (
    ExtractionAlgorithmConstants,
    KnowledgeExtractionConstants,
)

# Legacy Layer - Backward Compatibility
from .legacy import (
    AzureServiceConstants,
    CacheConstants,
    ContentAnalysisConstants,
    MLModelConstants,
    ProcessingConstants,
    StubConstants,
)

# Performance Layer - Adaptive Performance Optimization
from .performance import (
    PerformanceAdaptiveConstants,
    SearchPerformanceAdaptiveConstants,
)

# Search Layer - ML Models and Search Intelligence
from .search import (
    MLModelStaticConstants,
    StatisticalConstants,
    UniversalSearchConstants,
)

# System Layer - Boundaries and Infrastructure
from .system import (
    FileSystemConstants,
    InfrastructureConstants,
    SystemBoundaryConstants,
)

# Validation Layer - Security and Data Integrity
from .validation import (
    DataModelConstants,
    SecurityConstants,
)

# Workflow Layer - Coordination and Orchestration
from .workflow import (
    ErrorHandlingCoordinatedConstants,
    WorkflowConstants,
    WorkflowCoordinationConstants,
)

# =============================================================================
# PRESERVE ORIGINAL UTILITY FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================


def get_constant_by_category(category: str) -> Dict[str, Any]:
    """
    Get all constants for a specific category

    Args:
        category: Category name from the organized structure

    Returns:
        Dictionary of constants for the category
    """
    category_mapping = {
        # New organized categories
        "mathematical": MathematicalConstants,
        "scaling": BaseScalingFactors,
        "system_boundary": SystemBoundaryConstants,
        "infrastructure": InfrastructureConstants,
        "filesystem": FileSystemConstants,
        "domain_adaptive": DomainAdaptiveConstants,
        "content_adaptive": ContentAnalysisAdaptiveConstants,
        "performance_adaptive": PerformanceAdaptiveConstants,
        "search_performance_adaptive": SearchPerformanceAdaptiveConstants,
        "ml_model_static": MLModelStaticConstants,
        "statistical": StatisticalConstants,
        "workflow_coordination": WorkflowCoordinationConstants,
        "error_handling_coordinated": ErrorHandlingCoordinatedConstants,
        "security": SecurityConstants,
        "extraction_algorithm": ExtractionAlgorithmConstants,
        "data_models": DataModelConstants,
        # Legacy categories (for backward compatibility)
        "extraction": KnowledgeExtractionConstants,
        "search": UniversalSearchConstants,
        "workflow": WorkflowConstants,
        # Backward compatibility aliases
        "cache": CacheConstants,
        "processing": ProcessingConstants,
        "azure": AzureServiceConstants,
        "content_analysis": ContentAnalysisConstants,
        "ml_model": MLModelConstants,
        "stub": StubConstants,
    }

    if category not in category_mapping:
        raise ValueError(
            f"Unknown category: {category}. Available: {list(category_mapping.keys())}"
        )

    const_class = category_mapping[category]
    return {
        attr: getattr(const_class, attr)
        for attr in dir(const_class)
        if not attr.startswith("_")
    }


def get_automation_potential_summary() -> Dict[str, str]:
    """
    Get summary of automation potential for each constant category

    Returns:
        Dictionary mapping category to automation potential description
    """
    return {
        "mathematical": "STATIC - Foundation constants, never auto-generate",
        "scaling": "STATIC - Mathematical relationships, never auto-generate",
        "system_boundary": "STATIC - Never auto-generate (system limits)",
        "infrastructure": "HIGH - Can discover from Azure deployment",
        "filesystem": "LOW - Conventional paths, rarely change",
        "domain_adaptive": "VERY HIGH - Should be learned by Domain Intelligence Agent",
        "content_adaptive": "HIGH - Learn from corpus analysis",
        "performance_adaptive": "HIGH - Optimize from performance metrics",
        "search_performance_adaptive": "HIGH - Optimize from search quality metrics",
        "ml_model_static": "MEDIUM - Some hyperparameters learnable",
        "statistical": "MEDIUM - Mix of standards and learnable thresholds",
        "workflow_coordination": "MEDIUM-HIGH - Optimize interdependent groups",
        "error_handling_coordinated": "MEDIUM - Coordinate for resilience",
        "security": "LOW - Keep static for consistency",
        "extraction_algorithm": "MEDIUM-HIGH - Learn from extraction performance",
        "data_models": "LOW - Keep static for API consistency",
    }


def get_interdependency_groups() -> Dict[str, List[str]]:
    """
    Get groups of constants that should be optimized together

    Returns:
        Dictionary mapping group names to lists of interdependent constants
    """
    return {
        "entity_extraction_quality": [
            "ENTITY_CONFIDENCE_THRESHOLD",
            "RELATIONSHIP_CONFIDENCE_THRESHOLD",
            "MIN_RELATIONSHIP_STRENGTH",
            "MAX_ENTITIES_PER_CHUNK",
        ],
        "document_processing_parameters": [
            "DEFAULT_CHUNK_SIZE",
            "DEFAULT_CHUNK_OVERLAP",
            "MIN_ENTITY_LENGTH",
            "MAX_ENTITY_LENGTH",
        ],
        "search_quality_thresholds": [
            "RESULT_RELEVANCE_THRESHOLD",
            "MIN_CONFIDENCE_THRESHOLD",
        ],
        "tri_modal_search_weights": [
            "MULTI_MODAL_WEIGHT_VECTOR",
            "MULTI_MODAL_WEIGHT_GRAPH",
            "MULTI_MODAL_WEIGHT_GNN",
        ],
        "timeout_retry_strategy": [
            "DEFAULT_TIMEOUT",
            "MAX_RETRIES",
            "RETRY_DELAY",
            "EXPONENTIAL_BACKOFF_MULTIPLIER",
        ],
        "batch_processing_optimization": [
            "DEFAULT_BATCH_SIZE",
            "PARALLEL_WORKERS",
            "MAX_CONCURRENT_CHUNKS",
            "MAX_BATCH_SIZE",
        ],
        "cache_performance_tuning": [
            "DEFAULT_CACHE_TTL",
            "TARGET_CACHE_HIT_RATE",
            "SHORT_CACHE_TTL",
            "LONG_CACHE_TTL",
        ],
        "synthesis_weights": [
            "CONFIDENCE_WEIGHT",
            "AGREEMENT_WEIGHT",
            "QUALITY_WEIGHT",
        ],
    }


def validate_constants() -> List[str]:
    """
    Validate that all constants are properly defined and interdependent groups are consistent

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []

    # Validate tri-modal weights sum to 1.0
    total_weight = (
        SearchPerformanceAdaptiveConstants.MULTI_MODAL_WEIGHT_VECTOR
        + SearchPerformanceAdaptiveConstants.MULTI_MODAL_WEIGHT_GRAPH
        + SearchPerformanceAdaptiveConstants.MULTI_MODAL_WEIGHT_GNN
    )
    if abs(total_weight - 1.0) > 0.001:
        errors.append(f"Tri-modal weights sum to {total_weight}, should sum to 1.0")

    # Validate synthesis weights sum to 1.0
    synthesis_total = (
        WorkflowCoordinationConstants.CONFIDENCE_WEIGHT
        + WorkflowCoordinationConstants.AGREEMENT_WEIGHT
        + WorkflowCoordinationConstants.QUALITY_WEIGHT
    )
    if abs(synthesis_total - 1.0) > 0.001:
        errors.append(f"Synthesis weights sum to {synthesis_total}, should sum to 1.0")

    # Validate threshold relationships
    if (
        DomainAdaptiveConstants.RELATIONSHIP_CONFIDENCE_THRESHOLD
        > DomainAdaptiveConstants.ENTITY_CONFIDENCE_THRESHOLD
    ):
        errors.append(
            "Relationship confidence threshold should not exceed entity confidence threshold"
        )

    # Validate chunk overlap is less than chunk size
    if (
        DomainAdaptiveConstants.DEFAULT_CHUNK_OVERLAP
        >= DomainAdaptiveConstants.DEFAULT_CHUNK_SIZE
    ):
        errors.append("Chunk overlap should be less than chunk size")

    # Validate performance thresholds are in order
    perf_thresholds = [
        WorkflowCoordinationConstants.EXCELLENT_PERFORMANCE_THRESHOLD,
        WorkflowCoordinationConstants.GOOD_PERFORMANCE_THRESHOLD,
        WorkflowCoordinationConstants.ACCEPTABLE_PERFORMANCE_THRESHOLD,
    ]
    if perf_thresholds != sorted(perf_thresholds):
        errors.append("Performance thresholds should be in ascending order")

    return errors


# =============================================================================
# PRESERVE ORIGINAL DOMAIN INTELLIGENCE CONSTANTS (Empty placeholder)
# =============================================================================
# Maintaining the original structure for any remaining dependencies


class DomainIntelligenceConstants:
    """Domain Intelligence Agent specific constants - Legacy placeholder"""

    pass


# =============================================================================
# COMPLETE EXPORT LIST - FULL BACKWARD COMPATIBILITY
# =============================================================================

__all__ = [
    # Foundation Layer
    "MathematicalConstants",
    "BaseScalingFactors",
    "derive_timeout",
    "derive_chunk_size",
    "derive_confidence",
    # Modular Organization - New Structure
    "SystemBoundaryConstants",
    "InfrastructureConstants",
    "FileSystemConstants",
    "DomainAdaptiveConstants",
    "ContentAnalysisAdaptiveConstants",
    "PerformanceAdaptiveConstants",
    "SearchPerformanceAdaptiveConstants",
    "MLModelStaticConstants",
    "StatisticalConstants",
    "WorkflowCoordinationConstants",
    "ErrorHandlingCoordinatedConstants",
    "SecurityConstants",
    "ExtractionAlgorithmConstants",
    "DataModelConstants",
    # Legacy Agent-Specific Constants (backward compatibility)
    "DomainIntelligenceConstants",
    "KnowledgeExtractionConstants",
    "UniversalSearchConstants",
    "WorkflowConstants",
    # Backward Compatibility Aliases
    "CacheConstants",
    "ProcessingConstants",
    "AzureServiceConstants",
    "ContentAnalysisConstants",
    "MLModelConstants",
    "StubConstants",
    # Utility Functions
    "get_constant_by_category",
    "get_automation_potential_summary",
    "get_interdependency_groups",
    "validate_constants",
]
