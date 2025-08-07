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

**Phase 3 Implementation Features (NEW):**
- Automation classification system for constants by generation potential
- Performance feedback loops for adaptive constant optimization
- Domain Intelligence Agent integration for automatic constant generation
- Comprehensive safety validation for automated generation
- Configuration discovery from Azure deployment scanning

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
- **automation_classifier.py: Automation potential classification (Phase 3)**
- **automation_interface.py: Automation coordination (Phase 3)**
- **domain_intelligence_integration.py: Domain-driven generation (Phase 3)**
- **performance_feedback_loops.py: Real-time optimization (Phase 3)**
- **safety_validation.py: Comprehensive validation (Phase 3)**

Import this module using any of these patterns (all work identically):
    from agents.core.constants import SystemBoundaryConstants
    from agents.core.constants.system import SystemBoundaryConstants
    from agents.core.constants import MathematicalConstants
    
**Phase 3 Automation Access:**
    from agents.core.constants import automation_coordinator
    from agents.core.constants import performance_feedback_orchestrator
    from agents.core.constants import constant_safety_validator
"""

# =============================================================================
# IMPORT ALL MODULAR CONSTANTS - CENTRAL COORDINATION
# =============================================================================

from typing import Any, Dict, List

# Foundation Layer - Mathematical Base Constants
from .base import (
    BaseScalingFactors,
    MathematicalConstants,
    MathematicalFoundationConstants,
    derive_chunk_size,
    derive_confidence,
    derive_timeout,
)

# Domain Layer - Adaptive Domain Intelligence
from .domain import (
    ContentAnalysisAdaptiveConstants,
    DomainAdaptiveConstants,
    DomainIntelligenceConstants,
)

# Extraction Layer - Knowledge Processing
from .extraction import (
    ExtractionAlgorithmConstants,
    ExtractionQualityConstants,
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
    SystemPerformanceConstants,
)

# Validation Layer - Security and Data Integrity
from .validation import (
    DataModelConstants,
    SecurityConstants,
)

# Workflow Layer - Coordination and Orchestration
from .workflow import (
    ErrorHandlingConstants,
    ErrorHandlingCoordinatedConstants,
    WorkflowConstants,
    WorkflowCoordinationConstants,
    WorkflowExecutionConstants,
)

# =============================================================================
# PHASE 3: AUTOMATION SYSTEM IMPORTS
# =============================================================================

# Automation Classification and Coordination
from .automation_classifier import (
    AutomationPotential,
    LearningMechanism,
    ConstantClassification,
    AutomationClassifier,
    automation_classifier,
)

# Automation Interface and Coordination
from .automation_interface import (
    GenerationRequest,
    GenerationResult,
    AutomationCoordinator,
    automation_coordinator,
)

# Domain Intelligence Integration
from .domain_intelligence_integration import (
    DomainAnalysisResult,
    DomainIntelligenceConstantGenerator,
    domain_intelligence_generator,
)

# Performance Feedback Loops
from .performance_feedback_loops import (
    MetricType,
    PerformanceMetric,
    AdaptationRule,
    PerformanceFeedbackOrchestrator,
    performance_feedback_orchestrator,
    record_response_time,
    record_cache_hit_rate,
    record_extraction_accuracy,
    record_search_relevance,
)

# Safety Validation
from .safety_validation import (
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ConstantSafetyValidator,
    constant_safety_validator,
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
# ORIGINAL DOMAIN INTELLIGENCE CONSTANTS NOW IMPORTED FROM DOMAIN MODULE
# =============================================================================
# The DomainIntelligenceConstants are now properly imported from domain.py


# =============================================================================
# COMPLETE EXPORT LIST - FULL BACKWARD COMPATIBILITY
# =============================================================================

__all__ = [
    # Foundation Layer
    "MathematicalConstants",
    "MathematicalFoundationConstants",
    "BaseScalingFactors",
    "derive_timeout",
    "derive_chunk_size",
    "derive_confidence",
    # Modular Organization - New Structure
    "SystemBoundaryConstants",
    "SystemPerformanceConstants",
    "InfrastructureConstants",
    "FileSystemConstants",
    "DomainAdaptiveConstants",
    "DomainIntelligenceConstants",
    "ContentAnalysisAdaptiveConstants",
    "PerformanceAdaptiveConstants",
    "SearchPerformanceAdaptiveConstants",
    "MLModelStaticConstants",
    "StatisticalConstants",
    "WorkflowCoordinationConstants",
    "WorkflowExecutionConstants",
    "ErrorHandlingConstants",
    "ErrorHandlingCoordinatedConstants",
    "SecurityConstants",
    "ExtractionAlgorithmConstants",
    "ExtractionQualityConstants",
    "DataModelConstants",
    # Legacy Agent-Specific Constants (backward compatibility)
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
    # Phase 3: Automation System Components
    "AutomationPotential",
    "LearningMechanism",
    "ConstantClassification",
    "AutomationClassifier",
    "automation_classifier",
    "GenerationRequest",
    "GenerationResult",
    "AutomationCoordinator",
    "automation_coordinator",
    "DomainAnalysisResult",
    "DomainIntelligenceConstantGenerator",
    "domain_intelligence_generator",
    "MetricType",
    "PerformanceMetric",
    "AdaptationRule",
    "PerformanceFeedbackOrchestrator",
    "performance_feedback_orchestrator",
    "record_response_time",
    "record_cache_hit_rate",
    "record_extraction_accuracy",
    "record_search_relevance",
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationResult",
    "ConstantSafetyValidator",
    "constant_safety_validator",
]
