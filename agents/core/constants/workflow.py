"""
Workflow Coordination Constants
===============================

This module contains constants that work together in workflows and should be optimized
as coordinated groups. These support multi-agent coordination and
workflow orchestration.

Key Interdependent Groups:
1. Performance Grading and Quality Gates - coordinated quality thresholds
2. Multi-Modal Search Coordination - tri-modal search synthesis
3. Error Handling and Resilience - circuit breaker and fallback coordination
4. Workflow Step Confidence - learning and validation patterns

AUTO-GENERATION POTENTIAL: MEDIUM-HIGH
These should be optimized as interdependent groups for workflow effectiveness.
"""


class WorkflowCoordinationConstants:
    """Workflow constants that must work together as coordinated groups"""

    # AUTO-GENERATION POTENTIAL: MEDIUM-HIGH
    # These should be optimized as interdependent groups

    # INTERDEPENDENT GROUP 3: Performance Grading and Quality Gates
    # These define coordinated quality thresholds across the system
    EXCELLENT_PERFORMANCE_THRESHOLD = (
        1.0  # COORDINATED: with good/acceptable thresholds
    )
    GOOD_PERFORMANCE_THRESHOLD = 2.0  # COORDINATED: with SLA targets
    ACCEPTABLE_PERFORMANCE_THRESHOLD = 3.0  # COORDINATED: with SLA limits
    MIN_EXTRACTION_ACCURACY = 0.85  # COORDINATED: with search relevance requirements
    MIN_SEARCH_RELEVANCE = 0.7  # COORDINATED: with extraction accuracy

    # INTERDEPENDENT GROUP 4: Multi-Modal Search Coordination
    # These must be coordinated for effective tri-modal search
    CONFIDENCE_WEIGHT = 0.4  # COORDINATED: with agreement and quality weights
    AGREEMENT_WEIGHT = 0.3  # COORDINATED: synthesis weight coordination
    QUALITY_WEIGHT = 0.3  # COORDINATED: synthesis weight coordination


class ErrorHandlingCoordinatedConstants:
    """Error handling constants that work together for system resilience"""

    # AUTO-GENERATION POTENTIAL: MEDIUM
    # These should be coordinated for consistent error handling behavior

    # INTERDEPENDENT GROUP: Circuit Breaker and Fallback Coordination
    DEFAULT_FAILURE_THRESHOLD = 5  # COORDINATED: with retry attempts and recovery
    CRITICAL_ERROR_THRESHOLD = 50  # COORDINATED: with warning threshold
    WARNING_ERROR_THRESHOLD = 20  # COORDINATED: with critical threshold
    DEFAULT_CONFIDENCE_FALLBACK = 0.3  # COORDINATED: with minimum confidence thresholds
    MIN_RECOVERY_RATE_THRESHOLD = 50  # COORDINATED: with failures


class WorkflowConstants:
    """Workflow orchestration constants - many could be moved to WorkflowCoordinationConstants"""

    # Confidence Levels - Could be LEARNABLE from workflow success patterns
    DISCOVERY_CONFIDENCE = 0.9  # LEARNABLE: workflow step confidence
    ANALYSIS_CONFIDENCE = 0.85  # LEARNABLE: analysis quality patterns
    PATTERN_CONFIDENCE = 0.8  # LEARNABLE: pattern recognition quality
    CONFIG_CONFIDENCE = 0.9  # LEARNABLE: configuration generation quality
    EXTRACTION_CONFIDENCE = 0.75  # LEARNABLE: extraction quality patterns
    QUALITY_SCORE = 0.85  # LEARNABLE: overall quality requirements
    VALIDATION_CONFIDENCE = 0.9  # LEARNABLE: validation effectiveness

    # Utility constants for workflow operations (imported from base)
    from .base import MathematicalConstants

    BYTES_TO_MB_DIVISOR = MathematicalConstants.BYTES_PER_MB
    STORAGE_SIZE_PRECISION = 2  # Decimal precision for storage size


class ErrorHandlingConstants:
    """Error handling and recovery constants"""

    # Backoff and delay
    MAX_BACKOFF_DELAY_SECONDS = 30.0

    # Adaptation thresholds
    ADAPTATION_THRESHOLD = 0.7


class WorkflowExecutionConstants:
    """Workflow execution coordination constants"""

    # Execution flow constants
    DEFAULT_EXECUTION_TIMEOUT = 300
    MAX_WORKFLOW_STEPS = 50


# Export all constants
__all__ = [
    "WorkflowCoordinationConstants",
    "ErrorHandlingCoordinatedConstants",
    "WorkflowConstants",
    "ErrorHandlingConstants",
    "WorkflowExecutionConstants",
]
