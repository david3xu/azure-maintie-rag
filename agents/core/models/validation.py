"""
Validation and Error Handling Models
===================================

Data models for validation results, error handling, and system monitoring.
These models provide comprehensive error context, validation outcomes,
and recovery strategies for the multi-agent system.

This module provides:
- Validation result models with detailed feedback
- Error context and categorization
- Error metrics and recovery tracking
- Performance feedback integration
- Quality assurance models
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from agents.core.constants import (
    ErrorHandlingConstants,
    MathematicalFoundationConstants,
    SystemPerformanceConstants,
)

from .base import ErrorCategory, ErrorSeverity, PydanticAIContextualModel

# =============================================================================
# VALIDATION RESULT MODELS
# =============================================================================


class ValidationResult(BaseModel):
    """Unified validation result model - eliminates duplicates"""

    domain: str = Field(description="Validation domain")
    valid: bool = Field(default=True, description="Validation success status")
    missing_keys: List[str] = Field(
        default_factory=list, description="Missing required keys"
    )
    invalid_values: List[str] = Field(
        default_factory=list, description="Invalid value descriptions"
    )
    source_validation: List[Dict[str, Any]] = Field(
        default_factory=list, description="Source-specific validation results"
    )
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    confidence: float = Field(
        default=MathematicalFoundationConstants.PERFECT_SCORE,
        ge=0.0,
        le=1.0,
        description="Validation confidence",
    )
    validation_time: float = Field(
        default=0.0, ge=0.0, description="Validation processing time"
    )


class ValidationResultPydanticAI(PydanticAIContextualModel):
    """Simple validation result using PydanticAI validation"""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    entity_count: int = Field(..., ge=0, description="Number of entities validated")
    relationship_count: int = Field(
        ..., ge=0, description="Number of relationships validated"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI validation operations"""
        return {
            "validation_result": {
                "is_valid": self.is_valid,
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
                "entity_count": self.entity_count,
                "relationship_count": self.relationship_count,
            },
            "validation_metrics": {
                "success_rate": 1.0 if self.is_valid else 0.0,
                "quality_indicators": {
                    "has_errors": len(self.errors) > 0,
                    "has_warnings": len(self.warnings) > 0,
                    "entity_validation": self.entity_count > 0,
                    "relationship_validation": self.relationship_count > 0,
                },
            },
        }


# =============================================================================
# ERROR HANDLING MODELS
# =============================================================================


class ErrorHandlingContract(BaseModel):
    """Error handling contract specification"""

    supported_error_types: List[ErrorCategory] = Field(
        description="Supported error categories"
    )
    error_recovery_strategies: Dict[str, str] = Field(
        description="Error recovery strategies"
    )
    escalation_rules: Dict[str, str] = Field(description="Error escalation rules")
    monitoring_requirements: Dict[str, Any] = Field(
        description="Error monitoring requirements"
    )
    notification_channels: List[str] = Field(description="Error notification channels")


@dataclass
class ErrorContext:
    """Comprehensive error context for analysis and recovery"""

    error: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    component: str
    parameters: Dict[str, Any] = None
    timestamp: float = None
    attempt_count: int = 0
    max_retries: int = SystemPerformanceConstants.DEFAULT_MAX_RETRIES
    recovery_strategy: Optional[str] = None
    user_message: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def should_retry(self) -> bool:
        """Determine if error should be retried"""
        return self.attempt_count < self.max_retries and self.category in [
            ErrorCategory.AZURE_SERVICE,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
        ]

    @property
    def backoff_delay(self) -> float:
        """Calculate exponential backoff delay"""
        base_delay = MathematicalFoundationConstants.BASE_DELAY_SECONDS
        return min(
            ErrorHandlingConstants.MAX_BACKOFF_DELAY_SECONDS,
            base_delay
            * (
                MathematicalFoundationConstants.EXPONENTIAL_BACKOFF_BASE
                ** self.attempt_count
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "severity": self.severity.value,
            "category": self.category.value,
            "operation": self.operation,
            "component": self.component,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "attempt_count": self.attempt_count,
            "max_retries": self.max_retries,
            "recovery_strategy": self.recovery_strategy,
            "user_message": self.user_message,
        }


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""

    total_errors: int = 0
    errors_by_category: Dict[str, int] = None
    errors_by_severity: Dict[str, int] = None
    errors_by_component: Dict[str, int] = None
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_recovery_time: float = 0.0
    recent_errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors_by_category is None:
            self.errors_by_category = {}
        if self.errors_by_severity is None:
            self.errors_by_severity = {}
        if self.errors_by_component is None:
            self.errors_by_component = {}
        if self.recent_errors is None:
            self.recent_errors = []

    @property
    def recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        total_recoveries = self.successful_recoveries + self.failed_recoveries
        if total_recoveries == 0:
            return 0.0
        return (self.successful_recoveries / total_recoveries) * 100

    def add_error(self, error_context: ErrorContext):
        """Add error to metrics tracking"""
        self.total_errors += 1

        # Track by category
        category = error_context.category.value
        self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1

        # Track by severity
        severity = error_context.severity.value
        self.errors_by_severity[severity] = self.errors_by_severity.get(severity, 0) + 1

        # Track by component
        component = error_context.component
        self.errors_by_component[component] = (
            self.errors_by_component.get(component, 0) + 1
        )

        # Add to recent errors (keep last 100)
        self.recent_errors.append(error_context.to_dict())
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)

    def record_recovery(self, success: bool, recovery_time: float):
        """Record error recovery attempt"""
        if success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1

        # Update average recovery time
        total_recoveries = self.successful_recoveries + self.failed_recoveries
        self.average_recovery_time = (
            self.average_recovery_time * (total_recoveries - 1) + recovery_time
        ) / total_recoveries


# =============================================================================
# PERFORMANCE FEEDBACK MODELS
# =============================================================================


class PerformanceFeedbackPoint(BaseModel):
    """Performance feedback point for dynamic configuration learning"""

    # Agent and operation identification
    agent_type: str = Field(description="Agent type (domain, extraction, search)")
    domain_name: str = Field(description="Domain context")
    operation_type: str = Field(description="Type of operation performed")

    # Configuration context
    configuration_used: Dict[str, Any] = Field(
        description="Configuration parameters used"
    )
    configuration_source: str = Field(
        description="Source of configuration (static, learned, adapted)"
    )

    # Performance metrics
    execution_time_seconds: float = Field(
        ge=0.0, description="Operation execution time"
    )
    success: bool = Field(description="Operation success status")
    quality_score: float = Field(ge=0.0, le=1.0, description="Operation quality score")

    # Quality breakdown
    output_quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Detailed quality metrics"
    )
    error_indicators: List[str] = Field(
        default_factory=list, description="Error or quality issues identified"
    )

    # Context for learning
    input_characteristics: Dict[str, Any] = Field(
        default_factory=dict, description="Input data characteristics"
    )
    environmental_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Environmental factors (load, time, etc.)"
    )

    # Feedback timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Feedback generation timestamp"
    )

    def calculate_overall_score(self) -> float:
        """Calculate overall performance score"""
        # Weight different factors
        time_score = min(
            1.0, 10.0 / max(0.1, self.execution_time_seconds)
        )  # Prefer faster
        success_score = 1.0 if self.success else 0.0
        quality_score = self.quality_score

        # Weighted average
        return time_score * 0.3 + success_score * 0.4 + quality_score * 0.3

    def should_trigger_adaptation(
        self, threshold: float = ErrorHandlingConstants.ADAPTATION_THRESHOLD
    ) -> bool:
        """Determine if this feedback should trigger configuration adaptation"""
        return self.calculate_overall_score() < threshold

    def get_adaptation_suggestions(self) -> List[str]:
        """Generate adaptation suggestions based on feedback"""
        suggestions = []

        if self.execution_time_seconds > 5.0:
            suggestions.append("Consider reducing processing complexity")
            suggestions.append("Enable parallel processing")
            suggestions.append("Increase cache TTL")

        if not self.success:
            suggestions.append("Adjust confidence thresholds")
            suggestions.append("Enable additional validation")
            suggestions.append("Increase retry attempts")

        if self.quality_score < 0.7:
            suggestions.append("Fine-tune extraction parameters")
            suggestions.append("Improve preprocessing steps")
            suggestions.append("Enhance domain-specific patterns")

        return suggestions


# =============================================================================
# QUALITY ASSURANCE MODELS
# =============================================================================


class QualityGate(BaseModel):
    """Quality gate for validation and approval processes"""

    gate_name: str = Field(description="Quality gate identifier")
    criteria: Dict[str, Any] = Field(description="Gate criteria and thresholds")
    passed: bool = Field(description="Whether gate passed")
    score: float = Field(ge=0.0, le=1.0, description="Gate quality score")

    # Detailed results
    criterion_results: Dict[str, bool] = Field(
        default_factory=dict, description="Individual criterion results"
    )
    warnings: List[str] = Field(default_factory=list, description="Quality warnings")
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    # Metadata
    evaluation_time: float = Field(
        default=0.0, ge=0.0, description="Time taken for evaluation"
    )
    evaluator: str = Field(description="Component that performed evaluation")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Evaluation timestamp"
    )

    def get_failing_criteria(self) -> List[str]:
        """Get list of criteria that failed"""
        return [
            criterion
            for criterion, passed in self.criterion_results.items()
            if not passed
        ]

    def get_success_rate(self) -> float:
        """Calculate success rate across criteria"""
        if not self.criterion_results:
            return 1.0 if self.passed else 0.0

        passed_count = sum(1 for passed in self.criterion_results.values() if passed)
        return passed_count / len(self.criterion_results)


class ValidationSummary(BaseModel):
    """Summary of validation results across multiple components"""

    validation_id: str = Field(description="Validation session identifier")
    overall_valid: bool = Field(description="Overall validation status")

    # Component validation results
    component_results: Dict[str, ValidationResult] = Field(
        description="Validation results by component"
    )
    quality_gates: List[QualityGate] = Field(
        default_factory=list, description="Quality gates evaluated"
    )

    # Summary metrics
    total_errors: int = Field(ge=0, description="Total validation errors")
    total_warnings: int = Field(ge=0, description="Total validation warnings")
    overall_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall validation confidence"
    )

    # Performance
    total_validation_time: float = Field(ge=0.0, description="Total validation time")
    validation_efficiency: float = Field(
        ge=0.0, description="Validation items per second"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Validation completion timestamp"
    )
    validator_version: str = Field(description="Validator version")

    def get_failed_components(self) -> List[str]:
        """Get list of components that failed validation"""
        return [
            component
            for component, result in self.component_results.items()
            if not result.valid
        ]

    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score from all gates"""
        if not self.quality_gates:
            return 1.0 if self.overall_valid else 0.0

        total_score = sum(gate.score for gate in self.quality_gates)
        return total_score / len(self.quality_gates)

    def generate_improvement_plan(self) -> Dict[str, List[str]]:
        """Generate improvement plan based on validation results"""
        plan = {}

        # Component-specific improvements
        for component, result in self.component_results.items():
            if not result.valid:
                improvements = []
                if result.missing_keys:
                    improvements.append(
                        f"Add missing required fields: {', '.join(result.missing_keys)}"
                    )
                if result.invalid_values:
                    improvements.append(
                        f"Fix invalid values: {', '.join(result.invalid_values)}"
                    )
                if improvements:
                    plan[component] = improvements

        # Quality gate improvements
        for gate in self.quality_gates:
            if not gate.passed and gate.recommendations:
                gate_key = f"quality_gate_{gate.gate_name}"
                plan[gate_key] = gate.recommendations

        return plan
