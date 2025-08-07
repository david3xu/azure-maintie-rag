"""
Safety Validation for Automated Constant Generation
===================================================

This module provides comprehensive safety validation for automatically generated
constants, ensuring system stability and preventing harmful adaptations.

Phase 3 Implementation: Safety-first automation with comprehensive validation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks"""

    RANGE_VALIDATION = "range_validation"
    TYPE_VALIDATION = "type_validation"
    CONSISTENCY_VALIDATION = "consistency_validation"
    SAFETY_CONSTRAINT = "safety_constraint"
    INTERDEPENDENCY = "interdependency"
    BUSINESS_LOGIC = "business_logic"
    PERFORMANCE_IMPACT = "performance_impact"


@dataclass
class ValidationIssue:
    """Individual validation issue"""

    constant_name: str
    constant_key: str
    issue_type: ValidationCategory
    severity: ValidationSeverity
    message: str
    suggested_value: Optional[Any] = None
    current_value: Optional[Any] = None
    constraint_violated: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation check"""

    passed: bool
    issues: List[ValidationIssue]
    confidence_adjustment: float = 0.0  # Adjustment to generation confidence
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConstantSafetyValidator:
    """Comprehensive safety validator for automated constant generation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define safety constraints for different constant types
        self._safety_constraints = self._initialize_safety_constraints()

        # Define interdependency rules
        self._interdependency_rules = self._initialize_interdependency_rules()

        # Define performance impact thresholds
        self._performance_thresholds = self._initialize_performance_thresholds()

    def _initialize_safety_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize safety constraints for different types of constants"""

        return {
            # Confidence and threshold constants
            "confidence_thresholds": {
                "min_value": 0.0,
                "max_value": 1.0,
                "recommended_min": 0.1,
                "recommended_max": 0.95,
                "pattern": r"(THRESHOLD|CONFIDENCE)",
                "type": float,
            },
            # Timeout constants (seconds)
            "timeout_constants": {
                "min_value": 1,
                "max_value": 3600,  # 1 hour max
                "recommended_min": 5,
                "recommended_max": 300,  # 5 minutes recommended max
                "pattern": r"(TIMEOUT|DELAY).*SECONDS?",
                "type": (int, float),
            },
            # Size and count constants
            "size_constants": {
                "min_value": 1,
                "max_value": 1000000,  # 1M max
                "recommended_min": 10,
                "recommended_max": 50000,
                "pattern": r"(SIZE|COUNT|LIMIT)(?!.*THRESHOLD)",
                "type": int,
            },
            # Chunk size constants
            "chunk_constants": {
                "min_value": 100,
                "max_value": 50000,
                "recommended_min": 300,
                "recommended_max": 5000,
                "pattern": r"CHUNK.*SIZE",
                "type": int,
            },
            # Batch size constants
            "batch_constants": {
                "min_value": 1,
                "max_value": 1000,
                "recommended_min": 5,
                "recommended_max": 100,
                "pattern": r"BATCH.*SIZE",
                "type": int,
            },
            # Rate and percentage constants
            "rate_constants": {
                "min_value": 0.0,
                "max_value": 1.0,
                "recommended_min": 0.01,
                "recommended_max": 1.0,
                "pattern": r"(RATE|PERCENTAGE|RATIO)",
                "type": float,
            },
            # Weight constants (should sum to 1.0 for groups)
            "weight_constants": {
                "min_value": 0.0,
                "max_value": 1.0,
                "recommended_min": 0.05,
                "recommended_max": 0.95,
                "pattern": r"WEIGHT",
                "type": float,
            },
            # Memory constants (MB)
            "memory_constants": {
                "min_value": 1,
                "max_value": 16384,  # 16GB max
                "recommended_min": 10,
                "recommended_max": 2048,  # 2GB recommended max
                "pattern": r"MEMORY.*MB",
                "type": int,
            },
        }

    def _initialize_interdependency_rules(self) -> List[Dict[str, Any]]:
        """Initialize interdependency validation rules"""

        return [
            # Entity vs Relationship confidence
            {
                "name": "entity_relationship_confidence",
                "primary": "ENTITY_CONFIDENCE_THRESHOLD",
                "secondary": "RELATIONSHIP_CONFIDENCE_THRESHOLD",
                "rule": "primary >= secondary",
                "message": "Entity confidence should be >= relationship confidence",
            },
            # Chunk size vs overlap
            {
                "name": "chunk_size_overlap",
                "primary": "DEFAULT_CHUNK_SIZE",
                "secondary": "DEFAULT_CHUNK_OVERLAP",
                "rule": "secondary < primary",
                "message": "Chunk overlap must be less than chunk size",
            },
            # Chunk overlap ratio
            {
                "name": "chunk_overlap_ratio",
                "primary": "DEFAULT_CHUNK_SIZE",
                "secondary": "DEFAULT_CHUNK_OVERLAP",
                "rule": "secondary <= primary * 0.5",
                "message": "Chunk overlap should not exceed 50% of chunk size",
            },
            # Min vs Max thresholds
            {
                "name": "min_max_thresholds",
                "primary": "MIN_CONFIDENCE_THRESHOLD",
                "secondary": "MAX_CONFIDENCE_THRESHOLD",
                "rule": "primary <= secondary",
                "message": "Min threshold must be <= max threshold",
            },
            # Vector top-k vs max results
            {
                "name": "vector_topk_limits",
                "primary": "VECTOR_TOP_K",
                "secondary": "MAX_SEARCH_RESULTS",
                "rule": "primary <= secondary",
                "message": "Vector top-k should not exceed max search results",
            },
            # Timeout consistency
            {
                "name": "timeout_consistency",
                "primary": "DEFAULT_TIMEOUT_SECONDS",
                "secondary": "SLOW_OPERATION_THRESHOLD_SECONDS",
                "rule": "secondary <= primary",
                "message": "Slow operation threshold should not exceed default timeout",
            },
            # Multi-modal weight sum
            {
                "name": "multimodal_weight_sum",
                "group": [
                    "MULTI_MODAL_WEIGHT_VECTOR",
                    "MULTI_MODAL_WEIGHT_GRAPH",
                    "MULTI_MODAL_WEIGHT_GNN",
                ],
                "rule": "abs(sum(group) - 1.0) <= 0.01",
                "message": "Multi-modal weights must sum to approximately 1.0",
            },
        ]

    def _initialize_performance_thresholds(self) -> Dict[str, float]:
        """Initialize performance impact thresholds"""

        return {
            "max_memory_increase_factor": 2.0,  # Max 2x memory increase
            "max_processing_time_factor": 3.0,  # Max 3x processing time increase
            "min_accuracy_threshold": 0.5,  # Don't allow accuracy below 50%
            "max_error_rate_threshold": 0.2,  # Don't allow error rate above 20%
            "min_cache_efficiency": 0.1,  # Minimum cache hit rate
            "max_resource_utilization": 0.9,  # Maximum resource utilization
        }

    async def validate_generated_constants(
        self,
        constant_name: str,
        generated_values: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> ValidationResult:
        """Comprehensive validation of generated constants"""

        context = context or {}
        issues = []

        try:
            # 1. Type and range validation
            type_issues = await self._validate_types_and_ranges(
                constant_name, generated_values
            )
            issues.extend(type_issues)

            # 2. Safety constraint validation
            safety_issues = await self._validate_safety_constraints(
                constant_name, generated_values
            )
            issues.extend(safety_issues)

            # 3. Interdependency validation
            interdep_issues = await self._validate_interdependencies(generated_values)
            issues.extend(interdep_issues)

            # 4. Business logic validation
            business_issues = await self._validate_business_logic(
                constant_name, generated_values
            )
            issues.extend(business_issues)

            # 5. Performance impact validation
            performance_issues = await self._validate_performance_impact(
                generated_values, context
            )
            issues.extend(performance_issues)

            # 6. Consistency validation
            consistency_issues = await self._validate_consistency(
                constant_name, generated_values, context
            )
            issues.extend(consistency_issues)

            # Determine overall validation result
            critical_issues = [
                i for i in issues if i.severity == ValidationSeverity.CRITICAL
            ]
            error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]

            passed = len(critical_issues) == 0 and len(error_issues) == 0

            # Calculate confidence adjustment based on issues
            confidence_adjustment = self._calculate_confidence_adjustment(issues)

            return ValidationResult(
                passed=passed,
                issues=issues,
                confidence_adjustment=confidence_adjustment,
                metadata={
                    "validation_timestamp": datetime.now().isoformat(),
                    "constant_name": constant_name,
                    "values_validated": len(generated_values),
                    "critical_issues": len(critical_issues),
                    "error_issues": len(error_issues),
                    "warning_issues": len(
                        [i for i in issues if i.severity == ValidationSeverity.WARNING]
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"Validation failed for {constant_name}: {e}")

            # Return failed validation with critical error
            return ValidationResult(
                passed=False,
                issues=[
                    ValidationIssue(
                        constant_name=constant_name,
                        constant_key="validation_system",
                        issue_type=ValidationCategory.SAFETY_CONSTRAINT,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation system error: {str(e)}",
                    )
                ],
                confidence_adjustment=-0.5,  # Significant confidence reduction
            )

    async def _validate_types_and_ranges(
        self, constant_name: str, generated_values: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate types and ranges of generated values"""

        issues = []

        for key, value in generated_values.items():
            # Find applicable constraint category
            constraint_category = self._find_constraint_category(key)

            if not constraint_category:
                # No specific constraints, do basic type check
                if not isinstance(value, (int, float, str, bool, list, dict)):
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.TYPE_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            message=f"Unsupported value type: {type(value)}",
                            current_value=value,
                        )
                    )
                continue

            constraints = self._safety_constraints[constraint_category]

            # Type validation
            expected_type = constraints["type"]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.TYPE_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            message=f"Invalid type. Expected {expected_type}, got {type(value)}",
                            current_value=value,
                        )
                    )
                    continue
            else:
                if not isinstance(value, expected_type):
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.TYPE_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            message=f"Invalid type. Expected {expected_type}, got {type(value)}",
                            current_value=value,
                        )
                    )
                    continue

            # Range validation
            if isinstance(value, (int, float)):
                min_val = constraints.get("min_value")
                max_val = constraints.get("max_value")
                rec_min = constraints.get("recommended_min")
                rec_max = constraints.get("recommended_max")

                # Hard limits
                if min_val is not None and value < min_val:
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.RANGE_VALIDATION,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Value {value} below minimum {min_val}",
                            current_value=value,
                            suggested_value=max(min_val, rec_min or min_val),
                        )
                    )

                if max_val is not None and value > max_val:
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.RANGE_VALIDATION,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Value {value} exceeds maximum {max_val}",
                            current_value=value,
                            suggested_value=min(max_val, rec_max or max_val),
                        )
                    )

                # Recommended ranges
                if rec_min is not None and value < rec_min:
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.RANGE_VALIDATION,
                            severity=ValidationSeverity.WARNING,
                            message=f"Value {value} below recommended minimum {rec_min}",
                            current_value=value,
                            suggested_value=rec_min,
                        )
                    )

                if rec_max is not None and value > rec_max:
                    issues.append(
                        ValidationIssue(
                            constant_name=constant_name,
                            constant_key=key,
                            issue_type=ValidationCategory.RANGE_VALIDATION,
                            severity=ValidationSeverity.WARNING,
                            message=f"Value {value} exceeds recommended maximum {rec_max}",
                            current_value=value,
                            suggested_value=rec_max,
                        )
                    )

        return issues

    async def _validate_safety_constraints(
        self, constant_name: str, generated_values: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate safety constraints to prevent system harm"""

        issues = []

        # Check for dangerous configurations
        dangerous_patterns = {
            "infinite_loop_risk": {
                "patterns": ["MAX_RETRIES", "RETRY_COUNT", "LOOP_LIMIT"],
                "max_safe_value": 10,
                "severity": ValidationSeverity.CRITICAL,
            },
            "memory_exhaustion_risk": {
                "patterns": ["CACHE_SIZE", "BUFFER_SIZE", "MAX_ITEMS"],
                "max_safe_value": 100000,
                "severity": ValidationSeverity.ERROR,
            },
            "timeout_too_short": {
                "patterns": ["TIMEOUT", "DEADLINE"],
                "min_safe_value": 1,
                "severity": ValidationSeverity.ERROR,
            },
            "precision_too_low": {
                "patterns": ["THRESHOLD", "CONFIDENCE"],
                "min_safe_value": 0.01,
                "severity": ValidationSeverity.WARNING,
            },
        }

        for key, value in generated_values.items():
            if not isinstance(value, (int, float)):
                continue

            for risk_name, risk_config in dangerous_patterns.items():
                if any(pattern in key.upper() for pattern in risk_config["patterns"]):

                    if (
                        "max_safe_value" in risk_config
                        and value > risk_config["max_safe_value"]
                    ):
                        issues.append(
                            ValidationIssue(
                                constant_name=constant_name,
                                constant_key=key,
                                issue_type=ValidationCategory.SAFETY_CONSTRAINT,
                                severity=risk_config["severity"],
                                message=f"Safety risk: {risk_name} - value {value} exceeds safe maximum {risk_config['max_safe_value']}",
                                current_value=value,
                                suggested_value=risk_config["max_safe_value"],
                                constraint_violated=risk_name,
                            )
                        )

                    if (
                        "min_safe_value" in risk_config
                        and value < risk_config["min_safe_value"]
                    ):
                        issues.append(
                            ValidationIssue(
                                constant_name=constant_name,
                                constant_key=key,
                                issue_type=ValidationCategory.SAFETY_CONSTRAINT,
                                severity=risk_config["severity"],
                                message=f"Safety risk: {risk_name} - value {value} below safe minimum {risk_config['min_safe_value']}",
                                current_value=value,
                                suggested_value=risk_config["min_safe_value"],
                                constraint_violated=risk_name,
                            )
                        )

        return issues

    async def _validate_interdependencies(
        self, generated_values: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate interdependency constraints between constants"""

        issues = []

        for rule in self._interdependency_rules:
            try:
                if "group" in rule:
                    # Group validation (e.g., weights sum to 1)
                    group_keys = rule["group"]
                    group_values = []

                    for key in group_keys:
                        if key in generated_values:
                            group_values.append(generated_values[key])

                    if (
                        len(group_values) >= 2
                    ):  # Need at least 2 values for group validation
                        # Evaluate rule
                        rule_code = rule["rule"].replace("group", "group_values")
                        try:
                            if not eval(
                                rule_code,
                                {"abs": abs, "sum": sum, "group_values": group_values},
                            ):
                                issues.append(
                                    ValidationIssue(
                                        constant_name="group_validation",
                                        constant_key=",".join(group_keys),
                                        issue_type=ValidationCategory.INTERDEPENDENCY,
                                        severity=ValidationSeverity.ERROR,
                                        message=f"Interdependency violation: {rule['message']}",
                                        current_value=group_values,
                                    )
                                )
                        except Exception:
                            # If evaluation fails, log warning but don't fail validation
                            self.logger.warning(
                                f"Failed to evaluate group rule: {rule['rule']}"
                            )

                else:
                    # Pairwise validation
                    primary_key = rule["primary"]
                    secondary_key = rule["secondary"]

                    if (
                        primary_key in generated_values
                        and secondary_key in generated_values
                    ):
                        primary_val = generated_values[primary_key]
                        secondary_val = generated_values[secondary_key]

                        # Evaluate rule
                        rule_code = (
                            rule["rule"]
                            .replace("primary", str(primary_val))
                            .replace("secondary", str(secondary_val))
                        )
                        try:
                            if not eval(rule_code):
                                issues.append(
                                    ValidationIssue(
                                        constant_name="interdependency_validation",
                                        constant_key=f"{primary_key},{secondary_key}",
                                        issue_type=ValidationCategory.INTERDEPENDENCY,
                                        severity=ValidationSeverity.ERROR,
                                        message=f"Interdependency violation: {rule['message']}",
                                        current_value={
                                            "primary": primary_val,
                                            "secondary": secondary_val,
                                        },
                                    )
                                )
                        except Exception:
                            # If evaluation fails, log warning
                            self.logger.warning(
                                f"Failed to evaluate interdependency rule: {rule['rule']}"
                            )

            except Exception as e:
                self.logger.error(
                    f"Error validating interdependency rule {rule.get('name', 'unknown')}: {e}"
                )

        return issues

    async def _validate_business_logic(
        self, constant_name: str, generated_values: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate business logic constraints"""

        issues = []

        # Domain-specific business rules
        business_rules = {
            "extraction_constants": {
                "required_pairs": [
                    (
                        "ENTITY_CONFIDENCE_THRESHOLD",
                        "RELATIONSHIP_CONFIDENCE_THRESHOLD",
                    ),
                    ("DEFAULT_CHUNK_SIZE", "DEFAULT_CHUNK_OVERLAP"),
                ],
                "logical_constraints": [
                    "ENTITY_CONFIDENCE_THRESHOLD should be higher than RELATIONSHIP_CONFIDENCE_THRESHOLD for quality",
                    "DEFAULT_CHUNK_OVERLAP should be 15-25% of DEFAULT_CHUNK_SIZE",
                ],
            },
            "search_constants": {
                "required_balance": [
                    ("VECTOR_SIMILARITY_THRESHOLD", "RESULT_RELEVANCE_THRESHOLD")
                ],
                "optimization_rules": [
                    "Higher similarity threshold should correspond to higher relevance threshold"
                ],
            },
        }

        # Apply business rules based on constant type
        for domain, rules in business_rules.items():
            if domain.lower() in constant_name.lower():

                # Check required pairs
                for pair in rules.get("required_pairs", []):
                    primary, secondary = pair
                    if primary in generated_values and secondary in generated_values:
                        primary_val = generated_values[primary]
                        secondary_val = generated_values[secondary]

                        # Apply domain-specific logic
                        if (
                            "ENTITY_CONFIDENCE" in primary
                            and "RELATIONSHIP_CONFIDENCE" in secondary
                        ):
                            if primary_val <= secondary_val:
                                issues.append(
                                    ValidationIssue(
                                        constant_name=constant_name,
                                        constant_key=f"{primary},{secondary}",
                                        issue_type=ValidationCategory.BUSINESS_LOGIC,
                                        severity=ValidationSeverity.WARNING,
                                        message="Entity confidence should typically be higher than relationship confidence",
                                        current_value={
                                            "entity": primary_val,
                                            "relationship": secondary_val,
                                        },
                                    )
                                )

                        elif "CHUNK_SIZE" in primary and "CHUNK_OVERLAP" in secondary:
                            overlap_ratio = (
                                secondary_val / primary_val if primary_val > 0 else 1
                            )
                            if overlap_ratio < 0.1 or overlap_ratio > 0.4:
                                issues.append(
                                    ValidationIssue(
                                        constant_name=constant_name,
                                        constant_key=f"{primary},{secondary}",
                                        issue_type=ValidationCategory.BUSINESS_LOGIC,
                                        severity=ValidationSeverity.WARNING,
                                        message=f"Chunk overlap ratio {overlap_ratio:.2f} is outside optimal range 0.15-0.25",
                                        current_value={
                                            "chunk_size": primary_val,
                                            "overlap": secondary_val,
                                        },
                                        suggested_value={
                                            "overlap": int(primary_val * 0.2)
                                        },
                                    )
                                )

        return issues

    async def _validate_performance_impact(
        self, generated_values: Dict[str, Any], context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate potential performance impact of generated constants"""

        issues = []

        # Get baseline performance metrics from context
        baseline_metrics = context.get("baseline_performance", {})

        # Estimate performance impact of key constants
        performance_estimates = self._estimate_performance_impact(
            generated_values, baseline_metrics
        )

        for metric, impact_factor in performance_estimates.items():
            threshold_key = f"max_{metric}_factor"
            max_factor = self._performance_thresholds.get(threshold_key, 2.0)

            if impact_factor > max_factor:
                severity = (
                    ValidationSeverity.CRITICAL
                    if impact_factor > max_factor * 1.5
                    else ValidationSeverity.ERROR
                )

                issues.append(
                    ValidationIssue(
                        constant_name="performance_impact",
                        constant_key=metric,
                        issue_type=ValidationCategory.PERFORMANCE_IMPACT,
                        severity=severity,
                        message=f"Estimated {metric} impact factor {impact_factor:.2f}x exceeds maximum {max_factor:.2f}x",
                        current_value=impact_factor,
                    )
                )

        return issues

    async def _validate_consistency(
        self,
        constant_name: str,
        generated_values: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """Validate consistency with existing system configuration"""

        issues = []

        # Check against current system constants
        current_constants = context.get("current_constants", {})

        for key, new_value in generated_values.items():
            if key in current_constants:
                current_value = current_constants[key]

                # Check for dramatic changes that might indicate instability
                if isinstance(new_value, (int, float)) and isinstance(
                    current_value, (int, float)
                ):
                    if current_value != 0:
                        change_ratio = abs(new_value - current_value) / abs(
                            current_value
                        )

                        if change_ratio > 5.0:  # More than 5x change
                            issues.append(
                                ValidationIssue(
                                    constant_name=constant_name,
                                    constant_key=key,
                                    issue_type=ValidationCategory.CONSISTENCY_VALIDATION,
                                    severity=ValidationSeverity.WARNING,
                                    message=f"Large change detected: {current_value} -> {new_value} ({change_ratio:.2f}x change)",
                                    current_value=new_value,
                                    suggested_value=current_value
                                    * min(2.0, max(0.5, new_value / current_value)),
                                )
                            )

        return issues

    def _find_constraint_category(self, key: str) -> Optional[str]:
        """Find the applicable constraint category for a constant key"""

        key_upper = key.upper()

        for category, constraints in self._safety_constraints.items():
            pattern = constraints.get("pattern", "")
            if pattern and re.search(pattern, key_upper):
                return category

        return None

    def _estimate_performance_impact(
        self, generated_values: Dict[str, Any], baseline_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate performance impact of generated constants"""

        impact_estimates = {}

        # Simple heuristic-based impact estimation
        for key, value in generated_values.items():
            if isinstance(value, (int, float)):

                # Memory impact estimation
                if any(
                    term in key.upper() for term in ["SIZE", "CACHE", "BUFFER", "BATCH"]
                ):
                    if "CACHE" in key.upper() or "BUFFER" in key.upper():
                        # Cache/buffer sizes directly impact memory
                        baseline_size = baseline_metrics.get(key, value)
                        if baseline_size > 0:
                            memory_factor = value / baseline_size
                            impact_estimates["memory_increase"] = max(
                                impact_estimates.get("memory_increase", 1.0),
                                memory_factor,
                            )

                # Processing time impact estimation
                if any(
                    term in key.upper()
                    for term in ["BATCH_SIZE", "CHUNK_SIZE", "MAX_ENTITIES"]
                ):
                    baseline_size = baseline_metrics.get(key, value)
                    if baseline_size > 0:
                        processing_factor = value / baseline_size
                        # Larger batches/chunks generally increase processing time
                        impact_estimates["processing_time"] = max(
                            impact_estimates.get("processing_time", 1.0),
                            processing_factor,
                        )

        return impact_estimates

    def _calculate_confidence_adjustment(self, issues: List[ValidationIssue]) -> float:
        """Calculate confidence adjustment based on validation issues"""

        adjustment = 0.0

        severity_weights = {
            ValidationSeverity.CRITICAL: -0.5,
            ValidationSeverity.ERROR: -0.2,
            ValidationSeverity.WARNING: -0.05,
            ValidationSeverity.INFO: 0.0,
        }

        for issue in issues:
            adjustment += severity_weights.get(issue.severity, 0.0)

        # Cap the adjustment to prevent extreme values
        return max(-0.8, min(0.0, adjustment))

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics"""

        return {
            "safety_constraints_count": len(self._safety_constraints),
            "interdependency_rules_count": len(self._interdependency_rules),
            "performance_thresholds_count": len(self._performance_thresholds),
            "constraint_categories": list(self._safety_constraints.keys()),
            "validation_categories": [
                category.value for category in ValidationCategory
            ],
            "severity_levels": [severity.value for severity in ValidationSeverity],
        }


# Global validator instance
constant_safety_validator = ConstantSafetyValidator()


# Export all classes and instances
__all__ = [
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationResult",
    "ConstantSafetyValidator",
    "constant_safety_validator",
]
