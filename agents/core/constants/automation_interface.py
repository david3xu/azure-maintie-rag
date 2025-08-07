"""
Automation Interface for Dynamic Constant Generation
===================================================

This module provides the interface layer between the classification system and
the actual automation mechanisms. It coordinates constant generation, validation,
and integration with the existing system.

Phase 3 Implementation: Automation coordination and dynamic generation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from .automation_classifier import (
    AutomationPotential,
    LearningMechanism,
    ConstantClassification,
    automation_classifier,
)


@dataclass
class GenerationRequest:
    """Request for constant generation"""

    constant_name: str
    learning_mechanisms: List[LearningMechanism]
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 2=medium, 3=high
    deadline: Optional[datetime] = None
    safety_constraints: List[str] = field(default_factory=list)
    interdependent_with: List[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of constant generation"""

    constant_name: str
    generated_values: Dict[str, Any]
    confidence_score: float
    learning_source: str
    generation_timestamp: datetime
    validation_passed: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomationCoordinator:
    """Coordinates automation of constant generation across the system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._generation_queue: List[GenerationRequest] = []
        self._generation_results: Dict[str, GenerationResult] = {}
        self._active_generations: Dict[str, asyncio.Task] = {}

    async def queue_generation_request(self, request: GenerationRequest) -> str:
        """Queue a constant generation request"""

        # Validate the request
        classification = automation_classifier.get_classification(request.constant_name)
        if not classification:
            raise ValueError(f"Unknown constant: {request.constant_name}")

        if classification.automation_potential == AutomationPotential.STATIC:
            raise ValueError(
                f"Constant {request.constant_name} cannot be auto-generated (STATIC)"
            )

        # Check if already queued or in progress
        if request.constant_name in self._active_generations:
            return f"Generation already in progress for {request.constant_name}"

        # Add to queue
        self._generation_queue.append(request)
        self.logger.info(f"Queued generation request for {request.constant_name}")

        return f"Generation queued for {request.constant_name}"

    async def process_generation_queue(self) -> List[GenerationResult]:
        """Process all pending generation requests"""

        if not self._generation_queue:
            return []

        # Sort by priority and deadline
        self._generation_queue.sort(
            key=lambda x: (x.priority, x.deadline or datetime.max), reverse=True
        )

        results = []
        for request in self._generation_queue[
            :
        ]:  # Copy to avoid modification during iteration
            try:
                result = await self._generate_constants(request)
                results.append(result)
                self._generation_queue.remove(request)
            except Exception as e:
                self.logger.error(f"Generation failed for {request.constant_name}: {e}")
                error_result = GenerationResult(
                    constant_name=request.constant_name,
                    generated_values={},
                    confidence_score=0.0,
                    learning_source="error",
                    generation_timestamp=datetime.now(),
                    validation_passed=False,
                    error_message=str(e),
                )
                results.append(error_result)
                self._generation_queue.remove(request)

        return results

    async def _generate_constants(self, request: GenerationRequest) -> GenerationResult:
        """Generate constants using appropriate learning mechanisms"""

        classification = automation_classifier.get_classification(request.constant_name)
        if not classification:
            raise ValueError(f"No classification found for {request.constant_name}")

        generated_values = {}
        learning_sources = []
        total_confidence = 0.0

        # Process each learning mechanism
        for mechanism in request.learning_mechanisms:
            try:
                values, confidence, source = await self._apply_learning_mechanism(
                    mechanism, request.constant_name, request.context
                )

                # Merge values (later mechanisms can override earlier ones)
                generated_values.update(values)
                learning_sources.append(source)
                total_confidence += confidence

            except Exception as e:
                self.logger.warning(
                    f"Learning mechanism {mechanism} failed for {request.constant_name}: {e}"
                )
                continue

        # Calculate average confidence
        avg_confidence = (
            total_confidence / len(request.learning_mechanisms)
            if request.learning_mechanisms
            else 0.0
        )

        # Create result
        result = GenerationResult(
            constant_name=request.constant_name,
            generated_values=generated_values,
            confidence_score=avg_confidence,
            learning_source=" + ".join(learning_sources),
            generation_timestamp=datetime.now(),
            metadata={
                "classification": classification.name,
                "automation_potential": classification.automation_potential.value,
                "learning_mechanisms_used": [
                    m.value for m in request.learning_mechanisms
                ],
                "interdependent_groups": classification.interdependent_groups or [],
            },
        )

        # Validate the result
        result.validation_passed = await self._validate_generated_constants(
            result, request
        )

        # Store result
        self._generation_results[request.constant_name] = result

        return result

    async def _apply_learning_mechanism(
        self, mechanism: LearningMechanism, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Apply a specific learning mechanism to generate constants"""

        if mechanism == LearningMechanism.DOMAIN_ANALYSIS:
            return await self._domain_analysis_learning(constant_name, context)

        elif mechanism == LearningMechanism.PERFORMANCE_FEEDBACK:
            return await self._performance_feedback_learning(constant_name, context)

        elif mechanism == LearningMechanism.HYPERPARAMETER_OPT:
            return await self._hyperparameter_optimization(constant_name, context)

        elif mechanism == LearningMechanism.AZURE_DISCOVERY:
            return await self._azure_discovery_learning(constant_name, context)

        elif mechanism == LearningMechanism.USAGE_PATTERNS:
            return await self._usage_pattern_learning(constant_name, context)

        elif mechanism == LearningMechanism.CORRELATION_ANALYSIS:
            return await self._correlation_analysis_learning(constant_name, context)

        elif mechanism == LearningMechanism.QUALITY_OPTIMIZATION:
            return await self._quality_optimization_learning(constant_name, context)

        else:
            raise ValueError(f"Unknown learning mechanism: {mechanism}")

    async def _domain_analysis_learning(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants from domain analysis using Domain Intelligence integration"""

        try:
            # Import the domain intelligence generator
            from .domain_intelligence_integration import domain_intelligence_generator

            # Get domain name from context
            domain_name = context.get("domain_name", "general")

            if constant_name == "DomainAdaptiveConstants":
                return await domain_intelligence_generator.generate_domain_adaptive_constants(
                    domain_name, context
                )
            elif constant_name == "ContentAnalysisAdaptiveConstants":
                return await domain_intelligence_generator.generate_content_analysis_constants(
                    domain_name, context
                )
            else:
                # Generic domain analysis for other constants
                domain_analysis = (
                    await domain_intelligence_generator._get_domain_analysis(
                        domain_name, context
                    )
                )

                if domain_analysis:
                    # Extract relevant values based on constant type
                    values = {}
                    confidence = domain_analysis.confidence_score

                    if "extraction" in constant_name.lower():
                        values["DEFAULT_CONFIDENCE_THRESHOLD"] = min(
                            0.9, max(0.6, domain_analysis.entity_density * 5)
                        )
                        values["MAX_ENTITIES_PER_DOCUMENT"] = min(
                            200, max(50, int(domain_analysis.entity_density * 500))
                        )

                    if "search" in constant_name.lower():
                        values["VECTOR_SIMILARITY_THRESHOLD"] = (
                            0.65 if domain_analysis.vocabulary_richness > 0.3 else 0.70
                        )
                        values["VECTOR_TOP_K"] = (
                            15 if domain_analysis.document_complexity == "high" else 10
                        )

                    return (
                        values,
                        confidence * 0.8,
                        f"Domain Analysis ({domain_analysis.confidence_score:.3f})",
                    )

                return {}, 0.0, "no_domain_analysis"

        except Exception as e:
            self.logger.error(f"Domain analysis learning failed: {e}")
            return {}, 0.0, f"domain_analysis_error: {str(e)}"

    async def _performance_feedback_learning(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants from performance feedback using feedback loops"""

        try:
            # Import the performance feedback orchestrator
            from .performance_feedback_loops import performance_feedback_orchestrator

            # Process current feedback to get adaptations
            feedback_results = (
                await performance_feedback_orchestrator.process_feedback_loop()
            )

            # Extract adaptations relevant to the requested constant
            relevant_adaptations = {}
            confidence = 0.0
            adaptation_sources = []

            for adaptation in feedback_results.get("triggered_adaptations", []):
                if constant_name in adaptation.get("target_constants", []):
                    relevant_adaptations.update(adaptation.get("adaptations", {}))
                    adaptation_sources.append(adaptation["rule"])

            # If no recent adaptations, use performance metrics directly
            if not relevant_adaptations:
                performance_metrics = context.get("performance_metrics", {})
                relevant_adaptations, confidence = (
                    await self._direct_performance_adaptation(
                        constant_name, performance_metrics
                    )
                )
                adaptation_sources = ["direct_metrics"]
            else:
                confidence = 0.8  # High confidence from feedback loop adaptations

            source_description = (
                f"Performance Feedback ({', '.join(adaptation_sources)})"
            )

            return relevant_adaptations, confidence, source_description

        except Exception as e:
            self.logger.error(f"Performance feedback learning failed: {e}")
            return {}, 0.0, f"performance_feedback_error: {str(e)}"

    async def _direct_performance_adaptation(
        self, constant_name: str, performance_metrics: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """Direct performance adaptation when no feedback loop adaptations are available"""

        values = {}
        confidence = 0.6  # Medium confidence for direct adaptation

        if constant_name == "PerformanceAdaptiveConstants":

            if "average_response_time" in performance_metrics:
                # Adapt timeouts based on actual performance
                avg_response = performance_metrics["average_response_time"]
                values["ADAPTIVE_TIMEOUT_SECONDS"] = min(60, max(10, avg_response * 2))

            if "cache_hit_rate" in performance_metrics:
                # Optimize cache parameters based on hit rate
                hit_rate = performance_metrics["cache_hit_rate"]
                if hit_rate < 0.5:  # Low hit rate, increase cache size
                    values["OPTIMAL_CACHE_SIZE"] = min(
                        10000, performance_metrics.get("current_cache_size", 1000) * 1.5
                    )
                elif hit_rate > 0.8:  # High hit rate, can reduce cache size
                    values["OPTIMAL_CACHE_SIZE"] = max(
                        100, performance_metrics.get("current_cache_size", 1000) * 0.8
                    )

            if "concurrent_users" in performance_metrics:
                # Adapt concurrency based on load
                users = performance_metrics["concurrent_users"]
                values["OPTIMAL_CONCURRENT_REQUESTS"] = min(50, max(5, users))

        elif constant_name == "SystemPerformanceConstants":

            if "average_response_time" in performance_metrics:
                avg_response = performance_metrics["average_response_time"]
                values["DEFAULT_TIMEOUT_SECONDS"] = min(
                    120, max(15, int(avg_response * 3))
                )

            if "error_rate" in performance_metrics:
                error_rate = performance_metrics["error_rate"]
                if error_rate > 0.05:  # High error rate
                    values["DEFAULT_MAX_RETRIES"] = min(
                        5, max(1, 3 + int(error_rate * 10))
                    )

        return values, confidence

    async def _hyperparameter_optimization(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants through hyperparameter optimization"""

        if constant_name == "MLModelStaticConstants":
            # This would integrate with actual hyperparameter optimization
            optimization_results = context.get("hyperparameter_results", {})

            values = {}
            if "optimal_learning_rate" in optimization_results:
                values["GNN_LEARNING_RATE"] = optimization_results[
                    "optimal_learning_rate"
                ]

            if "optimal_batch_size" in optimization_results:
                values["BATCH_SIZE"] = optimization_results["optimal_batch_size"]

            return values, 0.80, "Hyperparameter Optimization"

        return {}, 0.0, "hyperparameter_opt_placeholder"

    async def _azure_discovery_learning(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants from Azure deployment scanning"""

        azure_config = context.get("azure_deployment", {})

        if constant_name == "InfrastructureConstants":
            values = {}

            if "openai_model_deployments" in azure_config:
                # Discover available models
                deployments = azure_config["openai_model_deployments"]
                if "text-embedding-3-large" in deployments:
                    values["DEFAULT_EMBEDDING_MODEL"] = "text-embedding-3-large"
                elif "text-embedding-ada-002" in deployments:
                    values["DEFAULT_EMBEDDING_MODEL"] = "text-embedding-ada-002"

            if "cosmos_throughput" in azure_config:
                values["MIN_COSMOS_THROUGHPUT_RU"] = azure_config["cosmos_throughput"]

            return values, 0.90, "Azure Deployment Scanner"

        return {}, 0.0, "azure_discovery_placeholder"

    async def _usage_pattern_learning(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants from usage patterns"""

        usage_stats = context.get("usage_statistics", {})

        if "query_complexity_distribution" in usage_stats:
            complexity_dist = usage_stats["query_complexity_distribution"]

            values = {}
            if "average_query_length" in complexity_dist:
                avg_length = complexity_dist["average_query_length"]
                values["TYPICAL_QUERY_LENGTH"] = avg_length
                values["QUERY_PROCESSING_TIMEOUT"] = min(30, max(5, avg_length * 0.5))

            return values, 0.70, "Usage Pattern Analysis"

        return {}, 0.0, "usage_patterns_placeholder"

    async def _correlation_analysis_learning(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants through correlation analysis of interdependent groups"""

        correlation_data = context.get("correlation_analysis", {})

        if constant_name == "WorkflowCoordinationConstants":
            values = {}

            if "quality_performance_correlation" in correlation_data:
                # Optimize coordinated thresholds based on correlations
                correlation = correlation_data["quality_performance_correlation"]
                if correlation > 0.8:  # Strong positive correlation
                    values["MIN_EXTRACTION_ACCURACY"] = (
                        0.90  # Increase quality requirement
                    )
                    values["MIN_SEARCH_RELEVANCE"] = (
                        0.75  # Correspondingly increase search requirement
                    )

            return values, 0.75, "Correlation Analysis"

        return {}, 0.0, "correlation_analysis_placeholder"

    async def _quality_optimization_learning(
        self, constant_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Learn constants through quality optimization"""

        quality_metrics = context.get("quality_assessment", {})

        values = {}
        if "extraction_accuracy_by_threshold" in quality_metrics:
            # Optimize thresholds based on quality curves
            accuracy_curve = quality_metrics["extraction_accuracy_by_threshold"]
            optimal_threshold = max(accuracy_curve, key=accuracy_curve.get)
            values["OPTIMAL_CONFIDENCE_THRESHOLD"] = optimal_threshold

        return values, 0.80, "Quality Optimization"

    async def _validate_generated_constants(
        self, result: GenerationResult, request: GenerationRequest
    ) -> bool:
        """Validate generated constants using comprehensive safety validation"""

        try:
            # Import the safety validator
            from .safety_validation import constant_safety_validator

            # Prepare validation context
            context = {
                "current_constants": request.context.get("current_constants", {}),
                "baseline_performance": request.context.get("performance_metrics", {}),
                "domain_name": request.context.get("domain_name"),
                "generation_metadata": {
                    "learning_mechanisms": [
                        m.value for m in request.learning_mechanisms
                    ],
                    "priority": request.priority,
                    "interdependent_with": request.interdependent_with,
                },
            }

            # Perform comprehensive validation
            validation_result = (
                await constant_safety_validator.validate_generated_constants(
                    request.constant_name, result.generated_values, context
                )
            )

            # Update generation result with validation information
            result.metadata["validation_result"] = {
                "passed": validation_result.passed,
                "issues_count": len(validation_result.issues),
                "critical_issues": [
                    {"key": issue.constant_key, "message": issue.message}
                    for issue in validation_result.issues
                    if issue.severity.value == "critical"
                ],
                "error_issues": [
                    {"key": issue.constant_key, "message": issue.message}
                    for issue in validation_result.issues
                    if issue.severity.value == "error"
                ],
                "confidence_adjustment": validation_result.confidence_adjustment,
            }

            # Apply confidence adjustment
            if validation_result.confidence_adjustment != 0:
                result.confidence_score = max(
                    0.0,
                    min(
                        1.0,
                        result.confidence_score
                        + validation_result.confidence_adjustment,
                    ),
                )

            # Log validation issues
            if validation_result.issues:
                critical_count = len(
                    [
                        i
                        for i in validation_result.issues
                        if i.severity.value == "critical"
                    ]
                )
                error_count = len(
                    [i for i in validation_result.issues if i.severity.value == "error"]
                )
                warning_count = len(
                    [
                        i
                        for i in validation_result.issues
                        if i.severity.value == "warning"
                    ]
                )

                self.logger.info(
                    f"Validation for {request.constant_name}: "
                    f"{critical_count} critical, {error_count} errors, {warning_count} warnings"
                )

                # Log critical and error issues in detail
                for issue in validation_result.issues:
                    if issue.severity.value in ["critical", "error"]:
                        self.logger.warning(
                            f"Validation issue [{issue.severity.value}] {issue.constant_key}: {issue.message}"
                        )

            # Set error message if validation failed
            if not validation_result.passed:
                critical_messages = [
                    issue.message
                    for issue in validation_result.issues
                    if issue.severity.value == "critical"
                ]
                error_messages = [
                    issue.message
                    for issue in validation_result.issues
                    if issue.severity.value == "error"
                ]

                all_messages = critical_messages + error_messages
                result.error_message = "; ".join(
                    all_messages[:3]
                )  # Limit to first 3 messages

                if len(all_messages) > 3:
                    result.error_message += (
                        f" (and {len(all_messages) - 3} more issues)"
                    )

            return validation_result.passed

        except Exception as e:
            self.logger.error(
                f"Validation system error for {request.constant_name}: {e}"
            )
            result.error_message = f"Validation system error: {str(e)}"
            return False

    async def _check_safety_constraint(
        self, result: GenerationResult, constraint: str
    ) -> bool:
        """Check a specific safety constraint"""

        # Implement specific safety constraint checks
        if constraint == "Mathematical standards":
            # Ensure mathematical constants are within expected ranges
            return True  # Placeholder

        elif constraint == "Hardware limits":
            # Ensure values don't exceed system capabilities
            return True  # Placeholder

        elif constraint == "System compatibility":
            # Ensure values maintain system compatibility
            return True  # Placeholder

        return True  # Default to safe

    async def _validate_constant_value(self, key: str, value: Any) -> bool:
        """Validate a specific constant value"""

        # Type and range validation
        if "THRESHOLD" in key.upper() or "CONFIDENCE" in key.upper():
            # Confidence/threshold values should be between 0 and 1
            if isinstance(value, (int, float)):
                return 0.0 <= value <= 1.0

        elif "SIZE" in key.upper() or "COUNT" in key.upper():
            # Size/count values should be positive integers
            if isinstance(value, int):
                return value > 0

        elif "TIMEOUT" in key.upper() or "DELAY" in key.upper():
            # Timeout values should be positive numbers with reasonable upper bounds
            if isinstance(value, (int, float)):
                return 0 < value <= 600  # Max 10 minutes

        return True  # Default to valid

    async def _validate_interdependencies(
        self, result: GenerationResult, interdependent_groups: List[str]
    ) -> bool:
        """Validate interdependency constraints between constants"""

        # Check if interdependent constants maintain proper relationships
        values = result.generated_values

        # Example: Entity confidence should be >= relationship confidence
        if (
            "ENTITY_CONFIDENCE_THRESHOLD" in values
            and "RELATIONSHIP_CONFIDENCE_THRESHOLD" in values
        ):
            return (
                values["ENTITY_CONFIDENCE_THRESHOLD"]
                >= values["RELATIONSHIP_CONFIDENCE_THRESHOLD"]
            )

        # Example: Chunk overlap should be < chunk size
        if "DEFAULT_CHUNK_SIZE" in values and "DEFAULT_CHUNK_OVERLAP" in values:
            return values["DEFAULT_CHUNK_OVERLAP"] < values["DEFAULT_CHUNK_SIZE"]

        return True  # Default to valid

    def get_generation_results(self) -> Dict[str, GenerationResult]:
        """Get all generation results"""
        return self._generation_results.copy()

    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation status"""

        summary = automation_classifier.get_automation_summary()
        mechanism_summary = automation_classifier.get_learning_mechanism_summary()

        return {
            "total_constants": sum(summary.values()),
            "automation_potential_summary": {k.value: v for k, v in summary.items()},
            "learning_mechanism_summary": {
                k.value: v for k, v in mechanism_summary.items()
            },
            "active_generations": len(self._active_generations),
            "queued_requests": len(self._generation_queue),
            "completed_generations": len(self._generation_results),
            "interdependent_groups": len(
                automation_classifier.get_interdependent_groups()
            ),
        }


# Global coordinator instance
automation_coordinator = AutomationCoordinator()


# Export all classes and instances
__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "AutomationCoordinator",
    "automation_coordinator",
]
