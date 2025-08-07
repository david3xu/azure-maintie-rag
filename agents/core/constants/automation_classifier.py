"""
Automation Classification System
===============================

This module provides systematic classification of constants by their automation potential
and learning mechanisms. It serves as the master registry for which constants can be
dynamically generated, learned, or must remain static.

Phase 3 Implementation: Dynamic Configuration and Automation
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


class AutomationPotential(Enum):
    """Classification of automation potential for constants"""

    STATIC = "static"  # Never auto-generate (system limits, standards)
    LOW = "low"  # Rarely change (conventional paths, basic thresholds)
    MEDIUM = "medium"  # Moderately learnable (performance thresholds, error handling)
    HIGH = "high"  # Highly learnable (domain patterns, search quality)
    VERY_HIGH = (
        "very_high"  # Should be auto-generated (domain-adaptive, performance-adaptive)
    )


class LearningMechanism(Enum):
    """Types of learning mechanisms for constant generation"""

    DOMAIN_ANALYSIS = "domain_analysis"  # Learn from document corpus analysis
    PERFORMANCE_FEEDBACK = (
        "performance_feedback"  # Learn from system performance metrics
    )
    HYPERPARAMETER_OPT = "hyperparameter_opt"  # Learn through optimization algorithms
    AZURE_DISCOVERY = "azure_discovery"  # Discover from Azure deployment scanning
    USAGE_PATTERNS = "usage_patterns"  # Learn from user interaction patterns
    CORRELATION_ANALYSIS = (
        "correlation_analysis"  # Learn from interdependent relationships
    )
    QUALITY_OPTIMIZATION = (
        "quality_optimization"  # Learn from quality assessment results
    )


@dataclass
class ConstantClassification:
    """Classification metadata for a constant or constant group"""

    name: str
    automation_potential: AutomationPotential
    learning_mechanisms: List[LearningMechanism]
    interdependent_groups: Optional[List[str]] = None
    safety_constraints: Optional[List[str]] = None
    current_source: str = "hardcoded"
    target_source: Optional[str] = None


class AutomationClassifier:
    """Master classification system for constant automation potential"""

    def __init__(self):
        self._classifications = self._build_classifications()

    def _build_classifications(self) -> Dict[str, ConstantClassification]:
        """Build comprehensive classification registry"""

        classifications = {}

        # === BASE CONSTANTS (STATIC - Mathematical foundations) ===
        classifications.update(
            {
                "MathematicalConstants": ConstantClassification(
                    name="MathematicalConstants",
                    automation_potential=AutomationPotential.STATIC,
                    learning_mechanisms=[],
                    safety_constraints=[
                        "Mathematical standards",
                        "System compatibility",
                    ],
                ),
                "BaseScalingFactors": ConstantClassification(
                    name="BaseScalingFactors",
                    automation_potential=AutomationPotential.STATIC,
                    learning_mechanisms=[],
                    safety_constraints=[
                        "Algebraic relationships",
                        "Derived value consistency",
                    ],
                ),
                "MathematicalFoundationConstants": ConstantClassification(
                    name="MathematicalFoundationConstants",
                    automation_potential=AutomationPotential.STATIC,
                    learning_mechanisms=[],
                    safety_constraints=[
                        "Perfect mathematical values",
                        "Model integration consistency",
                    ],
                ),
            }
        )

        # === SYSTEM CONSTANTS (STATIC to HIGH automation) ===
        classifications.update(
            {
                "SystemBoundaryConstants": ConstantClassification(
                    name="SystemBoundaryConstants",
                    automation_potential=AutomationPotential.STATIC,
                    learning_mechanisms=[],
                    safety_constraints=[
                        "Hardware limits",
                        "API constraints",
                        "Security boundaries",
                    ],
                ),
                "InfrastructureConstants": ConstantClassification(
                    name="InfrastructureConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.AZURE_DISCOVERY,
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                    ],
                    target_source="Azure deployment scanning",
                ),
                "SystemPerformanceConstants": ConstantClassification(
                    name="SystemPerformanceConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.USAGE_PATTERNS,
                    ],
                    target_source="Performance monitoring",
                ),
                "FileSystemConstants": ConstantClassification(
                    name="FileSystemConstants",
                    automation_potential=AutomationPotential.LOW,
                    learning_mechanisms=[],
                    safety_constraints=["Conventional naming", "System compatibility"],
                ),
            }
        )

        # === DOMAIN CONSTANTS (VERY HIGH automation) ===
        classifications.update(
            {
                "DomainAdaptiveConstants": ConstantClassification(
                    name="DomainAdaptiveConstants",
                    automation_potential=AutomationPotential.VERY_HIGH,
                    learning_mechanisms=[
                        LearningMechanism.DOMAIN_ANALYSIS,
                        LearningMechanism.CORRELATION_ANALYSIS,
                        LearningMechanism.QUALITY_OPTIMIZATION,
                    ],
                    interdependent_groups=[
                        "Entity Extraction Thresholds",
                        "Document Processing Parameters",
                        "Search Quality Thresholds",
                        "Domain Classification Thresholds",
                    ],
                    target_source="Domain Intelligence Agent",
                ),
                "ContentAnalysisAdaptiveConstants": ConstantClassification(
                    name="ContentAnalysisAdaptiveConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.DOMAIN_ANALYSIS,
                        LearningMechanism.USAGE_PATTERNS,
                    ],
                    target_source="Corpus analysis and document structure patterns",
                ),
                "DomainIntelligenceConstants": ConstantClassification(
                    name="DomainIntelligenceConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[
                        LearningMechanism.HYPERPARAMETER_OPT,
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                    ],
                    target_source="Agent performance optimization",
                ),
            }
        )

        # === EXTRACTION CONSTANTS (MEDIUM-HIGH automation) ===
        classifications.update(
            {
                "ExtractionAlgorithmConstants": ConstantClassification(
                    name="ExtractionAlgorithmConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.QUALITY_OPTIMIZATION,
                    ],
                    target_source="Extraction performance analysis",
                ),
                "KnowledgeExtractionConstants": ConstantClassification(
                    name="KnowledgeExtractionConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.DOMAIN_ANALYSIS,
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.QUALITY_OPTIMIZATION,
                    ],
                    target_source="Domain extraction patterns and performance",
                ),
                "ExtractionQualityConstants": ConstantClassification(
                    name="ExtractionQualityConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[
                        LearningMechanism.QUALITY_OPTIMIZATION,
                        LearningMechanism.DOMAIN_ANALYSIS,
                    ],
                    target_source="Quality assessment optimization",
                ),
            }
        )

        # === SEARCH CONSTANTS (MEDIUM-HIGH automation) ===
        classifications.update(
            {
                "MLModelStaticConstants": ConstantClassification(
                    name="MLModelStaticConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[LearningMechanism.HYPERPARAMETER_OPT],
                    target_source="Hyperparameter optimization and architecture search",
                ),
                "StatisticalConstants": ConstantClassification(
                    name="StatisticalConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[
                        LearningMechanism.DOMAIN_ANALYSIS,
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                    ],
                    target_source="Domain statistical analysis",
                ),
                "UniversalSearchConstants": ConstantClassification(
                    name="UniversalSearchConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.USAGE_PATTERNS,
                        LearningMechanism.QUALITY_OPTIMIZATION,
                    ],
                    target_source="Search quality optimization and usage analysis",
                ),
            }
        )

        # === WORKFLOW CONSTANTS (MEDIUM-HIGH automation) ===
        classifications.update(
            {
                "WorkflowCoordinationConstants": ConstantClassification(
                    name="WorkflowCoordinationConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.CORRELATION_ANALYSIS,
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                    ],
                    interdependent_groups=[
                        "Performance Grading and Quality Gates",
                        "Multi-Modal Search Coordination",
                    ],
                    target_source="Workflow optimization and interdependency analysis",
                ),
                "ErrorHandlingCoordinatedConstants": ConstantClassification(
                    name="ErrorHandlingCoordinatedConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.CORRELATION_ANALYSIS,
                    ],
                    interdependent_groups=["Circuit Breaker and Fallback Coordination"],
                    target_source="Error pattern analysis and resilience optimization",
                ),
                "WorkflowConstants": ConstantClassification(
                    name="WorkflowConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.QUALITY_OPTIMIZATION,
                    ],
                    target_source="Workflow success pattern analysis",
                ),
            }
        )

        # === PERFORMANCE CONSTANTS (HIGH automation) ===
        classifications.update(
            {
                "PerformanceAdaptiveConstants": ConstantClassification(
                    name="PerformanceAdaptiveConstants",
                    automation_potential=AutomationPotential.VERY_HIGH,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.CORRELATION_ANALYSIS,
                        LearningMechanism.USAGE_PATTERNS,
                    ],
                    interdependent_groups=[
                        "Query Performance Optimization",
                        "Resource Management",
                        "Cache Performance Tuning",
                    ],
                    target_source="Real-time performance optimization",
                ),
                "CachePerformanceConstants": ConstantClassification(
                    name="CachePerformanceConstants",
                    automation_potential=AutomationPotential.HIGH,
                    learning_mechanisms=[
                        LearningMechanism.PERFORMANCE_FEEDBACK,
                        LearningMechanism.USAGE_PATTERNS,
                    ],
                    target_source="Cache hit rate optimization and access pattern analysis",
                ),
            }
        )

        # === VALIDATION CONSTANTS (MEDIUM automation) ===
        classifications.update(
            {
                "ValidationConstants": ConstantClassification(
                    name="ValidationConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[
                        LearningMechanism.QUALITY_OPTIMIZATION,
                        LearningMechanism.DOMAIN_ANALYSIS,
                    ],
                    target_source="Validation effectiveness analysis",
                ),
                "ErrorHandlingConstants": ConstantClassification(
                    name="ErrorHandlingConstants",
                    automation_potential=AutomationPotential.MEDIUM,
                    learning_mechanisms=[LearningMechanism.PERFORMANCE_FEEDBACK],
                    target_source="Error recovery pattern analysis",
                ),
            }
        )

        return classifications

    def get_classification(
        self, constant_name: str
    ) -> Optional[ConstantClassification]:
        """Get classification for a specific constant"""
        return self._classifications.get(constant_name)

    def get_by_automation_potential(
        self, potential: AutomationPotential
    ) -> List[ConstantClassification]:
        """Get all constants with specific automation potential"""
        return [
            classification
            for classification in self._classifications.values()
            if classification.automation_potential == potential
        ]

    def get_by_learning_mechanism(
        self, mechanism: LearningMechanism
    ) -> List[ConstantClassification]:
        """Get all constants that use a specific learning mechanism"""
        return [
            classification
            for classification in self._classifications.values()
            if mechanism in classification.learning_mechanisms
        ]

    def get_interdependent_groups(self) -> Dict[str, List[str]]:
        """Get all interdependent constant groups"""
        groups = {}
        for classification in self._classifications.values():
            if classification.interdependent_groups:
                for group in classification.interdependent_groups:
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(classification.name)
        return groups

    def get_automation_summary(self) -> Dict[AutomationPotential, int]:
        """Get count of constants by automation potential"""
        summary = {potential: 0 for potential in AutomationPotential}
        for classification in self._classifications.values():
            summary[classification.automation_potential] += 1
        return summary

    def get_learning_mechanism_summary(self) -> Dict[LearningMechanism, int]:
        """Get count of constants by learning mechanism"""
        summary = {mechanism: 0 for mechanism in LearningMechanism}
        for classification in self._classifications.values():
            for mechanism in classification.learning_mechanisms:
                summary[mechanism] += 1
        return summary


# Global classifier instance
automation_classifier = AutomationClassifier()


# Export all classes and instances
__all__ = [
    "AutomationPotential",
    "LearningMechanism",
    "ConstantClassification",
    "AutomationClassifier",
    "automation_classifier",
]
