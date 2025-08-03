"""
Validation Processor - Extraction Quality Validation and Optimization

This module provides comprehensive validation of extraction results to ensure
quality, consistency, and compliance with extraction configuration requirements.

Key Features:
- Multi-level validation (entity, relationship, graph, semantic)
- Quality scoring and confidence assessment
- Anomaly detection and consistency checking
- Performance validation against configuration targets
- Feedback generation for configuration optimization

Architecture Integration:
- Used by Knowledge Extraction Agent for result validation
- Integrates with extraction configuration quality requirements
- Provides structured feedback for Config-Extraction workflow improvement
- Supports validation criteria and quality thresholds
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

from pydantic import BaseModel, Field

# Interface contracts
from config.extraction_interface import ExtractionConfiguration, ConfigurationFeedback

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: str  # "error", "warning", "info"
    category: str  # "entity", "relationship", "graph", "performance", "consistency"
    message: str
    affected_items: List[str] = None
    confidence_impact: float = 0.0
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.affected_items is None:
            self.affected_items = []
        if self.suggestions is None:
            self.suggestions = []


class ValidationReport(BaseModel):
    """Comprehensive validation report"""
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    validation_passed: bool = Field(..., description="Whether validation passed")
    processing_time: float = Field(..., description="Validation processing time")
    
    # Detailed scores
    entity_quality_score: float = Field(..., description="Entity extraction quality")
    relationship_quality_score: float = Field(..., description="Relationship extraction quality")
    graph_quality_score: float = Field(..., description="Knowledge graph quality")
    consistency_score: float = Field(..., description="Internal consistency score")
    performance_score: float = Field(..., description="Performance metrics score")
    
    # Issue analysis
    total_issues: int = Field(..., description="Total issues found")
    error_count: int = Field(..., description="Number of errors")
    warning_count: int = Field(..., description="Number of warnings")
    info_count: int = Field(..., description="Number of info items")
    
    # Detailed issues
    issues: List[Dict[str, Any]] = Field(..., description="Detailed validation issues")
    
    # Quality metrics
    confidence_metrics: Dict[str, float] = Field(..., description="Confidence analysis")
    coverage_metrics: Dict[str, float] = Field(..., description="Coverage analysis")
    performance_metrics: Dict[str, float] = Field(..., description="Performance analysis")
    
    # Recommendations
    configuration_feedback: Dict[str, Any] = Field(..., description="Configuration optimization feedback")
    improvement_suggestions: List[str] = Field(..., description="Improvement suggestions")


class ValidationProcessor:
    """
    Comprehensive validation processor for extraction results with
    quality assessment and optimization feedback.
    """
    
    def __init__(self):
        self._validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "average_processing_time": 0.0,
            "average_quality_score": 0.0,
            "issue_frequency": {
                "entity": {"error": 0, "warning": 0, "info": 0},
                "relationship": {"error": 0, "warning": 0, "info": 0},
                "graph": {"error": 0, "warning": 0, "info": 0},
                "performance": {"error": 0, "warning": 0, "info": 0},
                "consistency": {"error": 0, "warning": 0, "info": 0}
            }
        }
    
    async def validate_extraction_results(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration,
        performance_metrics: Dict[str, Any] = None
    ) -> ValidationReport:
        """
        Validate complete extraction results against configuration requirements.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            config: Extraction configuration with validation criteria
            performance_metrics: Performance metrics from extraction
            
        Returns:
            ValidationReport: Comprehensive validation results and recommendations
        """
        start_time = time.time()
        
        try:
            issues = []
            
            # Validate entities
            entity_issues, entity_score = await self._validate_entities(entities, config)
            issues.extend(entity_issues)
            
            # Validate relationships
            relationship_issues, relationship_score = await self._validate_relationships(
                relationships, entities, config
            )
            issues.extend(relationship_issues)
            
            # Validate knowledge graph structure
            graph_issues, graph_score = await self._validate_knowledge_graph(
                entities, relationships, config
            )
            issues.extend(graph_issues)
            
            # Validate consistency
            consistency_issues, consistency_score = await self._validate_consistency(
                entities, relationships, config
            )
            issues.extend(consistency_issues)
            
            # Validate performance
            performance_issues, performance_score = await self._validate_performance(
                performance_metrics or {}, config
            )
            issues.extend(performance_issues)
            
            # Calculate overall metrics
            overall_score = self._calculate_overall_score(
                entity_score, relationship_score, graph_score, consistency_score, performance_score
            )
            
            validation_passed = self._determine_validation_status(overall_score, issues, config)
            
            # Generate feedback and recommendations
            configuration_feedback = self._generate_configuration_feedback(
                entities, relationships, issues, config
            )
            improvement_suggestions = self._generate_improvement_suggestions(issues, config)
            
            # Calculate detailed metrics
            confidence_metrics = self._calculate_confidence_metrics(entities, relationships)
            coverage_metrics = self._calculate_coverage_metrics(entities, relationships, config)
            performance_metrics_calc = self._calculate_performance_metrics(performance_metrics or {})
            
            processing_time = time.time() - start_time
            
            # Count issues by severity
            error_count = len([i for i in issues if i.severity == "error"])
            warning_count = len([i for i in issues if i.severity == "warning"])
            info_count = len([i for i in issues if i.severity == "info"])
            
            # Create validation report
            report = ValidationReport(
                overall_score=overall_score,
                validation_passed=validation_passed,
                processing_time=processing_time,
                entity_quality_score=entity_score,
                relationship_quality_score=relationship_score,
                graph_quality_score=graph_score,
                consistency_score=consistency_score,
                performance_score=performance_score,
                total_issues=len(issues),
                error_count=error_count,
                warning_count=warning_count,
                info_count=info_count,
                issues=[self._issue_to_dict(issue) for issue in issues],
                confidence_metrics=confidence_metrics,
                coverage_metrics=coverage_metrics,
                performance_metrics=performance_metrics_calc,
                configuration_feedback=configuration_feedback,
                improvement_suggestions=improvement_suggestions
            )
            
            # Update statistics
            self._update_validation_stats(processing_time, overall_score, validation_passed, issues)
            
            return report
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Validation failed: {e}")
            
            # Return minimal error report
            return ValidationReport(
                overall_score=0.0,
                validation_passed=False,
                processing_time=processing_time,
                entity_quality_score=0.0,
                relationship_quality_score=0.0,
                graph_quality_score=0.0,
                consistency_score=0.0,
                performance_score=0.0,
                total_issues=1,
                error_count=1,
                warning_count=0,
                info_count=0,
                issues=[{
                    "severity": "error",
                    "category": "system",
                    "message": f"Validation failed: {str(e)}",
                    "affected_items": [],
                    "confidence_impact": 1.0,
                    "suggestions": ["Check system configuration and retry"]
                }],
                confidence_metrics={},
                coverage_metrics={},
                performance_metrics={},
                configuration_feedback={},
                improvement_suggestions=["Fix validation system errors"]
            )
    
    async def _validate_entities(
        self,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration
    ) -> Tuple[List[ValidationIssue], float]:
        """Validate entity extraction results"""
        issues = []
        
        # Check minimum entity count
        min_entities = config.validation_criteria.get("min_entities_per_document", 0)
        if len(entities) < min_entities:
            issues.append(ValidationIssue(
                severity="error",
                category="entity",
                message=f"Insufficient entities: {len(entities)} found, {min_entities} required",
                affected_items=[],
                confidence_impact=0.3,
                suggestions=[
                    "Lower entity confidence threshold",
                    "Add more entity types to expected list",
                    "Improve entity extraction patterns"
                ]
            ))
        
        # Check entity quality
        if entities:
            # Confidence distribution analysis
            confidences = [e.get("confidence", 0.0) for e in entities]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < config.entity_confidence_threshold:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="entity",
                    message=f"Low average entity confidence: {avg_confidence:.2f}",
                    confidence_impact=0.2,
                    suggestions=["Review entity extraction patterns", "Adjust confidence calculation"]
                ))
            
            # Check for duplicate entities
            entity_texts = [e.get("name", "") for e in entities]
            duplicates = [text for text, count in Counter(entity_texts).items() if count > 1]
            
            if duplicates:
                issues.append(ValidationIssue(
                    severity="info",
                    category="entity",
                    message=f"Duplicate entities found: {len(duplicates)} types",
                    affected_items=duplicates[:5],
                    suggestions=["Improve entity deduplication", "Check entity normalization"]
                ))
            
            # Check entity type coverage
            expected_types = set(config.expected_entity_types)
            found_types = set(e.get("type", "") for e in entities)
            missing_types = expected_types - found_types
            
            if missing_types and len(missing_types) > len(expected_types) / 2:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="entity",
                    message=f"Many expected entity types missing: {len(missing_types)} of {len(expected_types)}",
                    affected_items=list(missing_types)[:5],
                    suggestions=[
                        "Review entity type patterns",
                        "Add more entity extraction methods",
                        "Check domain-specific vocabulary"
                    ]
                ))
        
        # Calculate entity quality score
        entity_score = self._calculate_entity_quality_score(entities, config, issues)
        
        return issues, entity_score
    
    async def _validate_relationships(
        self,
        relationships: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration
    ) -> Tuple[List[ValidationIssue], float]:
        """Validate relationship extraction results"""
        issues = []
        
        # Check minimum relationship count
        min_relationships = config.validation_criteria.get("min_relationships_per_document", 0)
        if len(relationships) < min_relationships:
            issues.append(ValidationIssue(
                severity="error",
                category="relationship",
                message=f"Insufficient relationships: {len(relationships)} found, {min_relationships} required",
                confidence_impact=0.3,
                suggestions=[
                    "Lower relationship confidence threshold",
                    "Add more relationship patterns",
                    "Improve relationship extraction methods"
                ]
            ))
        
        if relationships:
            # Confidence analysis
            confidences = [r.get("confidence", 0.0) for r in relationships]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < config.relationship_confidence_threshold:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="relationship",
                    message=f"Low average relationship confidence: {avg_confidence:.2f}",
                    confidence_impact=0.2,
                    suggestions=["Review relationship extraction patterns", "Improve confidence calculation"]
                ))
            
            # Check relationship validity (entities must exist)
            entity_names = set(e.get("name", "") for e in entities)
            invalid_relationships = []
            
            for rel in relationships:
                source = rel.get("source", "")
                target = rel.get("target", "")
                
                if source not in entity_names or target not in entity_names:
                    invalid_relationships.append(f"{source} -> {target}")
            
            if invalid_relationships:
                issues.append(ValidationIssue(
                    severity="error",
                    category="relationship",
                    message=f"Relationships with non-existent entities: {len(invalid_relationships)}",
                    affected_items=invalid_relationships[:5],
                    confidence_impact=0.4,
                    suggestions=[
                        "Improve entity-relationship coordination",
                        "Add entity validation before relationship extraction"
                    ]
                ))
            
            # Check relationship diversity
            relation_types = [r.get("relation", "") for r in relationships]
            unique_types = set(relation_types)
            
            if len(unique_types) < 2 and len(relationships) > 5:
                issues.append(ValidationIssue(
                    severity="info",
                    category="relationship",
                    message="Low relationship type diversity",
                    suggestions=["Add more relationship types", "Improve relationship classification"]
                ))
        
        # Calculate relationship quality score
        relationship_score = self._calculate_relationship_quality_score(relationships, config, issues)
        
        return issues, relationship_score
    
    async def _validate_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration
    ) -> Tuple[List[ValidationIssue], float]:
        """Validate knowledge graph structure and quality"""
        issues = []
        
        if not entities or not relationships:
            issues.append(ValidationIssue(
                severity="warning",
                category="graph",
                message="Incomplete knowledge graph: missing entities or relationships",
                confidence_impact=0.2,
                suggestions=["Ensure both entity and relationship extraction are working"]
            ))
            return issues, 0.3
        
        # Build graph structure
        entity_names = set(e.get("name", "") for e in entities)
        graph_edges = []
        connected_entities = set()
        
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            
            if source in entity_names and target in entity_names:
                graph_edges.append((source, target))
                connected_entities.add(source)
                connected_entities.add(target)
        
        # Check connectivity
        unconnected_entities = entity_names - connected_entities
        connectivity_ratio = len(connected_entities) / len(entity_names) if entity_names else 0
        
        if connectivity_ratio < 0.5:
            issues.append(ValidationIssue(
                severity="warning",
                category="graph",
                message=f"Low graph connectivity: {connectivity_ratio:.1%} of entities connected",
                affected_items=list(unconnected_entities)[:5],
                suggestions=[
                    "Improve relationship extraction coverage",
                    "Check for missing relationship patterns"
                ]
            ))
        
        # Check graph density
        max_edges = len(entity_names) * (len(entity_names) - 1) / 2
        actual_edges = len(graph_edges)
        density = actual_edges / max_edges if max_edges > 0 else 0
        
        if density > 0.8:
            issues.append(ValidationIssue(
                severity="info",
                category="graph",
                message=f"Very high graph density: {density:.1%}",
                suggestions=["Check for over-extraction of relationships"]
            ))
        elif density < 0.1 and len(entity_names) > 5:
            issues.append(ValidationIssue(
                severity="warning",
                category="graph",
                message=f"Very low graph density: {density:.1%}",
                suggestions=["Improve relationship extraction", "Check relationship patterns"]
            ))
        
        # Calculate graph quality score
        graph_score = self._calculate_graph_quality_score(entities, relationships, issues)
        
        return issues, graph_score
    
    async def _validate_consistency(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration
    ) -> Tuple[List[ValidationIssue], float]:
        """Validate internal consistency of extraction results"""
        issues = []
        
        # Check entity-relationship consistency
        entity_names = set(e.get("name", "") for e in entities)
        
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            
            # Check if relationship entities exist
            if source and source not in entity_names:
                issues.append(ValidationIssue(
                    severity="error",
                    category="consistency",
                    message=f"Relationship source '{source}' not found in entities",
                    confidence_impact=0.2
                ))
            
            if target and target not in entity_names:
                issues.append(ValidationIssue(
                    severity="error",
                    category="consistency",
                    message=f"Relationship target '{target}' not found in entities",
                    confidence_impact=0.2
                ))
        
        # Check confidence consistency
        if entities:
            entity_confidences = [e.get("confidence", 0.0) for e in entities if e.get("confidence") is not None]
            relationship_confidences = [r.get("confidence", 0.0) for r in relationships if r.get("confidence") is not None]
            
            if entity_confidences and relationship_confidences:
                entity_avg = sum(entity_confidences) / len(entity_confidences)
                relationship_avg = sum(relationship_confidences) / len(relationship_confidences)
                
                # Check for significant confidence imbalance
                if abs(entity_avg - relationship_avg) > 0.3:
                    issues.append(ValidationIssue(
                        severity="info",
                        category="consistency",
                        message=f"Confidence imbalance: entities {entity_avg:.2f}, relationships {relationship_avg:.2f}",
                        suggestions=["Review confidence calculation methods"]
                    ))
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(entities, relationships, issues)
        
        return issues, consistency_score
    
    async def _validate_performance(
        self,
        performance_metrics: Dict[str, Any],
        config: ExtractionConfiguration
    ) -> Tuple[List[ValidationIssue], float]:
        """Validate performance against configuration targets"""
        issues = []
        
        # Check processing time
        processing_time = performance_metrics.get("processing_time", 0.0)
        max_time = config.validation_criteria.get("max_processing_time_seconds", 30.0)
        
        if processing_time > max_time:
            issues.append(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Processing time {processing_time:.1f}s exceeds limit {max_time}s",
                suggestions=["Optimize extraction algorithms", "Increase parallel processing"]
            ))
        
        # Check target response time
        if hasattr(config, 'target_response_time_seconds'):
            target_time = config.target_response_time_seconds
            if processing_time > target_time:
                issues.append(ValidationIssue(
                    severity="info",
                    category="performance",
                    message=f"Processing time {processing_time:.1f}s exceeds target {target_time}s",
                    suggestions=["Enable caching", "Optimize processing pipeline"]
                ))
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(performance_metrics, config, issues)
        
        return issues, performance_score
    
    def _calculate_overall_score(
        self,
        entity_score: float,
        relationship_score: float,
        graph_score: float,
        consistency_score: float,
        performance_score: float
    ) -> float:
        """Calculate weighted overall quality score"""
        
        weights = {
            "entity": 0.25,
            "relationship": 0.25,
            "graph": 0.20,
            "consistency": 0.20,
            "performance": 0.10
        }
        
        overall_score = (
            entity_score * weights["entity"] +
            relationship_score * weights["relationship"] +
            graph_score * weights["graph"] +
            consistency_score * weights["consistency"] +
            performance_score * weights["performance"]
        )
        
        return min(1.0, overall_score)
    
    def _determine_validation_status(
        self,
        overall_score: float,
        issues: List[ValidationIssue],
        config: ExtractionConfiguration
    ) -> bool:
        """Determine if validation passes based on score and issues"""
        
        # Check for critical errors
        error_count = len([i for i in issues if i.severity == "error"])
        if error_count > 0:
            return False
        
        # Check overall score against minimum quality
        min_quality = config.minimum_quality_score
        if overall_score < min_quality:
            return False
        
        # Check confidence impact
        total_confidence_impact = sum(i.confidence_impact for i in issues)
        if total_confidence_impact > 0.5:
            return False
        
        return True
    
    def _calculate_entity_quality_score(
        self,
        entities: List[Dict[str, Any]],
        config: ExtractionConfiguration,
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate entity-specific quality score"""
        
        if not entities:
            return 0.0
        
        base_score = 0.8
        
        # Confidence factor
        confidences = [e.get("confidence", 0.0) for e in entities]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_factor = avg_confidence
        
        # Coverage factor
        expected_types = set(config.expected_entity_types)
        found_types = set(e.get("type", "") for e in entities)
        coverage_factor = len(found_types & expected_types) / len(expected_types) if expected_types else 1.0
        
        # Issue penalty
        entity_issues = [i for i in issues if i.category == "entity" and i.severity in ["error", "warning"]]
        issue_penalty = min(0.3, len(entity_issues) * 0.1)
        
        score = base_score * 0.4 + confidence_factor * 0.3 + coverage_factor * 0.3 - issue_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_relationship_quality_score(
        self,
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration,
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate relationship-specific quality score"""
        
        if not relationships:
            return 0.0
        
        base_score = 0.8
        
        # Confidence factor
        confidences = [r.get("confidence", 0.0) for r in relationships]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_factor = avg_confidence
        
        # Diversity factor
        relation_types = set(r.get("relation", "") for r in relationships)
        diversity_factor = min(1.0, len(relation_types) / 5)  # Normalize to max 5 types
        
        # Issue penalty
        rel_issues = [i for i in issues if i.category == "relationship" and i.severity in ["error", "warning"]]
        issue_penalty = min(0.3, len(rel_issues) * 0.1)
        
        score = base_score * 0.4 + confidence_factor * 0.3 + diversity_factor * 0.3 - issue_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_graph_quality_score(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate graph structure quality score"""
        
        if not entities or not relationships:
            return 0.3
        
        # Connectivity factor
        entity_names = set(e.get("name", "") for e in entities)
        connected_entities = set()
        
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source in entity_names and target in entity_names:
                connected_entities.add(source)
                connected_entities.add(target)
        
        connectivity_factor = len(connected_entities) / len(entity_names) if entity_names else 0
        
        # Density factor (aim for moderate density)
        max_edges = len(entity_names) * (len(entity_names) - 1) / 2
        actual_edges = len(relationships)
        density = actual_edges / max_edges if max_edges > 0 else 0
        
        # Optimal density is around 0.2-0.6
        if 0.2 <= density <= 0.6:
            density_factor = 1.0
        elif density < 0.2:
            density_factor = density / 0.2
        else:
            density_factor = max(0.3, 1.0 - (density - 0.6) / 0.4)
        
        # Issue penalty
        graph_issues = [i for i in issues if i.category == "graph" and i.severity in ["error", "warning"]]
        issue_penalty = min(0.3, len(graph_issues) * 0.15)
        
        score = connectivity_factor * 0.5 + density_factor * 0.5 - issue_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_consistency_score(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate internal consistency score"""
        
        base_score = 1.0
        
        # Issue penalty
        consistency_issues = [i for i in issues if i.category == "consistency"]
        error_penalty = len([i for i in consistency_issues if i.severity == "error"]) * 0.3
        warning_penalty = len([i for i in consistency_issues if i.severity == "warning"]) * 0.1
        
        score = base_score - error_penalty - warning_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_performance_score(
        self,
        performance_metrics: Dict[str, Any],
        config: ExtractionConfiguration,
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate performance score"""
        
        base_score = 0.8
        
        # Time factor
        processing_time = performance_metrics.get("processing_time", 0.0)
        max_time = config.validation_criteria.get("max_processing_time_seconds", 30.0)
        
        if processing_time <= max_time:
            time_factor = 1.0
        else:
            time_factor = max(0.2, max_time / processing_time)
        
        # Issue penalty
        perf_issues = [i for i in issues if i.category == "performance" and i.severity in ["error", "warning"]]
        issue_penalty = min(0.3, len(perf_issues) * 0.15)
        
        score = base_score * 0.3 + time_factor * 0.7 - issue_penalty
        
        return max(0.0, min(1.0, score))
    
    def _generate_configuration_feedback(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        issues: List[ValidationIssue],
        config: ExtractionConfiguration
    ) -> Dict[str, Any]:
        """Generate feedback for configuration optimization"""
        
        feedback = {
            "entity_threshold_adjustment": None,
            "relationship_threshold_adjustment": None,
            "pattern_additions": [],
            "performance_optimizations": [],
            "quality_improvements": []
        }
        
        # Analyze entity confidence distribution
        if entities:
            confidences = [e.get("confidence", 0.0) for e in entities]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < config.entity_confidence_threshold + 0.1:
                feedback["entity_threshold_adjustment"] = {
                    "current": config.entity_confidence_threshold,
                    "suggested": max(0.1, config.entity_confidence_threshold - 0.1),
                    "reason": "Low entity confidence suggests threshold may be too high"
                }
        
        # Analyze relationship confidence
        if relationships:
            confidences = [r.get("confidence", 0.0) for r in relationships]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < config.relationship_confidence_threshold + 0.1:
                feedback["relationship_threshold_adjustment"] = {
                    "current": config.relationship_confidence_threshold,
                    "suggested": max(0.1, config.relationship_confidence_threshold - 0.1),
                    "reason": "Low relationship confidence suggests threshold may be too high"
                }
        
        # Generate pattern suggestions based on issues
        for issue in issues:
            if "missing" in issue.message.lower() and issue.category == "entity":
                feedback["pattern_additions"].append({
                    "type": "entity_pattern",
                    "suggestion": f"Add patterns for: {', '.join(issue.affected_items[:3])}"
                })
            elif "missing" in issue.message.lower() and issue.category == "relationship":
                feedback["pattern_additions"].append({
                    "type": "relationship_pattern", 
                    "suggestion": f"Add relationship patterns for better coverage"
                })
        
        return feedback
    
    def _generate_improvement_suggestions(
        self,
        issues: List[ValidationIssue],
        config: ExtractionConfiguration
    ) -> List[str]:
        """Generate actionable improvement suggestions"""
        
        suggestions = []
        
        # Prioritize by severity and frequency
        error_issues = [i for i in issues if i.severity == "error"]
        warning_issues = [i for i in issues if i.severity == "warning"]
        
        # Add error-based suggestions first
        for issue in error_issues[:3]:  # Top 3 errors
            suggestions.extend(issue.suggestions[:2])  # Top 2 suggestions per issue
        
        # Add warning-based suggestions
        for issue in warning_issues[:2]:  # Top 2 warnings
            suggestions.extend(issue.suggestions[:1])  # Top suggestion per issue
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]  # Limit to 5 total suggestions
    
    def _calculate_confidence_metrics(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate detailed confidence metrics"""
        
        metrics = {}
        
        if entities:
            entity_confidences = [e.get("confidence", 0.0) for e in entities]
            metrics["entity_avg_confidence"] = sum(entity_confidences) / len(entity_confidences)
            metrics["entity_min_confidence"] = min(entity_confidences)
            metrics["entity_max_confidence"] = max(entity_confidences)
        else:
            metrics.update({
                "entity_avg_confidence": 0.0,
                "entity_min_confidence": 0.0,
                "entity_max_confidence": 0.0
            })
        
        if relationships:
            rel_confidences = [r.get("confidence", 0.0) for r in relationships]
            metrics["relationship_avg_confidence"] = sum(rel_confidences) / len(rel_confidences)
            metrics["relationship_min_confidence"] = min(rel_confidences)
            metrics["relationship_max_confidence"] = max(rel_confidences)
        else:
            metrics.update({
                "relationship_avg_confidence": 0.0,
                "relationship_min_confidence": 0.0,
                "relationship_max_confidence": 0.0
            })
        
        return metrics
    
    def _calculate_coverage_metrics(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        config: ExtractionConfiguration
    ) -> Dict[str, float]:
        """Calculate coverage metrics"""
        
        metrics = {}
        
        # Entity type coverage
        expected_types = set(config.expected_entity_types)
        found_types = set(e.get("type", "") for e in entities)
        
        if expected_types:
            metrics["entity_type_coverage"] = len(found_types & expected_types) / len(expected_types)
        else:
            metrics["entity_type_coverage"] = 1.0
        
        # Relationship pattern coverage
        if hasattr(config, 'relationship_patterns') and config.relationship_patterns:
            pattern_relations = [p.split()[1] for p in config.relationship_patterns if len(p.split()) >= 3]
            found_relations = set(r.get("relation", "") for r in relationships)
            
            if pattern_relations:
                metrics["relationship_pattern_coverage"] = len(set(pattern_relations) & found_relations) / len(pattern_relations)
            else:
                metrics["relationship_pattern_coverage"] = 1.0
        else:
            metrics["relationship_pattern_coverage"] = 1.0
        
        return metrics
    
    def _calculate_performance_metrics(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance-related metrics"""
        
        return {
            "processing_time": performance_metrics.get("processing_time", 0.0),
            "entities_per_second": performance_metrics.get("entities_per_second", 0.0),
            "relationships_per_second": performance_metrics.get("relationships_per_second", 0.0),
            "memory_usage_mb": performance_metrics.get("memory_usage_mb", 0.0)
        }
    
    def _issue_to_dict(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Convert ValidationIssue to dictionary"""
        
        return {
            "severity": issue.severity,
            "category": issue.category,
            "message": issue.message,
            "affected_items": issue.affected_items,
            "confidence_impact": issue.confidence_impact,
            "suggestions": issue.suggestions
        }
    
    def _update_validation_stats(
        self,
        processing_time: float,
        quality_score: float,
        passed: bool,
        issues: List[ValidationIssue]
    ):
        """Update validation statistics"""
        
        self._validation_stats["total_validations"] += 1
        if passed:
            self._validation_stats["passed_validations"] += 1
        
        # Update average processing time
        total = self._validation_stats["total_validations"]
        current_avg_time = self._validation_stats["average_processing_time"]
        self._validation_stats["average_processing_time"] = (
            (current_avg_time * (total - 1) + processing_time) / total
        )
        
        # Update average quality score
        current_avg_quality = self._validation_stats["average_quality_score"]
        self._validation_stats["average_quality_score"] = (
            (current_avg_quality * (total - 1) + quality_score) / total
        )
        
        # Update issue frequency
        for issue in issues:
            if issue.category in self._validation_stats["issue_frequency"]:
                self._validation_stats["issue_frequency"][issue.category][issue.severity] += 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        total = self._validation_stats["total_validations"]
        return {
            **self._validation_stats,
            "success_rate": (
                self._validation_stats["passed_validations"] / total
                if total > 0 else 0.0
            )
        }


# Export main components
__all__ = [
    "ValidationProcessor",
    "ValidationIssue",
    "ValidationReport"
]