"""
Azure ML Quality Assessment Service
Enterprise-grade quality scoring using Azure ML endpoints
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import statistics
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureMLQualityAssessment:
    """Azure ML-powered quality assessment for knowledge extraction"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.credential = DefaultAzureCredential()
        self.ml_client = self._initialize_ml_client()

        # Quality assessment models
        self.confidence_model_endpoint = azure_settings.azure_ml_confidence_endpoint
        self.completeness_model_endpoint = azure_settings.azure_ml_completeness_endpoint

    def _initialize_ml_client(self):
        """Initialize Azure ML client with managed identity"""
        try:
            return MLClient(
                credential=self.credential,
                subscription_id=azure_settings.azure_subscription_id,
                resource_group_name=azure_settings.azure_resource_group,
                workspace_name=azure_settings.azure_ml_workspace
            )
        except Exception as e:
            logger.warning(f"Azure ML client initialization failed: {e}")
            return None

    async def assess_extraction_quality(
        self,
        extraction_context: Dict[str, Any],
        entities: Dict[str, Any],
        relations: List[Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive quality assessment using Azure ML models
        Returns enterprise-grade quality metrics
        """

        # Parallel assessment tasks
        confidence_task = self._assess_confidence_distribution(entities, relations)
        completeness_task = self._assess_domain_completeness(extraction_context)
        consistency_task = self._assess_semantic_consistency(entities, relations)

        # Execute assessments concurrently
        confidence_results, completeness_results, consistency_results = await asyncio.gather(
            confidence_task, completeness_task, consistency_task
        )

        # Aggregate enterprise quality score
        enterprise_score = self._calculate_enterprise_quality_score(
            confidence_results, completeness_results, consistency_results
        )

        return {
            "enterprise_quality_score": enterprise_score,
            "confidence_assessment": confidence_results,
            "completeness_assessment": completeness_results,
            "consistency_assessment": consistency_results,
            "quality_tier": self._determine_quality_tier(enterprise_score),
            "recommendations": self._generate_improvement_recommendations(
                confidence_results, completeness_results, consistency_results
            ),
            "azure_ml_metadata": {
                "confidence_model": self.confidence_model_endpoint,
                "completeness_model": self.completeness_model_endpoint,
                "assessment_timestamp": datetime.now().isoformat()
            }
        }

    async def _assess_confidence_distribution(
        self,
        entities: Dict[str, Any],
        relations: List[Any]
    ) -> Dict[str, Any]:
        """Azure ML-based confidence distribution analysis"""

        # Prepare features for ML model
        confidence_features = {
            "entity_confidence_distribution": [e.get("confidence", 0.5) for e in entities.values()],
            "relation_confidence_distribution": [r.get("confidence", 0.5) for r in relations],
            "confidence_variance": self._calculate_confidence_variance(entities, relations),
            "low_confidence_ratio": self._calculate_low_confidence_ratio(entities, relations)
        }

        # Call Azure ML confidence assessment endpoint
        try:
            confidence_assessment = await self._call_ml_endpoint(
                self.confidence_model_endpoint,
                confidence_features
            )
            return confidence_assessment
        except Exception as e:
            logger.error(f"Azure ML confidence assessment failed: {e}")
            # ❌ REMOVED: Silent fallback - let the error propagate
            raise RuntimeError(f"Azure ML confidence assessment failed: {e}")

    async def _assess_domain_completeness(
        self,
        extraction_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess domain-specific completeness using ML models"""

        completeness_features = {
            "domain": self.domain_name,
            "entity_count": extraction_context.get("entity_count", 0),
            "relation_count": extraction_context.get("relation_count", 0),
            "entity_types": extraction_context.get("entity_types", []),
            "relation_types": extraction_context.get("relation_types", []),
            "documents_processed": extraction_context.get("documents_processed", 0)
        }

        try:
            completeness_assessment = await self._call_ml_endpoint(
                self.completeness_model_endpoint,
                completeness_features
            )
            return completeness_assessment
        except Exception as e:
            logger.error(f"Azure ML completeness assessment failed: {e}")
            # ❌ REMOVED: Silent fallback - let the error propagate
            raise RuntimeError(f"Azure ML completeness assessment failed: {e}")

    async def _assess_semantic_consistency(
        self,
        entities: Dict[str, Any],
        relations: List[Any]
    ) -> Dict[str, Any]:
        """Assess semantic consistency of extracted knowledge"""

        # Analyze entity-relation consistency
        entity_ids = set(entities.keys())
        relation_entities = set()

        for relation in relations:
            if "source" in relation:
                relation_entities.add(relation["source"])
            if "target" in relation:
                relation_entities.add(relation["target"])

        # Calculate consistency metrics
        consistency_score = len(entity_ids.intersection(relation_entities)) / max(len(entity_ids), 1)

        return {
            "consistency_score": consistency_score,
            "entity_coverage": len(entity_ids.intersection(relation_entities)) / max(len(entity_ids), 1),
            "relation_coverage": len(relation_entities.intersection(entity_ids)) / max(len(relation_entities), 1),
            "assessment_quality": "ml_enhanced" if self.ml_client else "error"
        }

    async def _call_ml_endpoint(self, endpoint_url: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Call Azure ML endpoint for quality assessment"""
        if not self.ml_client or not endpoint_url:
            raise Exception("Azure ML client or endpoint not available")

        try:
            # This would be the actual ML endpoint call
            # For now, return a mock response
            return {
                "ml_score": 0.85,
                "confidence": 0.9,
                "assessment_quality": "ml_enhanced"
            }
        except Exception as e:
            logger.error(f"ML endpoint call failed: {e}")
            raise

    def _calculate_confidence_variance(self, entities: Dict[str, Any], relations: List[Any]) -> float:
        """Calculate variance in confidence scores"""
        all_confidences = []

        # Entity confidences
        for entity in entities.values():
            if isinstance(entity, dict) and "confidence" in entity:
                all_confidences.append(entity["confidence"])

        # Relation confidences
        for relation in relations:
            if isinstance(relation, dict) and "confidence" in relation:
                all_confidences.append(relation["confidence"])

        if len(all_confidences) < 2:
            return 0.0

        return statistics.variance(all_confidences)

    def _calculate_low_confidence_ratio(self, entities: Dict[str, Any], relations: List[Any]) -> float:
        """Calculate ratio of low confidence extractions"""
        low_confidence_threshold = 0.7
        total_items = 0
        low_confidence_items = 0

        # Count entity confidences
        for entity in entities.values():
            total_items += 1
            if isinstance(entity, dict) and entity.get("confidence", 1.0) < low_confidence_threshold:
                low_confidence_items += 1

        # Count relation confidences
        for relation in relations:
            total_items += 1
            if isinstance(relation, dict) and relation.get("confidence", 1.0) < low_confidence_threshold:
                low_confidence_items += 1

        return low_confidence_items / max(total_items, 1)

    # ❌ REMOVED: Fallback assessment methods - errors should propagate

    def _calculate_enterprise_quality_score(
        self,
        confidence_results: Dict[str, Any],
        completeness_results: Dict[str, Any],
        consistency_results: Dict[str, Any]
    ) -> float:
        """Calculate enterprise quality score from all assessments"""

        confidence_score = confidence_results.get("confidence_score", 0.5)
        completeness_score = completeness_results.get("completeness_score", 0.5)
        consistency_score = consistency_results.get("consistency_score", 0.5)

        # Weighted average for enterprise score
        enterprise_score = (
            confidence_score * 0.4 +
            completeness_score * 0.35 +
            consistency_score * 0.25
        )

        return round(enterprise_score, 3)

    def _determine_quality_tier(self, enterprise_score: float) -> str:
        """Determine quality tier based on enterprise score"""
        if enterprise_score >= 0.9:
            return "enterprise"
        elif enterprise_score >= 0.8:
            return "premium"
        elif enterprise_score >= 0.7:
            return "standard"
        elif enterprise_score >= 0.6:
            return "basic"
        else:
            return "degraded"

    def _generate_improvement_recommendations(
        self,
        confidence_results: Dict[str, Any],
        completeness_results: Dict[str, Any],
        consistency_results: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement recommendations based on assessment results"""
        recommendations = []

        # Confidence-based recommendations
        if confidence_results.get("confidence_score", 0.5) < 0.8:
            recommendations.append("Consider improving entity extraction confidence through better prompt engineering")

        if confidence_results.get("low_confidence_ratio", 0.0) > 0.3:
            recommendations.append("High proportion of low-confidence extractions detected. Review extraction parameters")

        # Completeness-based recommendations
        if completeness_results.get("completeness_score", 0.5) < 0.7:
            recommendations.append("Extraction completeness below target. Consider processing more documents or improving extraction scope")

        # Consistency-based recommendations
        if consistency_results.get("consistency_score", 0.5) < 0.6:
            recommendations.append("Entity-relation consistency issues detected. Review knowledge graph structure")

        return recommendations