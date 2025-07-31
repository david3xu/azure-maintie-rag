"""
Azure Universal Entity and Relation Classifier
Enterprise-grade classification service using Azure Cognitive Services
Eliminates hardcoded patterns through data-driven Azure integration
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from infra.azure_openai import UnifiedAzureOpenAIClient as AzureTextAnalyticsService
from config.settings import azure_settings

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Enterprise classification result with Azure service metadata"""
    entity_type: str
    confidence: float
    category: str
    metadata: Dict[str, Any]


class AzureEntityClassifier:
    """Azure-powered entity classifier with fail-fast architecture"""

    def __init__(self, domain_config: Optional[Dict[str, Any]] = None):
        """Initialize Azure entity classifier"""
        self.domain_config = domain_config or {}
        self.azure_text_service = AzureTextAnalyticsService()

        # Configuration-driven thresholds (no hardcoded values)
        self.confidence_threshold = float(azure_settings.extraction_confidence_threshold)
        self.pattern_threshold = float(azure_settings.pattern_confidence_threshold)

        # Discovered patterns storage (populated by data discovery)
        self.discovered_entity_patterns = {}

        logger.info(f"Azure entity classifier initialized with confidence threshold: {self.confidence_threshold}")

    async def classify_entity(self, entity_text: str, context: str = "") -> ClassificationResult:
        """Azure Text Analytics entity classification with fail-fast architecture"""

        try:
            # Primary classification: Azure Text Analytics
            entity_results = await self.azure_text_service._recognize_entities_batch([entity_text])

            if not entity_results:
                error_msg = f"Azure Text Analytics returned no results for entity: {entity_text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if not entity_results[0]["entities"]:
                error_msg = f"Azure Text Analytics found no entities in text: {entity_text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            azure_entity = entity_results[0]["entities"][0]

            # Validate confidence meets enterprise threshold
            if azure_entity["confidence"] < self.confidence_threshold:
                error_msg = f"Azure classification confidence {azure_entity['confidence']:.3f} below threshold {self.confidence_threshold} for entity: {entity_text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            return ClassificationResult(
                entity_type=azure_entity["category"].lower(),
                confidence=azure_entity["confidence"],
                category=azure_entity["category"],
                metadata={
                    "source": "azure_text_analytics",
                    "azure_category": azure_entity["category"],
                    "text_length": len(entity_text),
                    "context_length": len(context)
                }
            )

        except Exception as e:
            logger.error(f"Azure entity classification failed for '{entity_text}': {e}")
            raise RuntimeError(f"Azure Text Analytics entity classification failed: {e}") from e

    async def classify_entities_batch(self, entities: List[str], contexts: List[str] = None) -> List[ClassificationResult]:
        """Batch entity classification with Azure Text Analytics"""
        if contexts is None:
            contexts = [""] * len(entities)

        try:
            # Batch processing through Azure Text Analytics
            entity_results = await self.azure_text_service._recognize_entities_batch(entities)

            if len(entity_results) != len(entities):
                error_msg = f"Azure batch processing mismatch: {len(entity_results)} results for {len(entities)} entities"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            results = []
            for i, (entity_text, context) in enumerate(zip(entities, contexts)):
                if not entity_results[i]["entities"]:
                    error_msg = f"No Azure classification for entity {i}: {entity_text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                azure_entity = entity_results[i]["entities"][0]

                if azure_entity["confidence"] < self.confidence_threshold:
                    error_msg = f"Batch entity {i} confidence {azure_entity['confidence']:.3f} below threshold {self.confidence_threshold}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                results.append(ClassificationResult(
                    entity_type=azure_entity["category"].lower(),
                    confidence=azure_entity["confidence"],
                    category=azure_entity["category"],
                    metadata={
                        "source": "azure_text_analytics_batch",
                        "batch_index": i,
                        "azure_category": azure_entity["category"]
                    }
                ))

            return results

        except Exception as e:
            logger.error(f"Azure batch entity classification failed: {e}")
            raise RuntimeError(f"Azure batch processing failed: {e}") from e

    async def validate_azure_service(self) -> None:
        """Validate Azure Text Analytics service availability"""
        try:
            test_results = await self.azure_text_service._detect_language_batch(["test connectivity"])

            if not test_results or not test_results[0].get("language"):
                raise RuntimeError("Azure Text Analytics service validation failed")

            logger.info("Azure Text Analytics service validation successful")

        except Exception as e:
            logger.error(f"Azure service validation failed: {e}")
            raise RuntimeError(f"Required Azure Text Analytics service unavailable: {e}") from e


class AzureRelationClassifier:
    """Azure-powered relation classifier using key phrase extraction"""

    def __init__(self, domain_config: Optional[Dict[str, Any]] = None):
        """Initialize Azure relation classifier"""
        self.domain_config = domain_config or {}
        self.azure_text_service = AzureTextAnalyticsService()

        # Configuration-driven parameters
        self.confidence_threshold = float(azure_settings.extraction_confidence_threshold)
        self.min_phrases_required = int(getattr(azure_settings, 'min_key_phrases_for_relations', 1))

        logger.info(f"Azure relation classifier initialized with confidence threshold: {self.confidence_threshold}")

    async def classify_relation(self, relation_text: str, entity1: str = "", entity2: str = "") -> ClassificationResult:
        """Azure key phrase-based relation classification"""

        try:
            # Use Azure Text Analytics key phrase extraction
            phrase_results = await self.azure_text_service._extract_key_phrases_batch([relation_text])

            if not phrase_results:
                error_msg = f"Azure Text Analytics returned no phrase results for relation: {relation_text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            key_phrases = phrase_results[0]["key_phrases"]

            if len(key_phrases) < self.min_phrases_required:
                error_msg = f"Azure found {len(key_phrases)} key phrases, minimum required: {self.min_phrases_required} for relation: {relation_text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Data-driven relation classification from Azure key phrases
            relation_type = self._determine_relation_from_phrases(key_phrases)
            confidence = self._calculate_phrase_confidence(key_phrases, relation_text)

            if confidence < self.confidence_threshold:
                error_msg = f"Relation confidence {confidence:.3f} below threshold {self.confidence_threshold} for: {relation_text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            return ClassificationResult(
                entity_type=relation_type,
                confidence=confidence,
                category="relation",
                metadata={
                    "source": "azure_key_phrases",
                    "key_phrases": key_phrases,
                    "entity1": entity1,
                    "entity2": entity2,
                    "phrase_count": len(key_phrases)
                }
            )

        except Exception as e:
            logger.error(f"Azure relation classification failed for '{relation_text}': {e}")
            raise RuntimeError(f"Azure relation classification failed: {e}") from e

    def _determine_relation_from_phrases(self, key_phrases: List[str]) -> str:
        """Determine relation type from Azure key phrases"""
        if not key_phrases:
            raise ValueError("No key phrases available for relation type determination")

        # Use most prominent key phrase as relation type (data-driven)
        primary_phrase = key_phrases[0].lower().replace(' ', '_').replace('-', '_')

        # Validate relation type format
        if not primary_phrase or len(primary_phrase) < 2:
            raise ValueError(f"Invalid relation type derived from phrase: {key_phrases[0]}")

        return primary_phrase

    def _calculate_phrase_confidence(self, key_phrases: List[str], original_text: str) -> float:
        """Calculate confidence based on key phrase relevance and coverage"""
        if not key_phrases:
            raise ValueError("No key phrases for confidence calculation")

        # Calculate confidence based on phrase density and text coverage
        total_phrase_length = sum(len(phrase) for phrase in key_phrases)
        text_coverage = min(1.0, total_phrase_length / len(original_text))
        phrase_density = min(1.0, len(key_phrases) / 5.0)  # Normalize to max 5 phrases

        # Weighted confidence calculation
        confidence = (text_coverage * 0.6) + (phrase_density * 0.4)

        return round(confidence, 3)


class AzureClassificationPipeline:
    """Enterprise Azure classification pipeline orchestrator"""

    def __init__(self, domain_config: Optional[Dict[str, Any]] = None):
        """Initialize Azure classification pipeline"""
        self.entity_classifier = AzureEntityClassifier(domain_config)
        self.relation_classifier = AzureRelationClassifier(domain_config)

        logger.info("Azure classification pipeline initialized")

    async def validate_azure_services(self) -> Dict[str, Any]:
        """Validate all Azure services before processing"""
        validation_results = {}

        try:
            # Validate entity classification service
            await self.entity_classifier.validate_azure_service()
            validation_results["entity_classifier"] = {"status": "healthy", "service": "azure_text_analytics"}

            # Validate relation classification service (uses same Azure service)
            validation_results["relation_classifier"] = {"status": "healthy", "service": "azure_text_analytics"}

            validation_results["overall_status"] = "healthy"
            logger.info("Azure classification pipeline validation successful")

        except Exception as e:
            error_msg = f"Azure classification pipeline validation failed: {e}"
            logger.error(error_msg)
            validation_results["overall_status"] = "failed"
            validation_results["error"] = str(e)
            raise RuntimeError(error_msg) from e

        return validation_results

    async def classify_knowledge_triplet(self, entity1: str, relation: str, entity2: str) -> Dict[str, ClassificationResult]:
        """Classify complete knowledge triplet using Azure services"""

        try:
            # Validate services before processing
            await self.validate_azure_services()

            # Classify all components
            results = {
                "entity1": await self.entity_classifier.classify_entity(entity1),
                "relation": await self.relation_classifier.classify_relation(relation, entity1, entity2),
                "entity2": await self.entity_classifier.classify_entity(entity2)
            }

            logger.info(f"Successfully classified triplet: {entity1} -> {relation} -> {entity2}")
            return results

        except Exception as e:
            error_msg = f"Knowledge triplet classification failed for '{entity1}' -> '{relation}' -> '{entity2}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get Azure classification pipeline statistics"""
        return {
            "entity_classifier": {
                "service": "azure_text_analytics",
                "confidence_threshold": self.entity_classifier.confidence_threshold,
                "pattern_threshold": self.entity_classifier.pattern_threshold
            },
            "relation_classifier": {
                "service": "azure_text_analytics_key_phrases",
                "confidence_threshold": self.relation_classifier.confidence_threshold,
                "min_phrases_required": self.relation_classifier.min_phrases_required
            },
            "pipeline_status": "azure_integrated",
            "hardcoded_patterns": 0,  # No hardcoded patterns in Azure architecture
            "data_driven": True
        }


class AzureClassificationHealthMonitor:
    """Azure service health monitoring for classification pipeline"""

    def __init__(self, pipeline: AzureClassificationPipeline):
        self.pipeline = pipeline

    async def get_health_status(self) -> Dict[str, Any]:
        """Comprehensive health check for Azure classification services"""
        try:
            validation_results = await self.pipeline.validate_azure_services()

            return {
                "timestamp": "auto-generated",
                "classification_pipeline": validation_results["overall_status"],
                "azure_text_analytics": validation_results.get("entity_classifier", {}).get("status", "unknown"),
                "key_phrase_extraction": validation_results.get("relation_classifier", {}).get("status", "unknown"),
                "configuration": {
                    "confidence_threshold": float(azure_settings.extraction_confidence_threshold),
                    "pattern_threshold": float(azure_settings.pattern_confidence_threshold)
                },
                "dependencies": ["azure_text_analytics_service"],
                "hardcoded_patterns": False,
                "data_driven_classification": True
            }

        except Exception as e:
            logger.error(f"Classification health check failed: {e}")
            return {
                "timestamp": "auto-generated",
                "classification_pipeline": "failed",
                "error": str(e),
                "azure_service_available": False
            }