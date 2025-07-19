"""
Azure Text Analytics Service for Enterprise Knowledge Extraction
Pre-processing service to enhance extraction accuracy
"""

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from ...config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureTextAnalyticsService:
    """Enterprise text pre-processing with Azure Cognitive Services"""

    def __init__(self):
        self.client = TextAnalyticsClient(
            endpoint=azure_settings.azure_text_analytics_endpoint,
            credential=AzureKeyCredential(azure_settings.azure_text_analytics_key)
        )
        self.supported_languages = ["en", "es", "fr", "de"]

    async def preprocess_for_extraction(self, texts: List[str]) -> Dict[str, Any]:
        """
        Pre-process texts using Azure Text Analytics for enhanced extraction
        Returns enhanced context for knowledge extraction
        """
        # Detect language and entities
        language_results = await self._detect_language_batch(texts)
        entity_results = await self._recognize_entities_batch(texts)
        key_phrase_results = await self._extract_key_phrases_batch(texts)

        return {
            "enhanced_texts": texts,
            "language_metadata": language_results,
            "pre_identified_entities": entity_results,
            "key_phrases": key_phrase_results,
            "processing_confidence": self._calculate_preprocessing_confidence(
                language_results, entity_results
            )
        }

    async def _detect_language_batch(self, texts: List[str]) -> List[Dict]:
        """Batch language detection with confidence scoring"""
        try:
            response = self.client.detect_language(documents=texts)
            return [
                {
                    "language": doc.primary_language.iso6391_name,
                    "confidence": doc.primary_language.confidence_score,
                    "supported": doc.primary_language.iso6391_name in self.supported_languages
                }
                for doc in response if not doc.is_error
            ]
        except Exception as e:
            logger.error(f"Azure Text Analytics language detection failed: {e}")
            return [{"language": "en", "confidence": 0.5, "supported": True}] * len(texts)

    async def _recognize_entities_batch(self, texts: List[str]) -> List[Dict]:
        """Batch entity recognition with Azure Text Analytics"""
        try:
            response = self.client.recognize_entities(documents=texts)
            return [
                {
                    "entities": [
                        {
                            "text": entity.text,
                            "category": entity.category,
                            "confidence": entity.confidence_score
                        }
                        for entity in doc.entities
                    ] if not doc.is_error else []
                }
                for doc in response
            ]
        except Exception as e:
            logger.error(f"Azure Text Analytics entity recognition failed: {e}")
            return [{"entities": []}] * len(texts)

    async def _extract_key_phrases_batch(self, texts: List[str]) -> List[Dict]:
        """Batch key phrase extraction"""
        try:
            response = self.client.extract_key_phrases(documents=texts)
            return [
                {
                    "key_phrases": doc.key_phrases if not doc.is_error else []
                }
                for doc in response
            ]
        except Exception as e:
            logger.error(f"Azure Text Analytics key phrase extraction failed: {e}")
            return [{"key_phrases": []}] * len(texts)

    def _calculate_preprocessing_confidence(
        self,
        language_results: List[Dict],
        entity_results: List[Dict]
    ) -> float:
        """Calculate overall preprocessing confidence score"""
        if not language_results or not entity_results:
            return 0.5

        # Average language confidence
        lang_confidence = sum(r.get("confidence", 0.5) for r in language_results) / len(language_results)

        # Entity detection confidence (based on number of entities found)
        entity_confidence = min(1.0, sum(len(r.get("entities", [])) for r in entity_results) / len(entity_results) * 0.1)

        return (lang_confidence + entity_confidence) / 2.0

    async def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """Validate text quality for extraction processing"""
        try:
            # Language detection
            lang_result = await self._detect_language_batch([text])

            # Entity recognition
            entity_result = await self._recognize_entities_batch([text])

            # Sentiment analysis
            sentiment_result = await self._analyze_sentiment([text])

            return {
                "language": lang_result[0] if lang_result else {"language": "en", "confidence": 0.5},
                "entities_found": len(entity_result[0].get("entities", [])) if entity_result else 0,
                "sentiment": sentiment_result[0] if sentiment_result else {"sentiment": "neutral", "confidence": 0.5},
                "quality_score": self._calculate_text_quality_score(lang_result, entity_result, sentiment_result)
            }
        except Exception as e:
            logger.error(f"Text quality validation failed: {e}")
            return {
                "language": {"language": "en", "confidence": 0.5},
                "entities_found": 0,
                "sentiment": {"sentiment": "neutral", "confidence": 0.5},
                "quality_score": 0.5
            }

    async def _analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment of texts"""
        try:
            response = self.client.analyze_sentiment(documents=texts)
            return [
                {
                    "sentiment": doc.sentiment,
                    "confidence": doc.confidence_scores.get(doc.sentiment, 0.5)
                }
                for doc in response if not doc.is_error
            ]
        except Exception as e:
            logger.error(f"Azure Text Analytics sentiment analysis failed: {e}")
            return [{"sentiment": "neutral", "confidence": 0.5}] * len(texts)

    def _calculate_text_quality_score(
        self,
        language_results: List[Dict],
        entity_results: List[Dict],
        sentiment_results: List[Dict]
    ) -> float:
        """Calculate overall text quality score for extraction"""
        if not language_results or not entity_results or not sentiment_results:
            return 0.5

        # Language confidence
        lang_score = language_results[0].get("confidence", 0.5)

        # Entity richness (more entities = better quality)
        entity_count = len(entity_results[0].get("entities", []))
        entity_score = min(1.0, entity_count / 10.0)  # Normalize to 0-1

        # Sentiment confidence
        sentiment_score = sentiment_results[0].get("confidence", 0.5)

        # Weighted average
        return (lang_score * 0.4 + entity_score * 0.4 + sentiment_score * 0.2)