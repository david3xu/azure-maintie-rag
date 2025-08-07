"""
Domain Intelligence Agent Integration for Automatic Constant Generation
======================================================================

This module provides the integration layer between the Domain Intelligence Agent
and the automation system for automatic constant generation based on domain analysis.

Phase 3 Implementation: Domain-driven constant optimization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .automation_interface import GenerationResult, LearningMechanism


logger = logging.getLogger(__name__)


@dataclass
class DomainAnalysisResult:
    """Result of domain analysis for constant generation"""

    domain_name: str
    entity_density: float
    relationship_complexity: float
    vocabulary_richness: float
    document_complexity: str
    technical_vocabulary_size: int
    average_document_length: int
    processing_patterns: Dict[str, float]
    confidence_score: float
    analysis_timestamp: datetime
    corpus_statistics: Dict[str, Any]


class DomainIntelligenceConstantGenerator:
    """Generates constants based on Domain Intelligence Agent analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._analysis_cache: Dict[str, DomainAnalysisResult] = {}
        self._cache_ttl_hours = 24

    async def generate_domain_adaptive_constants(
        self, domain_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Generate domain-adaptive constants using Domain Intelligence analysis"""

        try:
            # Get or perform domain analysis
            domain_analysis = await self._get_domain_analysis(domain_name, context)

            if not domain_analysis:
                self.logger.warning(f"No domain analysis available for {domain_name}")
                return {}, 0.0, "no_analysis"

            # Generate domain-adaptive constants based on analysis
            constants = await self._generate_constants_from_analysis(domain_analysis)

            # Calculate confidence based on analysis quality
            confidence = self._calculate_generation_confidence(domain_analysis)

            self.logger.info(
                f"Generated {len(constants)} domain-adaptive constants for {domain_name} "
                f"with confidence {confidence:.3f}"
            )

            return (
                constants,
                confidence,
                f"Domain Intelligence Analysis ({domain_analysis.confidence_score:.3f})",
            )

        except Exception as e:
            self.logger.error(f"Failed to generate domain-adaptive constants: {e}")
            return {}, 0.0, f"error: {str(e)}"

    async def _get_domain_analysis(
        self, domain_name: str, context: Dict[str, Any]
    ) -> Optional[DomainAnalysisResult]:
        """Get domain analysis, using cache or performing new analysis"""

        # Check cache first
        cached_analysis = self._analysis_cache.get(domain_name)
        if cached_analysis and self._is_analysis_fresh(cached_analysis):
            return cached_analysis

        # Perform new domain analysis
        try:
            # Extract analysis context
            domain_data = context.get("domain_analysis", {})

            if not domain_data:
                # Trigger domain analysis through content analyzer
                domain_data = await self._perform_domain_analysis(domain_name, context)

            # Create analysis result
            analysis = DomainAnalysisResult(
                domain_name=domain_name,
                entity_density=domain_data.get("entity_density", 0.1),
                relationship_complexity=domain_data.get("relationship_complexity", 0.5),
                vocabulary_richness=domain_data.get("technical_vocabulary_size", 100)
                / 1000.0,
                document_complexity=domain_data.get("domain_complexity", "medium"),
                technical_vocabulary_size=domain_data.get(
                    "technical_vocabulary_size", 100
                ),
                average_document_length=domain_data.get(
                    "average_document_length", 1500
                ),
                processing_patterns=domain_data.get("processing_patterns", {}),
                confidence_score=domain_data.get("analysis_confidence", 0.8),
                analysis_timestamp=datetime.now(),
                corpus_statistics=context.get("corpus_statistics", {}),
            )

            # Cache the analysis
            self._analysis_cache[domain_name] = analysis

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to get domain analysis: {e}")
            return None

    async def _perform_domain_analysis(
        self, domain_name: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform domain analysis using available analyzers"""

        try:
            # Import dynamically to avoid circular imports
            from ...domain_intelligence.analyzers.unified_content_analyzer import (
                UnifiedContentAnalyzer,
            )

            analyzer = UnifiedContentAnalyzer()

            # Determine corpus path (would be configured based on domain)
            corpus_path = self._get_corpus_path_for_domain(domain_name)

            if corpus_path:
                # Analyze the corpus
                domain_profile = analyzer.analyze_content(corpus_path)

                return {
                    "entity_density": self._estimate_entity_density(domain_profile),
                    "relationship_complexity": self._estimate_relationship_complexity(
                        domain_profile
                    ),
                    "technical_vocabulary_size": len(
                        domain_profile.technical_vocabulary
                    ),
                    "domain_complexity": domain_profile.document_complexity.complexity_class,
                    "average_document_length": domain_profile.text_statistics.total_words,
                    "analysis_confidence": domain_profile.analysis_confidence.confidence,
                    "processing_patterns": {
                        "lexical_diversity": domain_profile.text_statistics.lexical_diversity,
                        "readability_score": domain_profile.text_statistics.readability_score,
                        "avg_sentence_length": domain_profile.text_statistics.avg_sentence_length,
                    },
                }
            else:
                # Use domain-based estimates
                return self._get_domain_estimates(domain_name)

        except Exception as e:
            self.logger.warning(f"Failed to perform domain analysis: {e}")
            return self._get_domain_estimates(domain_name)

    def _get_corpus_path_for_domain(self, domain_name: str) -> Optional[str]:
        """Get corpus path for domain analysis"""

        # Map domain names to available corpus paths
        domain_path_map = {
            "programming_language": "/workspace/azure-maintie-rag/data/raw/Programming-Language",
            "general": "/workspace/azure-maintie-rag/data/raw/Programming-Language",  # Default
        }

        return domain_path_map.get(domain_name)

    def _get_domain_estimates(self, domain_name: str) -> Dict[str, Any]:
        """Get estimated characteristics for domain"""

        # Domain-specific estimates based on typical characteristics
        domain_estimates = {
            "programming_language": {
                "entity_density": 0.15,
                "relationship_complexity": 0.8,
                "technical_vocabulary_size": 500,
                "domain_complexity": "high",
                "average_document_length": 2500,
                "analysis_confidence": 0.7,
            },
            "academic": {
                "entity_density": 0.12,
                "relationship_complexity": 0.7,
                "technical_vocabulary_size": 400,
                "domain_complexity": "high",
                "average_document_length": 3000,
                "analysis_confidence": 0.7,
            },
            "business": {
                "entity_density": 0.08,
                "relationship_complexity": 0.5,
                "technical_vocabulary_size": 200,
                "domain_complexity": "medium",
                "average_document_length": 1200,
                "analysis_confidence": 0.6,
            },
            "general": {
                "entity_density": 0.06,
                "relationship_complexity": 0.4,
                "technical_vocabulary_size": 150,
                "domain_complexity": "medium",
                "average_document_length": 1000,
                "analysis_confidence": 0.5,
            },
        }

        return domain_estimates.get(domain_name, domain_estimates["general"])

    def _estimate_entity_density(self, domain_profile) -> float:
        """Estimate entity density from domain profile"""

        try:
            # Calculate based on technical vocabulary size and content complexity
            vocab_size = len(domain_profile.technical_vocabulary)
            complexity_factor = {"low": 0.5, "medium": 0.75, "high": 1.0}.get(
                domain_profile.document_complexity.complexity_class, 0.75
            )

            # Normalize based on vocabulary density
            base_density = min(0.2, vocab_size / 2000.0)  # Max 20% entity density
            return base_density * complexity_factor

        except Exception:
            return 0.1  # Default fallback

    def _estimate_relationship_complexity(self, domain_profile) -> float:
        """Estimate relationship complexity from domain profile"""

        try:
            # Base complexity on document structure and vocabulary richness
            lexical_diversity = domain_profile.text_statistics.lexical_diversity or 0.5
            avg_sentence_length = (
                domain_profile.text_statistics.avg_sentence_length or 15
            )

            # Complex relationships likely in documents with:
            # - High lexical diversity (varied vocabulary)
            # - Longer sentences (complex structures)
            # - Technical domains

            complexity_base = min(
                1.0, lexical_diversity * 1.2
            )  # Lexical diversity influence
            sentence_factor = min(
                1.2, avg_sentence_length / 20.0
            )  # Sentence complexity influence

            return min(0.9, complexity_base * sentence_factor)

        except Exception:
            return 0.5  # Default fallback

    async def _generate_constants_from_analysis(
        self, analysis: DomainAnalysisResult
    ) -> Dict[str, Any]:
        """Generate constants based on domain analysis"""

        constants = {}

        # === Entity Extraction Thresholds ===
        # Adjust confidence thresholds based on entity density
        if analysis.entity_density > 0.12:  # High entity density
            constants["ENTITY_CONFIDENCE_THRESHOLD"] = (
                0.85  # Higher threshold for quality
            )
        elif analysis.entity_density < 0.05:  # Low entity density
            constants["ENTITY_CONFIDENCE_THRESHOLD"] = (
                0.70  # Lower threshold for recall
            )
        else:
            constants["ENTITY_CONFIDENCE_THRESHOLD"] = 0.80  # Balanced threshold

        # Relationship thresholds based on complexity
        if analysis.relationship_complexity > 0.7:  # High complexity
            constants["RELATIONSHIP_CONFIDENCE_THRESHOLD"] = 0.75
            constants["MIN_RELATIONSHIP_STRENGTH"] = 0.6
        elif analysis.relationship_complexity < 0.4:  # Low complexity
            constants["RELATIONSHIP_CONFIDENCE_THRESHOLD"] = 0.65
            constants["MIN_RELATIONSHIP_STRENGTH"] = 0.4
        else:
            constants["RELATIONSHIP_CONFIDENCE_THRESHOLD"] = 0.70
            constants["MIN_RELATIONSHIP_STRENGTH"] = 0.5

        # === Document Processing Parameters ===
        # Optimize chunk size based on document characteristics
        if analysis.average_document_length > 2000:  # Long documents
            constants["DEFAULT_CHUNK_SIZE"] = min(
                1500, analysis.average_document_length // 3
            )
        elif analysis.average_document_length < 500:  # Short documents
            constants["DEFAULT_CHUNK_SIZE"] = max(
                300, analysis.average_document_length // 2
            )
        else:
            constants["DEFAULT_CHUNK_SIZE"] = 1000  # Standard size

        # Chunk overlap based on complexity
        chunk_size = constants.get("DEFAULT_CHUNK_SIZE", 1000)
        if analysis.relationship_complexity > 0.6:
            constants["DEFAULT_CHUNK_OVERLAP"] = int(
                chunk_size * 0.25
            )  # More overlap for complex relationships
        else:
            constants["DEFAULT_CHUNK_OVERLAP"] = int(
                chunk_size * 0.2
            )  # Standard overlap

        # Entities per chunk based on density
        base_entities = 10
        density_multiplier = min(3.0, max(0.5, analysis.entity_density * 10))
        constants["MAX_ENTITIES_PER_CHUNK"] = int(base_entities * density_multiplier)

        # === Search Quality Thresholds ===
        # Domain-specific relevance thresholds
        if analysis.technical_vocabulary_size > 300:  # Technical domain
            constants["RESULT_RELEVANCE_THRESHOLD"] = (
                0.70  # Higher precision for technical content
            )
        else:
            constants["RESULT_RELEVANCE_THRESHOLD"] = 0.60  # Standard relevance

        # Domain classification thresholds
        if analysis.vocabulary_richness > 0.3:  # Rich vocabulary
            constants["DOMAIN_DETECTION_THRESHOLD"] = 0.80
            constants["TECHNICAL_CONTENT_SIMILARITY_THRESHOLD"] = 0.85
        else:
            constants["DOMAIN_DETECTION_THRESHOLD"] = 0.75
            constants["TECHNICAL_CONTENT_SIMILARITY_THRESHOLD"] = 0.80

        self.logger.debug(
            f"Generated constants for {analysis.domain_name}: "
            f"entity_threshold={constants.get('ENTITY_CONFIDENCE_THRESHOLD', 'N/A')}, "
            f"chunk_size={constants.get('DEFAULT_CHUNK_SIZE', 'N/A')}, "
            f"entities_per_chunk={constants.get('MAX_ENTITIES_PER_CHUNK', 'N/A')}"
        )

        return constants

    def _calculate_generation_confidence(self, analysis: DomainAnalysisResult) -> float:
        """Calculate confidence score for generated constants"""

        confidence_factors = []

        # Domain analysis confidence (primary factor)
        confidence_factors.append(analysis.confidence_score * 0.4)

        # Corpus size factor (more data = higher confidence)
        corpus_size_factor = min(
            1.0, analysis.corpus_statistics.get("document_count", 10) / 100.0
        )
        confidence_factors.append(corpus_size_factor * 0.2)

        # Vocabulary richness factor
        vocab_factor = min(1.0, analysis.vocabulary_richness)
        confidence_factors.append(vocab_factor * 0.2)

        # Complexity clarity factor (clearer complexity = higher confidence)
        complexity_clarity = {"low": 0.8, "medium": 0.9, "high": 0.95}.get(
            analysis.document_complexity, 0.8
        )
        confidence_factors.append(complexity_clarity * 0.2)

        return sum(confidence_factors)

    def _is_analysis_fresh(self, analysis: DomainAnalysisResult) -> bool:
        """Check if domain analysis is still fresh"""

        age_hours = (
            datetime.now() - analysis.analysis_timestamp
        ).total_seconds() / 3600
        return age_hours < self._cache_ttl_hours

    async def generate_content_analysis_constants(
        self, domain_name: str, context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, str]:
        """Generate content analysis adaptive constants"""

        try:
            domain_analysis = await self._get_domain_analysis(domain_name, context)

            if not domain_analysis:
                return {}, 0.0, "no_analysis"

            constants = {}

            # Document size classifications based on corpus analysis
            if domain_analysis.average_document_length > 2000:
                constants["SHORT_DOCUMENT_WORD_THRESHOLD"] = (
                    200  # Higher threshold for long-document domain
                )
            elif domain_analysis.average_document_length < 800:
                constants["SHORT_DOCUMENT_WORD_THRESHOLD"] = (
                    50  # Lower threshold for short-document domain
                )
            else:
                constants["SHORT_DOCUMENT_WORD_THRESHOLD"] = 100  # Standard threshold

            # Complexity analysis thresholds
            if domain_analysis.document_complexity == "high":
                constants["MEDIUM_COMPLEXITY_THRESHOLD"] = (
                    0.5  # Higher threshold for complex domains
                )
                constants["TECHNICAL_DENSITY_THRESHOLD"] = (
                    0.15  # Expect more technical content
                )
            elif domain_analysis.document_complexity == "low":
                constants["MEDIUM_COMPLEXITY_THRESHOLD"] = (
                    0.3  # Lower threshold for simple domains
                )
                constants["TECHNICAL_DENSITY_THRESHOLD"] = (
                    0.05  # Expect less technical content
                )
            else:
                constants["MEDIUM_COMPLEXITY_THRESHOLD"] = 0.4  # Standard threshold
                constants["TECHNICAL_DENSITY_THRESHOLD"] = (
                    0.1  # Standard technical density
                )

            # Similarity thresholds based on vocabulary richness
            if domain_analysis.vocabulary_richness > 0.3:
                constants["HIGH_SIMILARITY_THRESHOLD"] = (
                    0.5  # Higher threshold for rich vocabulary
                )
            else:
                constants["HIGH_SIMILARITY_THRESHOLD"] = 0.4  # Standard threshold

            # Optimal chunk sizes based on document structure
            constants["OPTIMAL_CHUNK_SIZE_MIN"] = max(
                300, domain_analysis.average_document_length // 8
            )
            constants["OPTIMAL_CHUNK_SIZE_MAX"] = min(
                2500, domain_analysis.average_document_length // 2
            )

            confidence = self._calculate_generation_confidence(domain_analysis)

            return constants, confidence, "Content Analysis Optimization"

        except Exception as e:
            self.logger.error(f"Failed to generate content analysis constants: {e}")
            return {}, 0.0, f"error: {str(e)}"

    def get_cached_analysis(self, domain_name: str) -> Optional[DomainAnalysisResult]:
        """Get cached domain analysis if available"""
        return self._analysis_cache.get(domain_name)

    def clear_analysis_cache(self, domain_name: Optional[str] = None) -> None:
        """Clear domain analysis cache"""
        if domain_name:
            self._analysis_cache.pop(domain_name, None)
        else:
            self._analysis_cache.clear()

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information"""
        return {
            "cached_domains": list(self._analysis_cache.keys()),
            "cache_count": len(self._analysis_cache),
            "cache_ttl_hours": self._cache_ttl_hours,
            "fresh_analyses": [
                domain
                for domain, analysis in self._analysis_cache.items()
                if self._is_analysis_fresh(analysis)
            ],
        }


# Global generator instance
domain_intelligence_generator = DomainIntelligenceConstantGenerator()


# Export all classes and instances
__all__ = [
    "DomainAnalysisResult",
    "DomainIntelligenceConstantGenerator",
    "domain_intelligence_generator",
]
