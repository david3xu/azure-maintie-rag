"""
Unified Content Analyzer - Consolidated Statistical Analysis

This module consolidates ContentAnalyzer and StatisticalDomainAnalyzer into a single,
unified analyzer that provides both basic and advanced statistical analysis without
any domain classification bias.

Key Features:
- Basic content analysis (word count, vocabulary richness, complexity)
- Advanced statistical methods (TF-IDF, entropy, clustering)  
- Quality validation and ML feature extraction
- Zero domain classification bias or semantic assumptions
- Single unified API for all content analysis needs

Architecture Benefits:
- Eliminates redundant statistical calculations
- Single source of truth for content analysis
- Cleaner separation of concerns
- Better performance through unified processing
"""

import logging
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Clean configuration (CODING_STANDARDS compliant)
# Simple pattern configurations for content analysis
class DomainConfig:
    technical_terms_pattern = r'\b[A-Z]{2,}(?:[_-][A-Z]{2,})*\b'
    model_names_pattern = r'\b(?:gpt|bert|llama|claude|openai|azure)\w*\b'
    process_steps_pattern = r'\b(?:step|phase|stage|process|workflow)\s+\d+\b'
    measurements_pattern = r'\b\d+(?:\.\d+)?\s*(?:ms|sec|min|mb|gb|kb|%)\b'
    identifiers_pattern = r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b'

class StatisticalConfig:
    min_samples_for_analysis = 5
    tfidf_max_features = 1000
    clustering_min_clusters = 2
    clustering_max_clusters = 10

# Backward compatibility
get_domain_analyzer_config = lambda: DomainConfig()
get_statistical_domain_analyzer_config = lambda: StatisticalConfig()

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAnalysis:
    """Unified content analysis results combining basic and advanced statistics"""
    
    # Basic content metrics (from ContentAnalyzer)
    word_count: int
    unique_words: int
    avg_sentence_length: float
    vocabulary_richness: float
    complexity_score: float
    concept_frequency: Dict[str, int]
    entity_candidates: List[str]
    action_patterns: List[str]
    technical_density: float
    is_meaningful_content: bool
    
    # Advanced statistical metrics (from StatisticalDomainAnalyzer)
    term_frequency_distribution: Dict[str, float]
    entropy_score: float
    complexity_metrics: Dict[str, float]
    clustering_results: Dict[str, Any]
    statistical_signatures: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Unified metadata
    source_file: Optional[str] = None
    processing_time: float = 0.0
    analysis_quality: str = "unknown"


@dataclass 
class ContentQuality:
    """Content quality assessment results"""
    
    is_valid: bool
    quality_score: float
    issues: List[str]
    word_count: int
    vocabulary_richness: float
    reasoning: str


class UnifiedContentAnalyzer:
    """
    Unified content analyzer combining basic and advanced statistical analysis.
    
    Consolidates ContentAnalyzer + StatisticalDomainAnalyzer functionality:
    - Basic content analysis without redundant calculations
    - Advanced statistical methods (TF-IDF, entropy, clustering)
    - Quality validation and ML feature extraction
    - Zero domain classification bias (removed from StatisticalAnalyzer)
    - Single unified processing pipeline
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Get centralized configurations
        self.domain_config = get_domain_analyzer_config()
        self.stat_config = get_statistical_domain_analyzer_config()
        
        # Entity recognition patterns (statistical, not semantic)
        self.entity_patterns = {
            "technical_terms": re.compile(self.domain_config.technical_terms_pattern),
            "model_names": re.compile(self.domain_config.model_names_pattern, re.IGNORECASE),
            "process_steps": re.compile(self.domain_config.process_steps_pattern, re.IGNORECASE),
            "measurements": re.compile(self.domain_config.measurements_pattern, re.IGNORECASE),
            "identifiers": re.compile(self.domain_config.identifiers_pattern),
        }

        # Action pattern recognition (statistical, not semantic)
        self.action_patterns = {
            "instructions": re.compile(self.domain_config.instructions_pattern, re.IGNORECASE),
            "operations": re.compile(self.domain_config.operations_pattern, re.IGNORECASE),
            "maintenance": re.compile(self.domain_config.maintenance_pattern, re.IGNORECASE),
            "troubleshooting": re.compile(self.domain_config.troubleshooting_pattern, re.IGNORECASE),
        }

        # Advanced statistical analysis components
        self.vectorizer = TfidfVectorizer(
            max_features=self.stat_config.max_features,
            stop_words=self.stat_config.stop_words_language,
            ngram_range=(self.stat_config.ngram_range_min, self.stat_config.ngram_range_max),
            min_df=self.stat_config.min_document_frequency,
            max_df=self.stat_config.max_document_frequency,
        )

        # Clustering components
        self.clusterer = KMeans(
            n_clusters=self.stat_config.n_clusters, 
            random_state=self.stat_config.random_state, 
            n_init=self.stat_config.n_init
        )

        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "avg_processing_time": 0.0,
            "quality_validation_count": 0,
            "advanced_analyses": 0,
        }

        logger.info("Unified content analyzer initialized with consolidated functionality")

    def analyze_content_complete(self, content_source: Path) -> UnifiedAnalysis:
        """
        Complete unified analysis combining basic and advanced statistical analysis.
        
        Single processing pass that generates both basic content metrics and 
        advanced statistical features without redundant calculations.
        """
        start_time = time.time()

        try:
            # Read content once
            if content_source.exists():
                with open(content_source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                raise FileNotFoundError(f"Content source not found: {content_source}")

            if not text.strip():
                return self._create_empty_unified_analysis(str(content_source), time.time() - start_time)

            # Single text preprocessing pass
            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Basic metrics (single calculation)
            word_count = len(words)
            unique_words = len(set(word.lower() for word in words))
            avg_sentence_length = (
                statistics.mean(len(s.split()) for s in sentences) if sentences else 0
            )
            vocabulary_richness = unique_words / max(1, word_count)

            # Extract patterns once (shared by basic and advanced analysis)
            concept_frequency = self._extract_concept_frequency(text)
            entity_candidates = self._extract_entity_candidates(text)
            action_patterns = self._extract_action_patterns(text)

            # Basic complexity metrics
            complexity_score = self._calculate_complexity_score(text, concept_frequency)
            technical_density = self._calculate_pattern_density(text, entity_candidates)

            # Advanced statistical analysis (reusing basic calculations)
            term_frequency_distribution = self._calculate_term_frequency_distribution(text)
            entropy_score = self._calculate_entropy(term_frequency_distribution)
            complexity_metrics = self._calculate_complexity_metrics(text, words)
            clustering_results = self._perform_clustering_analysis(text)
            statistical_signatures = self._generate_statistical_signatures(
                term_frequency_distribution, entropy_score, complexity_metrics
            )
            confidence_intervals = self._calculate_confidence_intervals(
                term_frequency_distribution, complexity_metrics
            )

            # Content quality validation
            is_meaningful_content = self._validate_content_quality(
                word_count, vocabulary_richness, concept_frequency
            )

            # Determine analysis quality
            analysis_quality = self._assess_analysis_quality(
                word_count, entropy_score, len(concept_frequency)
            )

            processing_time = time.time() - start_time

            # Update unified statistics
            self.analysis_stats["total_analyses"] += 1
            self.analysis_stats["advanced_analyses"] += 1
            self.analysis_stats["avg_processing_time"] = (
                self.analysis_stats["avg_processing_time"]
                * (self.analysis_stats["total_analyses"] - 1)
                + processing_time
            ) / self.analysis_stats["total_analyses"]

            return UnifiedAnalysis(
                # Basic metrics
                word_count=word_count,
                unique_words=unique_words,
                avg_sentence_length=avg_sentence_length,
                vocabulary_richness=vocabulary_richness,
                complexity_score=complexity_score,
                concept_frequency=concept_frequency,
                entity_candidates=entity_candidates,
                action_patterns=action_patterns,
                technical_density=technical_density,
                is_meaningful_content=is_meaningful_content,
                # Advanced metrics
                term_frequency_distribution=term_frequency_distribution,
                entropy_score=entropy_score,
                complexity_metrics=complexity_metrics,
                clustering_results=clustering_results,
                statistical_signatures=statistical_signatures,
                confidence_intervals=confidence_intervals,
                # Metadata
                source_file=str(content_source),
                processing_time=processing_time,
                analysis_quality=analysis_quality,
            )

        except Exception as e:
            logger.error(f"Unified content analysis failed for {content_source}: {e}")
            raise

    def analyze_content_basic(self, content_source: Path) -> UnifiedAnalysis:
        """
        Basic analysis only (for backward compatibility with ContentAnalyzer).
        
        Provides basic content metrics without advanced statistical analysis
        for cases where performance is more important than depth.
        """
        start_time = time.time()

        try:
            # Read content
            if content_source.exists():
                with open(content_source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                raise FileNotFoundError(f"Content source not found: {content_source}")

            # Basic text statistics only
            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            word_count = len(words)
            unique_words = len(set(word.lower() for word in words))
            avg_sentence_length = (
                statistics.mean(len(s.split()) for s in sentences) if sentences else 0
            )
            vocabulary_richness = unique_words / max(1, word_count)

            # Basic pattern extraction
            concept_frequency = self._extract_concept_frequency(text)
            entity_candidates = self._extract_entity_candidates(text)
            action_patterns = self._extract_action_patterns(text)

            # Basic complexity and quality
            complexity_score = self._calculate_complexity_score(text, concept_frequency)
            technical_density = self._calculate_pattern_density(text, entity_candidates)
            is_meaningful_content = self._validate_content_quality(
                word_count, vocabulary_richness, concept_frequency
            )

            processing_time = time.time() - start_time

            # Update statistics
            self.analysis_stats["total_analyses"] += 1
            self.analysis_stats["avg_processing_time"] = (
                self.analysis_stats["avg_processing_time"]
                * (self.analysis_stats["total_analyses"] - 1)
                + processing_time
            ) / self.analysis_stats["total_analyses"]

            return UnifiedAnalysis(
                # Basic metrics only
                word_count=word_count,
                unique_words=unique_words,
                avg_sentence_length=avg_sentence_length,
                vocabulary_richness=vocabulary_richness,
                complexity_score=complexity_score,
                concept_frequency=concept_frequency,
                entity_candidates=entity_candidates,
                action_patterns=action_patterns,
                technical_density=technical_density,
                is_meaningful_content=is_meaningful_content,
                # Empty advanced metrics
                term_frequency_distribution={},
                entropy_score=0.0,
                complexity_metrics={},
                clustering_results={},
                statistical_signatures={},
                confidence_intervals={},
                # Metadata
                source_file=str(content_source),
                processing_time=processing_time,
                analysis_quality="basic",
            )

        except Exception as e:
            logger.error(f"Basic content analysis failed for {content_source}: {e}")
            raise

    def validate_content_quality(self, content_source: Path) -> ContentQuality:
        """
        Validate content quality using unified statistical measures.
        
        Provides objective quality assessment based on both basic and 
        advanced statistical measures for comprehensive validation.
        """
        try:
            analysis = self.analyze_content_complete(content_source)
            
            issues = []
            quality_score = 0.0
            
            # Basic quality checks
            if analysis.word_count < self.domain_config.min_meaningful_content_words:
                issues.append(f"Low word count: {analysis.word_count}")
            else:
                quality_score += 0.25
                
            if analysis.vocabulary_richness < self.domain_config.min_vocabulary_richness:
                issues.append(f"Low vocabulary richness: {analysis.vocabulary_richness:.3f}")
            else:
                quality_score += 0.25
                
            if analysis.avg_sentence_length < self.domain_config.min_sentence_length:
                issues.append(f"Very short sentences: {analysis.avg_sentence_length:.1f}")
            else:
                quality_score += 0.2
                
            if len(analysis.concept_frequency) < self.domain_config.min_concepts_for_quality:
                issues.append(f"Few meaningful concepts: {len(analysis.concept_frequency)}")
            else:
                quality_score += 0.15

            # Advanced quality checks
            if analysis.entropy_score < self.stat_config.entropy_low_threshold:
                issues.append(f"Low information entropy: {analysis.entropy_score:.2f}")
            else:
                quality_score += 0.15
            
            is_valid = len(issues) == 0 and analysis.is_meaningful_content
            
            reasoning = (
                f"Unified quality assessment: "
                f"{analysis.word_count} words, "
                f"{analysis.vocabulary_richness:.3f} vocabulary richness, "
                f"{analysis.entropy_score:.2f} entropy, "
                f"{len(analysis.concept_frequency)} concepts identified"
            )
            
            self.analysis_stats["quality_validation_count"] += 1
            
            return ContentQuality(
                is_valid=is_valid,
                quality_score=quality_score, 
                issues=issues,
                word_count=analysis.word_count,
                vocabulary_richness=analysis.vocabulary_richness,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Quality validation failed for {content_source}: {e}")
            return ContentQuality(
                is_valid=False,
                quality_score=0.0,
                issues=[f"Analysis failed: {e}"],
                word_count=0,
                vocabulary_richness=0.0,
                reasoning="Unified quality validation failed due to analysis error"
            )

    # ========== BASIC ANALYSIS METHODS (from ContentAnalyzer) ==========

    def _extract_concept_frequency(self, text: str) -> Dict[str, int]:
        """Extract concept frequency using statistical patterns"""
        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text_lower)

        concept_freq = Counter()

        # Single meaningful words
        for word in words:
            if (len(word) > self.domain_config.min_word_length_for_concepts and 
                word not in self._get_stop_words()):
                concept_freq[word] += 1

        # Multi-word concepts (bigrams and trigrams)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if self._is_meaningful_concept(bigram):
                concept_freq[bigram] += 1

        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if self._is_meaningful_concept(trigram):
                concept_freq[trigram] += 1

        return dict(concept_freq.most_common(self.domain_config.top_concepts_limit))

    def _extract_entity_candidates(self, text: str) -> List[str]:
        """Extract entity candidates using statistical patterns"""
        entities = []

        for pattern_name, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            entities.extend(matches)

        # Deduplicate and filter
        unique_entities = list(set(entities))
        return [
            entity for entity in unique_entities 
            if len(entity.strip()) > self.domain_config.min_meaningful_entity_length
        ][:self.domain_config.top_entities_limit]

    def _extract_action_patterns(self, text: str) -> List[str]:
        """Extract action patterns using statistical analysis"""
        actions = []

        for pattern_name, pattern in self.action_patterns.items():
            matches = pattern.findall(text)
            actions.extend(matches)

        unique_actions = list(set(action.lower() for action in actions))
        return unique_actions[:self.domain_config.top_actions_limit]

    def _calculate_complexity_score(self, text: str, concept_frequency: Dict[str, int]) -> float:
        """Calculate objective content complexity score"""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        if not words or not sentences:
            return 0.0

        # Vocabulary diversity component
        unique_ratio = len(set(word.lower() for word in words)) / len(words)

        # Pattern density component
        pattern_density = len([word for word in words if len(word) > 6]) / len(words)

        # Sentence complexity component
        avg_sentence_length = statistics.mean(
            len(s.split()) for s in sentences if s.strip()
        )
        sentence_complexity = min(
            1.0, avg_sentence_length / self.domain_config.sentence_complexity_normalizer
        )

        # Concept richness component
        concept_richness = min(
            1.0, len(concept_frequency) / self.domain_config.concept_richness_normalizer
        )

        # Weighted combination
        complexity_score = (
            unique_ratio * self.domain_config.unique_ratio_weight
            + pattern_density * self.domain_config.tech_density_weight
            + sentence_complexity * self.domain_config.sentence_complexity_weight
            + concept_richness * self.domain_config.concept_richness_weight
        )

        return min(1.0, complexity_score)

    def _calculate_pattern_density(self, text: str, entity_candidates: List[str]) -> float:
        """Calculate density of identifiable patterns in text"""
        words = text.split()
        if not words:
            return 0.0

        # Count various pattern indicators
        pattern_indicators = len(entity_candidates)

        # Add other pattern types (technical terms, numbers, etc.)
        pattern_indicators += len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
        pattern_indicators += len(re.findall(r'\b\d+\.?\d*\b', text))  # Numbers
        pattern_indicators += len(re.findall(r'\b[a-z_]+\(\)', text))  # Function calls

        return min(1.0, pattern_indicators / len(words))

    def _validate_content_quality(
        self, word_count: int, vocabulary_richness: float, concept_frequency: Dict[str, int]
    ) -> bool:
        """Validate content quality using objective statistical criteria"""
        
        # Minimum content requirements
        if word_count < self.domain_config.min_meaningful_content_words:
            return False
            
        if vocabulary_richness < self.domain_config.min_vocabulary_richness:
            return False
            
        if len(concept_frequency) < self.domain_config.min_concepts_for_quality:
            return False
            
        return True

    # ========== ADVANCED STATISTICAL METHODS (from StatisticalDomainAnalyzer) ==========

    def _calculate_term_frequency_distribution(self, text: str) -> Dict[str, float]:
        """Calculate normalized term frequency distribution"""
        words = text.lower().split()
        if not words:
            return {}

        word_counts = Counter(words)
        total_words = len(words)

        # Return normalized frequencies
        return {word: count / total_words for word, count in word_counts.items()}

    def _calculate_entropy(self, term_freq: Dict[str, float]) -> float:
        """Calculate Shannon entropy of term distribution"""
        if not term_freq:
            return 0.0

        frequencies = list(term_freq.values())
        entropy = -sum(f * math.log2(f) for f in frequencies if f > 0)
        return entropy

    def _calculate_complexity_metrics(
        self, text: str, words: List[str]
    ) -> Dict[str, float]:
        """Calculate various complexity metrics using statistical methods"""
        if not words:
            return {
                "lexical_diversity": 0.0,
                "avg_word_length": 0.0,
                "sentence_complexity": 0.0,
            }

        # Lexical diversity (Herdan's C)
        unique_words = len(set(word.lower() for word in words))
        total_words = len(words)
        lexical_diversity = (
            math.log(unique_words) / math.log(total_words) if total_words > 1 else 0.0
        )

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Sentence complexity (based on clause markers)
        clause_markers = self.stat_config.clause_markers
        clause_count = sum(text.lower().count(marker) for marker in clause_markers)
        sentence_count = len([s for s in text.split(".") if s.strip()])
        sentence_complexity = clause_count / max(sentence_count, 1)

        return {
            "lexical_diversity": lexical_diversity,
            "avg_word_length": avg_word_length,
            "sentence_complexity": sentence_complexity,
        }

    def _perform_clustering_analysis(self, text: str) -> Dict[str, Any]:
        """Perform statistical clustering analysis on text"""
        try:
            # Vectorize text
            tfidf_matrix = self.vectorizer.fit_transform([text])

            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Create feature importance ranking
            feature_importance = {}
            for i, score in enumerate(tfidf_scores):
                if score > 0:
                    feature_importance[feature_names[i]] = score

            # Sort by importance
            top_features = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                    :self.stat_config.top_features_limit
                ]
            )

            return {
                "feature_count": len(feature_importance),
                "top_features": top_features,
                "max_tfidf_score": max(tfidf_scores) if len(tfidf_scores) > 0 else 0.0,
                "mean_tfidf_score": float(np.mean(tfidf_scores)),
                "tfidf_variance": float(np.var(tfidf_scores)),
            }

        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            return {"feature_count": 0, "top_features": {}, "error": str(e)}

    def _generate_statistical_signatures(
        self, term_freq: Dict[str, float], entropy: float, complexity: Dict[str, float]
    ) -> Dict[str, float]:
        """Generate statistical signatures for content identification"""
        signatures = {
            "entropy_normalized": entropy / 10.0,  # Normalize entropy
            "vocabulary_concentration": self._calculate_vocabulary_concentration(
                term_freq
            ),
            "complexity_composite": sum(complexity.values()) / len(complexity)
            if complexity
            else 0.0,
        }

        # Add frequency-based signatures
        if term_freq:
            top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            signatures["top_term_concentration"] = sum(freq for _, freq in top_terms)
            signatures["frequency_distribution_skew"] = self._calculate_frequency_skew(
                term_freq
            )

        return signatures

    def _calculate_confidence_intervals(
        self, term_freq: Dict[str, float], complexity: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate statistical confidence intervals"""
        intervals = {}

        # Confidence interval for term frequency variance
        if term_freq:
            frequencies = list(term_freq.values())
            mean_freq = statistics.mean(frequencies)
            std_freq = statistics.stdev(frequencies) if len(frequencies) > 1 else 0.0
            intervals["term_frequency"] = (
                mean_freq - self.stat_config.confidence_interval_multiplier * std_freq,
                mean_freq + self.stat_config.confidence_interval_multiplier * std_freq,
            )

        # Confidence intervals for complexity metrics
        for metric, value in complexity.items():
            # Simplified confidence interval (would need more data for proper calculation)
            margin = value * self.stat_config.complexity_confidence_margin
            intervals[f"complexity_{metric}"] = (value - margin, value + margin)

        return intervals

    # ========== SHARED UTILITY METHODS ==========

    def _get_stop_words(self) -> Set[str]:
        """Get stop words for concept extraction"""
        return set(self.domain_config.stop_words)

    def _is_meaningful_concept(self, concept: str) -> bool:
        """Check if a concept is meaningful for analysis"""
        words = concept.split()

        # Filter out concepts with only stop words
        meaningful_words = [
            word for word in words if word not in self._get_stop_words()
        ]

        return (
            len(meaningful_words) > 0 and 
            len(concept) > self.domain_config.min_concept_length
        )

    def _assess_analysis_quality(
        self, word_count: int, entropy_score: float, concept_count: int
    ) -> str:
        """Assess the quality/reliability of the analysis results"""
        if word_count < 100:
            return "low_sample"
        elif word_count < 500:
            return "medium_sample"
        elif entropy_score < 2.0:
            return "low_entropy"
        elif concept_count < 5:
            return "few_concepts"
        else:
            return "high_quality"

    def _calculate_vocabulary_concentration(self, term_freq: Dict[str, float]) -> float:
        """Calculate how concentrated the vocabulary is (Gini coefficient approximation)"""
        if not term_freq:
            return 0.0

        frequencies = sorted(term_freq.values(), reverse=True)
        n = len(frequencies)
        if n == 0:
            return 0.0

        # Simplified Gini coefficient calculation
        cumsum = 0
        for i, freq in enumerate(frequencies):
            cumsum += freq * (n - i)

        return (2 * cumsum) / (n * sum(frequencies)) - (n + 1) / n

    def _calculate_frequency_skew(self, term_freq: Dict[str, float]) -> float:
        """Calculate frequency distribution skewness"""
        if len(term_freq) < self.stat_config.min_frequency_samples_for_skew:
            return 0.0

        frequencies = list(term_freq.values())
        mean_freq = statistics.mean(frequencies)
        std_freq = statistics.stdev(frequencies)

        if std_freq == 0:
            return 0.0

        # Pearson's skewness coefficient
        skewness = sum((freq - mean_freq) ** 3 for freq in frequencies) / (
            len(frequencies) * std_freq**3
        )
        return skewness

    def _create_empty_unified_analysis(
        self, file_path: str, processing_time: float
    ) -> UnifiedAnalysis:
        """Create empty unified analysis for error cases"""
        return UnifiedAnalysis(
            # Basic metrics
            word_count=0,
            unique_words=0,
            avg_sentence_length=0.0,
            vocabulary_richness=0.0,
            complexity_score=0.0,
            concept_frequency={},
            entity_candidates=[],
            action_patterns=[],
            technical_density=0.0,
            is_meaningful_content=False,
            # Advanced metrics
            term_frequency_distribution={},
            entropy_score=0.0,
            complexity_metrics={},
            clustering_results={},
            statistical_signatures={},
            confidence_intervals={},
            # Metadata
            source_file=file_path,
            processing_time=processing_time,
            analysis_quality="error",
        )

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get unified analyzer performance statistics"""
        return {
            "total_analyses": self.analysis_stats["total_analyses"],
            "avg_processing_time": self.analysis_stats["avg_processing_time"],
            "quality_validation_count": self.analysis_stats["quality_validation_count"],
            "advanced_analyses": self.analysis_stats["advanced_analyses"],
            "entity_patterns_loaded": len(self.entity_patterns),
            "action_patterns_loaded": len(self.action_patterns),
        }

    def create_statistical_features(self, analysis: UnifiedAnalysis) -> Dict[str, float]:
        """Create unified statistical features for ML training without domain bias"""
        features = {
            # Basic features
            "word_count_normalized": min(1.0, analysis.word_count / 1000.0),
            "vocabulary_richness": analysis.vocabulary_richness,
            "complexity_score": analysis.complexity_score,
            "avg_sentence_length_normalized": min(1.0, analysis.avg_sentence_length / 20.0),
            "technical_density": analysis.technical_density,
            "concept_diversity": len(analysis.concept_frequency) / max(1, analysis.word_count) * 100,
            "entity_density": len(analysis.entity_candidates) / max(1, analysis.word_count) * 100,
            "action_density": len(analysis.action_patterns) / max(1, analysis.word_count) * 100,
        }

        # Add advanced features if available
        if analysis.entropy_score > 0:
            features.update({
                "entropy_normalized": analysis.entropy_score / 10.0,
                "lexical_diversity": analysis.complexity_metrics.get("lexical_diversity", 0.0),
                "avg_word_length": analysis.complexity_metrics.get("avg_word_length", 0.0),
                "sentence_complexity": analysis.complexity_metrics.get("sentence_complexity", 0.0),
                "vocabulary_concentration": analysis.statistical_signatures.get("vocabulary_concentration", 0.0),
                "frequency_skew": analysis.statistical_signatures.get("frequency_distribution_skew", 0.0),
            })

        return features


# Backward compatibility aliases
ContentAnalysis = UnifiedAnalysis  # For existing imports
StatisticalAnalysis = UnifiedAnalysis  # For existing imports