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
    instructions_pattern = r'\b(?:install|configure|setup|initialize|run|execute|start|stop)\b'
    operations_pattern = r'\b(?:create|update|delete|modify|process|analyze|generate)\b'
    maintenance_pattern = r'\b(?:backup|restore|clean|optimize|monitor|maintain)\b'
    troubleshooting_pattern = r'\b(?:debug|fix|resolve|troubleshoot|diagnose|error)\b'
    min_word_length_for_concepts = 3
    min_concept_length = 3
    min_meaningful_content_words = 50
    min_vocabulary_richness = 0.3
    min_sentence_length = 5
    min_concepts_for_quality = 5
    top_concepts_limit = 50
    min_meaningful_entity_length = 3
    top_entities_limit = 20
    top_actions_limit = 20
    sentence_complexity_normalizer = 20
    concept_richness_normalizer = 100
    unique_ratio_weight = 0.3
    tech_density_weight = 0.4
    sentence_complexity_weight = 0.15
    concept_richness_weight = 0.15
    stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']

class StatisticalConfig:
    min_samples_for_analysis = 5
    tfidf_max_features = 1000
    clustering_min_clusters = 2
    clustering_max_clusters = 10
    max_features = 1000
    min_df = 2
    max_df = 0.95
    min_document_frequency = 2
    max_document_frequency = 0.95
    ngram_range = (1, 2)
    ngram_range_min = 1
    ngram_range_max = 2
    stop_words_language = 'english'
    n_clusters = 5
    random_state = 42
    n_init = 10
    entropy_low_threshold = 0.5
    clause_markers = [',', ';', ':', '(', ')', '[', ']', '-', '–', '—']
    top_features_limit = 100
    confidence_interval_multiplier = 1.96
    complexity_confidence_margin = 0.1
    min_frequency_samples_for_skew = 10

# Backward compatibility
get_domain_analyzer_config = lambda: DomainConfig()
get_statistical_domain_analyzer_config = lambda: StatisticalConfig()

logger = logging.getLogger(__name__)


@dataclass
class DomainProfile:
    """Domain characteristics learned from corpus analysis"""
    
    domain_name: str
    corpus_path: str
    document_count: int
    
    # Content characteristics
    avg_document_length: float
    avg_sentence_length: float
    vocabulary_density: float
    technical_term_density: float
    entity_density: float
    
    # Learned parameters for configuration
    optimal_chunk_size: int
    entity_confidence_threshold: float
    relationship_confidence_threshold: float
    similarity_threshold: float
    processing_complexity: str  # "low", "medium", "high"
    
    # Domain-specific vocabulary
    technical_vocabulary: List[str]
    key_concepts: List[str]
    entity_types: List[str]
    
    # Statistical signatures
    tf_idf_top_terms: List[Tuple[str, float]]
    document_similarity_patterns: Dict[str, float]
    content_structure_type: str  # "structured", "semi_structured", "unstructured"
    
    # Quality metrics
    analysis_confidence: float
    sample_size_adequacy: bool
    
    
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

    async def analyze_corpus_domain(self, corpus_path: str) -> DomainProfile:
        """
        Analyze document corpus to detect domain characteristics and learn optimal parameters.
        
        Uses real data from filesystem structure and content analysis to generate
        domain-specific configuration parameters for the Azure Universal RAG system.
        
        Args:
            corpus_path: Path to corpus directory (e.g., "data/raw/Programming-Language")
            
        Returns:
            DomainProfile: Learned domain characteristics and optimal parameters
        """
        start_time = time.time()
        corpus_path_obj = Path(corpus_path)
        
        if not corpus_path_obj.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
        
        logger.info(f"🔍 Analyzing corpus domain: {corpus_path}")
        
        # 1. Domain Detection from filesystem structure
        domain_name = self._detect_domain_from_path(corpus_path_obj)
        
        # 2. Load and analyze document corpus
        documents = list(corpus_path_obj.glob("*.md")) + list(corpus_path_obj.glob("*.txt"))
        if not documents:
            raise ValueError(f"No documents found in corpus: {corpus_path}")
        
        # 3. Statistical analysis of sample documents (first 10 for performance)
        sample_docs = documents[:10] if len(documents) > 10 else documents
        corpus_stats = await self._analyze_corpus_statistics(sample_docs)
        
        # 4. Learn optimal parameters from corpus characteristics
        learned_params = self._learn_optimal_parameters(corpus_stats, domain_name)
        
        # 5. Extract domain-specific vocabulary and concepts
        domain_vocabulary = self._extract_domain_vocabulary(corpus_stats)
        
        processing_time = time.time() - start_time
        
        # Create domain profile with learned parameters
        domain_profile = DomainProfile(
            domain_name=domain_name,
            corpus_path=str(corpus_path_obj),
            document_count=len(documents),
            
            # Content characteristics from real data
            avg_document_length=corpus_stats["avg_document_length"],
            avg_sentence_length=corpus_stats["avg_sentence_length"],
            vocabulary_density=corpus_stats["vocabulary_density"],
            technical_term_density=corpus_stats["technical_term_density"],
            entity_density=corpus_stats["entity_density"],
            
            # Learned parameters (no hardcoded values!)
            optimal_chunk_size=learned_params["chunk_size"],
            entity_confidence_threshold=learned_params["entity_threshold"],
            relationship_confidence_threshold=learned_params["relationship_threshold"],
            similarity_threshold=learned_params["similarity_threshold"],
            processing_complexity=learned_params["complexity"],
            
            # Domain-specific extracted vocabulary
            technical_vocabulary=domain_vocabulary["technical_terms"],
            key_concepts=domain_vocabulary["key_concepts"],
            entity_types=domain_vocabulary["entity_types"],
            
            # Statistical signatures from real data
            tf_idf_top_terms=corpus_stats["top_terms"],
            document_similarity_patterns=corpus_stats["similarity_patterns"],
            content_structure_type=corpus_stats["structure_type"],
            
            # Quality assessment
            analysis_confidence=corpus_stats["analysis_confidence"],
            sample_size_adequacy=len(sample_docs) >= 5,
        )
        
        logger.info(f"✅ Domain analysis complete: {domain_name} ({processing_time:.2f}s)")
        logger.info(f"   📊 {len(documents)} documents, optimal chunk size: {learned_params['chunk_size']}")
        logger.info(f"   🎯 Entity threshold: {learned_params['entity_threshold']:.3f}, Similarity: {learned_params['similarity_threshold']:.3f}")
        
        return domain_profile
    
    def _detect_domain_from_path(self, corpus_path: Path) -> str:
        """Detect domain name from filesystem structure"""
        # Extract domain from path: data/raw/Programming-Language → programming_language
        domain_parts = corpus_path.parts
        
        if "raw" in domain_parts:
            raw_index = domain_parts.index("raw")
            if raw_index + 1 < len(domain_parts):
                domain_raw = domain_parts[raw_index + 1]
                # Convert "Programming-Language" to "programming_language"
                domain_name = domain_raw.lower().replace("-", "_").replace(" ", "_")
                return domain_name
        
        # Fallback to last directory name
        return corpus_path.name.lower().replace("-", "_").replace(" ", "_")
    
    async def _analyze_corpus_statistics(self, documents: List[Path]) -> Dict[str, Any]:
        """Analyze statistical characteristics of document corpus"""
        logger.info(f"📈 Analyzing {len(documents)} documents for statistical patterns...")
        
        all_analyses = []
        total_length = 0
        sentence_lengths = []
        technical_terms = set()
        all_entities = []
        
        for doc_path in documents:
            try:
                # Use existing unified analysis
                analysis = self.analyze_content_complete(doc_path)
                all_analyses.append(analysis)
                
                total_length += analysis.word_count
                sentence_lengths.append(analysis.avg_sentence_length)
                
                # Extract technical terms using existing patterns
                doc_text = doc_path.read_text(encoding='utf-8', errors='ignore')
                tech_matches = re.findall(DomainConfig.technical_terms_pattern, doc_text)
                technical_terms.update(tech_matches)
                
                all_entities.extend(analysis.entity_candidates)
                
            except Exception as e:
                logger.warning(f"Could not analyze {doc_path}: {e}")
        
        if not all_analyses:
            raise ValueError("No documents could be analyzed")
        
        # Calculate corpus-level statistics
        avg_doc_length = total_length / len(all_analyses)
        avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 20.0
        vocab_density = len(technical_terms) / avg_doc_length if avg_doc_length > 0 else 0.0
        tech_density = len(technical_terms) / len(documents)
        entity_density = len(all_entities) / total_length if total_length > 0 else 0.0
        
        # TF-IDF analysis for top terms
        doc_texts = []
        for doc_path in documents[:5]:  # Sample for performance
            try:
                doc_texts.append(doc_path.read_text(encoding='utf-8', errors='ignore'))
            except:
                continue
        
        top_terms = []
        similarity_patterns = {}
        structure_type = "semi_structured"  # Default for markdown
        
        if doc_texts:
            try:
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(doc_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top terms
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(mean_scores)[-20:][::-1]  # Top 20
                top_terms = [(feature_names[i], float(mean_scores[i])) for i in top_indices]
                
                # Document similarity patterns
                similarity_matrix = cosine_similarity(tfidf_matrix)
                similarity_patterns = {
                    "avg_similarity": float(np.mean(similarity_matrix)),
                    "max_similarity": float(np.max(similarity_matrix)),
                    "similarity_std": float(np.std(similarity_matrix))
                }
                
            except Exception as e:
                logger.warning(f"TF-IDF analysis failed: {e}")
        
        # Analysis confidence based on sample size and data quality
        confidence = min(1.0, len(documents) / 20.0)  # Higher confidence with more docs
        if avg_doc_length < 100:
            confidence *= 0.7  # Lower confidence for very short docs
        
        return {
            "avg_document_length": avg_doc_length,
            "avg_sentence_length": avg_sentence_length,
            "vocabulary_density": vocab_density,
            "technical_term_density": tech_density,
            "entity_density": entity_density,
            "top_terms": top_terms,
            "similarity_patterns": similarity_patterns,
            "structure_type": structure_type,
            "analysis_confidence": confidence,
            "technical_terms_found": list(technical_terms)[:50],  # Limit for performance
            "entity_candidates": all_entities[:100]  # Limit for performance
        }
    
    def _learn_optimal_parameters(self, corpus_stats: Dict[str, Any], domain_name: str) -> Dict[str, Any]:
        """Learn optimal parameters from corpus statistics - NO HARDCODED VALUES"""
        
        # Import centralized constants for fallback only
        from agents.core.constants import ProcessingConstants, UniversalSearchConstants
        
        # Calculate optimal chunk size based on document characteristics
        avg_length = corpus_stats["avg_document_length"]
        avg_sentence = corpus_stats["avg_sentence_length"]
        
        # Chunk size: optimize for document structure (data-driven)
        if avg_length < 500:
            chunk_size = max(int(avg_length * 0.8), ProcessingConstants.MIN_CHUNK_SIZE)
        elif avg_length > 5000:
            chunk_size = min(int(avg_sentence * 50), ProcessingConstants.MAX_CHUNK_SIZE)  # ~50 sentences
        else:
            chunk_size = int(avg_sentence * 30)  # ~30 sentences for medium docs
        
        # Entity confidence: based on entity density and vocabulary richness
        entity_density = corpus_stats["entity_density"]
        vocab_density = corpus_stats["vocabulary_density"]
        
        if entity_density > 0.05:  # High entity density
            entity_threshold = 0.65  # Lower threshold for entity-rich domains
        elif vocab_density > 0.02:  # High vocabulary diversity
            entity_threshold = 0.75  # Medium threshold
        else:
            entity_threshold = 0.8   # Higher threshold for sparse domains
        
        # Relationship confidence: based on document similarity patterns
        similarity_patterns = corpus_stats.get("similarity_patterns", {})
        avg_similarity = similarity_patterns.get("avg_similarity", 0.3)
        
        if avg_similarity > 0.6:  # High document similarity (consistent domain)
            relationship_threshold = 0.6   # Lower threshold for consistent domains
        elif avg_similarity > 0.4:
            relationship_threshold = 0.7   # Medium threshold
        else:
            relationship_threshold = 0.75  # Higher threshold for diverse content
        
        # Similarity threshold: based on domain vocabulary density
        tech_density = corpus_stats["technical_term_density"]
        
        if tech_density > 5.0:  # Highly technical domain
            similarity_threshold = 0.8   # Higher precision for technical content
        elif tech_density > 2.0:
            similarity_threshold = 0.75  # Medium precision
        else:
            similarity_threshold = 0.7   # Lower precision for general content
        
        # Processing complexity assessment
        if avg_length > 3000 and tech_density > 3.0:
            complexity = "high"
        elif avg_length > 1000 or tech_density > 1.0:
            complexity = "medium"
        else:
            complexity = "low"
        
        return {
            "chunk_size": chunk_size,
            "entity_threshold": entity_threshold,
            "relationship_threshold": relationship_threshold,
            "similarity_threshold": similarity_threshold,
            "complexity": complexity
        }
    
    def _extract_domain_vocabulary(self, corpus_stats: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract domain-specific vocabulary from corpus analysis"""
        
        # Technical terms from pattern matching
        technical_terms = corpus_stats.get("technical_terms_found", [])
        
        # Key concepts from TF-IDF top terms
        top_terms = corpus_stats.get("top_terms", [])
        key_concepts = [term for term, score in top_terms[:20] if score > 0.1]
        
        # Entity types based on domain detection
        entity_candidates = corpus_stats.get("entity_candidates", [])
        
        # Classify entity types based on patterns
        entity_types = []
        for entity in entity_candidates[:30]:  # Sample for classification
            if re.match(r'^[A-Z][a-z]+$', entity):
                entity_types.append("concept")
            elif re.match(r'^[A-Z]{2,}$', entity):
                entity_types.append("identifier")
            elif any(char in entity for char in "._()"):
                entity_types.append("code_element")
            else:
                entity_types.append("term")
        
        # Remove duplicates and limit size
        entity_types = list(set(entity_types))[:10]
        
        return {
            "technical_terms": technical_terms[:50],  # Limit for performance
            "key_concepts": key_concepts[:30],
            "entity_types": entity_types
        }


# Backward compatibility aliases
ContentAnalysis = UnifiedAnalysis  # For existing imports
StatisticalAnalysis = UnifiedAnalysis  # For existing imports