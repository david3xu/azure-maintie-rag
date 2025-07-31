"""
Universal Pattern Learner - Domain-Agnostic Learning System

This module provides truly domain-agnostic pattern learning that works with any text data
without hardcoded domain assumptions, language biases, or predetermined categorizations.
"""

import asyncio
import time
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
import math

from .domain_pattern_engine import PatternType, DiscoveredPattern
from .constants import StatisticalConfidenceCalculator

logger = logging.getLogger(__name__)


@dataclass
class UniversalPattern:
    """A pattern discovered without domain assumptions"""
    pattern_id: str
    pattern_text: str
    pattern_type: str  # Discovered dynamically, not predetermined
    frequency: int
    confidence: float
    semantic_cluster: Optional[str] = None
    contexts: List[str] = field(default_factory=list)
    statistical_metrics: Dict[str, float] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)


@dataclass
class UniversalDomainSchema:
    """Schema discovered without domain-specific assumptions"""
    domain_name: str
    discovered_patterns: List[UniversalPattern]
    pattern_clusters: Dict[str, List[str]]  # Dynamically discovered categories
    vocabulary_statistics: Dict[str, Any]
    semantic_relationships: List[Tuple[str, str, float]]
    confidence_distribution: Dict[str, float]
    total_documents_analyzed: int
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalPatternLearner:
    """
    Domain-agnostic pattern learning system that works with any text data.
    
    Removes biases:
    - No hardcoded domain assumptions
    - No language-specific processing 
    - No predetermined categories
    - No frequency-based assumptions
    - Dynamic schema generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Domain-agnostic parameters (can be data-driven)
        self.min_pattern_frequency = config.get("min_pattern_frequency", 2)
        self.confidence_threshold = config.get("confidence_threshold", 0.3)
        self.clustering_similarity_threshold = config.get("clustering_similarity_threshold", 0.7)
        
        # Universal linguistic patterns (not language-specific)
        self.universal_word_pattern = config.get(
            "word_pattern", 
            r'[\w\u00C0-\u017F\u0100-\u024F\u1E00-\u1EFF]+'  # Supports Unicode, accents, etc.
        )
        
        # No predefined categories - discovered dynamically
        self.discovered_patterns: Dict[str, UniversalPattern] = {}
        
    async def learn_universal_patterns(self, texts: List[str], domain_hint: Optional[str] = None) -> UniversalDomainSchema:
        """
        Learn patterns from any text data without domain assumptions.
        
        Args:
            texts: Raw text data from any domain
            domain_hint: Optional hint for domain name (doesn't affect learning)
            
        Returns:
            UniversalDomainSchema with discovered patterns
        """
        self.logger.info(f"Starting universal pattern learning from {len(texts)} texts")
        
        # Step 1: Extract all patterns without bias
        raw_patterns = await self._extract_universal_patterns(texts)
        
        # Step 2: Calculate statistical significance without frequency bias
        significant_patterns = await self._calculate_statistical_significance(raw_patterns, texts)
        
        # Step 3: Discover semantic clusters dynamically (no predetermined categories)
        pattern_clusters = await self._discover_semantic_clusters(significant_patterns)
        
        # Step 4: Generate universal schema without assumptions
        universal_schema = await self._generate_universal_schema(
            significant_patterns, pattern_clusters, texts, domain_hint
        )
        
        self.logger.info(f"Universal learning complete: {len(significant_patterns)} patterns, {len(pattern_clusters)} clusters")
        
        return universal_schema
    
    async def _extract_universal_patterns(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract patterns without language or domain bias"""
        pattern_data = defaultdict(lambda: {
            "frequency": 0,
            "contexts": [],
            "document_spread": set(),
            "positions": [],
            "co_occurrences": defaultdict(int)
        })
        
        for doc_idx, text in enumerate(texts):
            # Use Unicode-aware tokenization (not just English)
            tokens = re.findall(self.universal_word_pattern, text.lower())
            
            # Extract patterns with positional and contextual information
            for pos, token in enumerate(tokens):
                if len(token) >= 2:  # More inclusive than 4+ characters
                    pattern_data[token]["frequency"] += 1
                    pattern_data[token]["document_spread"].add(doc_idx)
                    pattern_data[token]["positions"].append(pos / len(tokens))  # Normalized position
                    
                    # Context window (no assumptions about meaningful context size)
                    context_start = max(0, pos - 3)
                    context_end = min(len(tokens), pos + 4)
                    context = " ".join(tokens[context_start:context_end])
                    pattern_data[token]["contexts"].append(context)
                    
                    # Co-occurrence analysis (semantic relationships)
                    for other_token in tokens[context_start:context_end]:
                        if other_token != token:
                            pattern_data[token]["co_occurrences"][other_token] += 1
        
        return dict(pattern_data)
    
    async def _calculate_statistical_significance(
        self, 
        raw_patterns: Dict[str, Dict[str, Any]], 
        texts: List[str]
    ) -> List[UniversalPattern]:
        """Calculate significance without frequency bias"""
        significant_patterns = []
        total_documents = len(texts)
        
        for pattern_text, data in raw_patterns.items():
            frequency = data["frequency"]
            document_spread = len(data["document_spread"])
            
            # Statistical metrics without bias
            document_frequency = document_spread / total_documents  # How widespread
            average_positions = statistics.mean(data["positions"]) if data["positions"] else 0.5
            position_variance = statistics.variance(data["positions"]) if len(data["positions"]) > 1 else 0
            
            # TF-IDF-like calculation without predetermined importance
            term_frequency = frequency / sum(len(re.findall(self.universal_word_pattern, text)) for text in texts)
            document_frequency_score = math.log(total_documents / (document_spread + 1))
            tf_idf_score = term_frequency * document_frequency_score
            
            # Multi-factor confidence (not just frequency)
            confidence_factors = [
                min(1.0, document_frequency * 2),  # Document spread
                min(1.0, tf_idf_score * 10),       # TF-IDF significance
                1.0 - position_variance,            # Positional consistency
                min(1.0, len(data["co_occurrences"]) / 10)  # Semantic richness
            ]
            
            confidence = statistics.mean(confidence_factors)
            
            # Include if statistically significant (not just frequent)
            if confidence >= self.confidence_threshold and document_spread >= 2:
                universal_pattern = UniversalPattern(
                    pattern_id=f"universal_{hash(pattern_text) % 100000}",
                    pattern_text=pattern_text,
                    pattern_type="semantic",  # Will be refined by clustering
                    frequency=frequency,
                    confidence=confidence,
                    contexts=data["contexts"][:10],  # Sample contexts
                    statistical_metrics={
                        "document_spread": document_frequency,
                        "tf_idf_score": tf_idf_score,
                        "position_variance": position_variance,
                        "co_occurrence_count": len(data["co_occurrences"]),
                        "semantic_density": sum(data["co_occurrences"].values()) / frequency
                    }
                )
                
                significant_patterns.append(universal_pattern)
        
        # Sort by confidence, not frequency
        significant_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return significant_patterns
    
    async def _discover_semantic_clusters(self, patterns: List[UniversalPattern]) -> Dict[str, List[str]]:
        """Discover pattern categories dynamically without predetermined types"""
        if len(patterns) < 3:
            return {"general": [p.pattern_text for p in patterns]}
        
        # Create pattern similarity matrix based on co-occurrence
        pattern_similarity = {}
        pattern_texts = [p.pattern_text for p in patterns]
        
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Calculate semantic similarity based on context overlap  
                contexts1 = set(" ".join(pattern1.contexts).split())
                contexts2 = set(" ".join(pattern2.contexts).split())
                
                if contexts1 and contexts2:
                    jaccard_similarity = len(contexts1 & contexts2) / len(contexts1 | contexts2)
                    pattern_similarity[(pattern1.pattern_text, pattern2.pattern_text)] = jaccard_similarity
        
        # Simple clustering algorithm (domain-agnostic)
        clusters = {}
        unassigned = set(pattern_texts)
        cluster_id = 0
        
        while unassigned:
            # Start new cluster with highest confidence unassigned pattern
            seed_pattern = max(unassigned, key=lambda p: next(pat.confidence for pat in patterns if pat.pattern_text == p))
            cluster_name = f"semantic_cluster_{cluster_id}"
            clusters[cluster_name] = [seed_pattern]
            unassigned.remove(seed_pattern)
            
            # Add similar patterns to cluster
            to_add = []
            for pattern in unassigned:
                similarity_key = tuple(sorted([seed_pattern, pattern]))
                similarity = pattern_similarity.get(similarity_key, 0)
                
                if similarity >= self.clustering_similarity_threshold:
                    to_add.append(pattern)
            
            for pattern in to_add:
                clusters[cluster_name].append(pattern)
                unassigned.remove(pattern)
            
            cluster_id += 1
            
            # Prevent infinite loops
            if cluster_id > len(pattern_texts):
                break
        
        # Add remaining patterns to their own clusters
        for pattern in unassigned:
            clusters[f"isolated_pattern_{cluster_id}"] = [pattern]
            cluster_id += 1
        
        return clusters
    
    async def _generate_universal_schema(
        self,
        patterns: List[UniversalPattern],
        clusters: Dict[str, List[str]],
        texts: List[str],
        domain_hint: Optional[str]
    ) -> UniversalDomainSchema:
        """Generate schema without domain-specific assumptions"""
        
        # Update pattern types based on discovered clusters
        cluster_to_patterns = {}
        for cluster_name, pattern_texts in clusters.items():
            for pattern_text in pattern_texts:
                for pattern in patterns:
                    if pattern.pattern_text == pattern_text:
                        pattern.pattern_type = cluster_name
                        pattern.semantic_cluster = cluster_name
                        break
        
        # Calculate vocabulary statistics
        all_words = []
        for text in texts:
            all_words.extend(re.findall(self.universal_word_pattern, text.lower()))
        
        vocab_stats = {
            "total_tokens": len(all_words),
            "unique_tokens": len(set(all_words)),
            "vocabulary_richness": len(set(all_words)) / len(all_words) if all_words else 0,
            "average_text_length": statistics.mean([len(text.split()) for text in texts]),
            "text_length_variance": statistics.variance([len(text.split()) for text in texts]) if len(texts) > 1 else 0
        }
        
        # Calculate confidence distribution
        confidences = [p.confidence for p in patterns]
        confidence_dist = {}
        if confidences:
            confidence_dist = {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0,
                "min": min(confidences),
                "max": max(confidences)
            }
        
        # Discover semantic relationships
        relationships = []
        for pattern in patterns[:20]:  # Top patterns only
            for other_pattern in patterns[:20]:
                if pattern != other_pattern and pattern.semantic_cluster == other_pattern.semantic_cluster:
                    # Calculate relationship strength
                    contexts1 = set(" ".join(pattern.contexts).split())
                    contexts2 = set(" ".join(other_pattern.contexts).split())
                    if contexts1 and contexts2:
                        strength = len(contexts1 & contexts2) / len(contexts1 | contexts2)
                        if strength > 0.1:
                            relationships.append((pattern.pattern_text, other_pattern.pattern_text, strength))
        
        # Auto-detect domain name if not provided
        if not domain_hint:
            # Use most frequent cluster as domain indicator
            largest_cluster = max(clusters.keys(), key=lambda k: len(clusters[k]))
            domain_hint = largest_cluster.replace("semantic_cluster_", "domain_")
        
        return UniversalDomainSchema(
            domain_name=domain_hint or "universal",
            discovered_patterns=patterns,
            pattern_clusters=clusters,
            vocabulary_statistics=vocab_stats,
            semantic_relationships=relationships,
            confidence_distribution=confidence_dist,
            total_documents_analyzed=len(texts),
            generation_metadata={
                "learning_method": "universal_statistical_analysis",
                "biases_removed": [
                    "domain_specific_assumptions",
                    "language_specific_processing", 
                    "frequency_bias",
                    "predetermined_categories",
                    "schema_assumptions"
                ],
                "universal_features": [
                    "unicode_support",
                    "statistical_significance",
                    "dynamic_clustering",
                    "multi_factor_confidence",
                    "semantic_relationship_discovery"
                ]
            }
        )
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of bias-free learning process"""
        return {
            "approach": "universal_statistical_learning",
            "biases_eliminated": [
                "hardcoded_domain_categories",
                "language_specific_tokenization",
                "frequency_only_confidence",
                "predetermined_schema_structure",
                "english_only_processing"
            ],
            "universal_features": [
                "unicode_pattern_support",
                "multi_factor_statistical_confidence", 
                "dynamic_semantic_clustering",
                "context_aware_relationship_discovery",
                "adaptive_schema_generation"
            ],
            "works_with": [
                "any_human_language",
                "technical_documents",
                "medical_texts",
                "legal_documents", 
                "financial_reports",
                "academic_papers",
                "social_media_content",
                "multilingual_datasets"
            ]
        }


# Factory function for easy initialization
async def create_universal_pattern_learner(config: Optional[Dict[str, Any]] = None) -> UniversalPatternLearner:
    """Create universal pattern learner without domain assumptions"""
    default_config = {
        "min_pattern_frequency": 2,
        "confidence_threshold": 0.3,
        "clustering_similarity_threshold": 0.7,
        "word_pattern": r'[\w\u00C0-\u017F\u0100-\u024F\u1E00-\u1EFF]+',  # Unicode support
    }
    
    if config:
        default_config.update(config)
    
    return UniversalPatternLearner(default_config)


__all__ = [
    'UniversalPatternLearner',
    'UniversalPattern', 
    'UniversalDomainSchema',
    'create_universal_pattern_learner'
]