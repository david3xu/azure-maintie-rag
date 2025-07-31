"""
Domain Pattern Engine - Core pattern detection and domain fingerprinting for zero-config adaptation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected in domain text"""
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


@dataclass
class DiscoveredPattern:
    """A discovered pattern with metadata"""
    pattern_id: str
    pattern_type: PatternType
    pattern_text: str
    confidence: float
    frequency: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)


@dataclass
class DomainFingerprint:
    """Unique fingerprint representing a domain's characteristics"""
    domain_id: str
    entity_patterns: Dict[str, float] = field(default_factory=dict)
    relationship_patterns: Dict[str, float] = field(default_factory=dict)
    semantic_clusters: List[Dict[str, Any]] = field(default_factory=list)
    temporal_patterns: List[Dict[str, Any]] = field(default_factory=list)
    vocabulary_distribution: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    sample_size: int = 0
    
    def calculate_similarity(self, other: 'DomainFingerprint') -> float:
        """Calculate similarity between two domain fingerprints"""
        similarities = []
        
        # Entity pattern similarity
        if self.entity_patterns and other.entity_patterns:
            common_entities = set(self.entity_patterns.keys()) & set(other.entity_patterns.keys())
            if common_entities:
                entity_sim = sum(
                    min(self.entity_patterns[e], other.entity_patterns[e]) 
                    for e in common_entities
                ) / len(common_entities)
                similarities.append(entity_sim)
        
        # Vocabulary similarity
        if self.vocabulary_distribution and other.vocabulary_distribution:
            common_vocab = set(self.vocabulary_distribution.keys()) & set(other.vocabulary_distribution.keys())
            if common_vocab:
                vocab_sim = sum(
                    min(self.vocabulary_distribution[w], other.vocabulary_distribution[w])
                    for w in common_vocab
                ) / len(common_vocab)
                similarities.append(vocab_sim)
        
        return statistics.mean(similarities) if similarities else 0.0


class DomainPatternEngine:
    """Core engine for detecting and analyzing domain patterns"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.pattern_cache: Dict[str, Any] = {}
        
    async def analyze_text_patterns(
        self, 
        text_corpus: List[str], 
        domain_name: Optional[str] = None
    ) -> DomainFingerprint:
        """Analyze text corpus to extract domain patterns"""
        
        if not text_corpus:
            return DomainFingerprint(
                domain_id=domain_name or "empty",
                confidence_score=0.0
            )
        
        # Extract entity patterns
        entity_patterns = await self._extract_entity_patterns(text_corpus)
        
        # Extract relationship patterns
        relationship_patterns = await self._extract_relationship_patterns(text_corpus)
        
        # Extract semantic clusters
        semantic_clusters = await self._extract_semantic_clusters(text_corpus)
        
        # Extract vocabulary distribution
        vocab_distribution = await self._extract_vocabulary_distribution(text_corpus)
        
        # Calculate confidence based on pattern strength
        confidence = self._calculate_pattern_confidence(
            entity_patterns, relationship_patterns, semantic_clusters
        )
        
        fingerprint = DomainFingerprint(
            domain_id=domain_name or f"auto_{hash(str(text_corpus[:3]))% 10000}",
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            semantic_clusters=semantic_clusters,
            vocabulary_distribution=vocab_distribution,
            confidence_score=confidence,
            sample_size=len(text_corpus)
        )
        
        return fingerprint
    
    async def _extract_entity_patterns(self, text_corpus: List[str]) -> Dict[str, float]:
        """Extract entity patterns from text corpus"""
        entity_counts = {}
        total_entities = 0
        
        for text in text_corpus:
            # Simple entity extraction - could be enhanced with NLP
            words = text.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():  # Basic entity filtering
                    if word.istitle() or word.isupper():  # Likely entities
                        entity_counts[word.lower()] = entity_counts.get(word.lower(), 0) + 1
                        total_entities += 1
        
        # Convert to frequencies
        if total_entities > 0:
            return {
                entity: count / total_entities 
                for entity, count in entity_counts.items()
                if count >= 2  # Minimum frequency threshold
            }
        
        return {}
    
    async def _extract_relationship_patterns(self, text_corpus: List[str]) -> Dict[str, float]:
        """Extract relationship patterns from text corpus"""
        relationship_counts = {}
        total_relationships = 0
        
        # Common relationship indicators
        relationship_words = [
            'has', 'is', 'contains', 'includes', 'requires', 'uses', 'manages',
            'controls', 'operates', 'maintains', 'produces', 'generates'
        ]
        
        for text in text_corpus:
            words = text.lower().split()
            for word in words:
                if word in relationship_words:
                    relationship_counts[word] = relationship_counts.get(word, 0) + 1
                    total_relationships += 1
        
        # Convert to frequencies
        if total_relationships > 0:
            return {
                rel: count / total_relationships 
                for rel, count in relationship_counts.items()
            }
        
        return {}
    
    async def _extract_semantic_clusters(self, text_corpus: List[str]) -> List[Dict[str, Any]]:
        """Extract semantic clusters from text corpus"""
        # Simple clustering based on word co-occurrence
        word_cooccurrence = {}
        
        for text in text_corpus:
            words = [w.lower() for w in text.split() if len(w) > 3 and w.isalpha()]
            
            # Track word co-occurrence
            for i, word1 in enumerate(words):
                for word2 in words[i+1:i+6]:  # Window of 5 words
                    pair = tuple(sorted([word1, word2]))
                    word_cooccurrence[pair] = word_cooccurrence.get(pair, 0) + 1
        
        # Create clusters from high co-occurrence pairs
        clusters = []
        for (word1, word2), count in word_cooccurrence.items():
            if count >= 2:  # Minimum co-occurrence threshold
                clusters.append({
                    "words": [word1, word2],
                    "strength": count,
                    "type": "cooccurrence"
                })
        
        return sorted(clusters, key=lambda x: x["strength"], reverse=True)[:20]
    
    async def _extract_vocabulary_distribution(self, text_corpus: List[str]) -> Dict[str, float]:
        """Extract vocabulary distribution from text corpus"""
        word_counts = {}
        total_words = 0
        
        for text in text_corpus:
            words = [w.lower() for w in text.split() if len(w) > 2 and w.isalpha()]
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                total_words += 1
        
        # Convert to frequencies, keep only significant words
        if total_words > 0:
            return {
                word: count / total_words 
                for word, count in word_counts.items()
                if count >= 2 and count / total_words >= 0.001  # Minimum frequency
            }
        
        return {}
    
    def _calculate_pattern_confidence(
        self, 
        entity_patterns: Dict[str, float],
        relationship_patterns: Dict[str, float], 
        semantic_clusters: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score based on pattern strength"""
        
        scores = []
        
        # Entity pattern strength
        if entity_patterns:
            entity_strength = sum(entity_patterns.values()) / len(entity_patterns)
            scores.append(min(entity_strength * 10, 1.0))  # Scale to 0-1
        
        # Relationship pattern strength
        if relationship_patterns:
            rel_strength = sum(relationship_patterns.values()) / len(relationship_patterns)
            scores.append(min(rel_strength * 20, 1.0))  # Scale to 0-1
        
        # Semantic cluster strength
        if semantic_clusters:
            cluster_strength = sum(c["strength"] for c in semantic_clusters[:5]) / 50
            scores.append(min(cluster_strength, 1.0))
        
        return statistics.mean(scores) if scores else 0.0
    
    async def compare_fingerprints(
        self, 
        fingerprint1: DomainFingerprint, 
        fingerprint2: DomainFingerprint
    ) -> float:
        """Compare two domain fingerprints for similarity"""
        return fingerprint1.calculate_similarity(fingerprint2)
    
    async def find_similar_domains(
        self, 
        target_fingerprint: DomainFingerprint,
        known_fingerprints: List[DomainFingerprint],
        threshold: float = 0.7
    ) -> List[Tuple[DomainFingerprint, float]]:
        """Find similar domains based on fingerprint comparison"""
        
        similar_domains = []
        
        for fingerprint in known_fingerprints:
            similarity = await self.compare_fingerprints(target_fingerprint, fingerprint)
            if similarity >= threshold:
                similar_domains.append((fingerprint, similarity))
        
        # Sort by similarity (highest first)
        return sorted(similar_domains, key=lambda x: x[1], reverse=True)