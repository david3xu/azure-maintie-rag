"""
Pattern Engine - Consolidated Pattern Extraction and Learning

This module consolidates pattern extraction and learning functionality from
multiple directories (discovery/pattern_*.py and domain/pattern_extractor.py)
into a unified, high-performance pattern engine that maintains all competitive
advantages while simplifying the architecture.

Key features preserved:
- Data-driven pattern learning (no hardcoded assumptions)
- Statistical pattern extraction and evolution tracking
- Dynamic domain adaptation based on learned patterns
- High-performance pattern matching and indexing
- Continuous learning from user interactions
"""

import hashlib
import json
import logging
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configuration imports
from config.centralized_config import get_pattern_engine_config

logger = logging.getLogger(__name__)


@dataclass
class LearnedPattern:
    """A pattern learned from data with confidence and usage tracking"""

    pattern_id: str
    pattern_text: str
    pattern_type: str  # entity, action, relationship, temporal, semantic
    confidence: float
    frequency: int
    domains: List[str]
    learned_from: List[str]  # Source documents/contexts
    first_seen: float
    last_updated: float
    usage_count: int = 0

    def update_usage(self):
        """Update usage statistics"""
        self.usage_count += 1
        self.last_updated = time.time()

    def calculate_relevance_score(self, domain: str = None, config=None) -> float:
        """Calculate relevance score for pattern"""
        if config is None:
            from config.centralized_config import get_pattern_engine_config
            config = get_pattern_engine_config()
            
        base_score = self.confidence * (1 + self.frequency / config.frequency_boost_divisor)

        if domain and domain in self.domains:
            base_score *= config.domain_match_boost

        # Age penalty (patterns lose relevance over time without updates)
        age_days = (time.time() - self.last_updated) / (24 * 3600)
        age_factor = max(config.age_factor_min, 1.0 - (age_days / config.pattern_age_half_life_days))

        return base_score * age_factor

    def is_high_confidence(self) -> bool:
        """Check if pattern has high confidence (>0.7 and frequency >2)"""
        return self.confidence > 0.7 and self.frequency > 2


@dataclass
class ExtractedPatterns:
    """Collection of patterns extracted from content"""

    entity_patterns: List[LearnedPattern]
    action_patterns: List[LearnedPattern]
    relationship_patterns: List[LearnedPattern]
    temporal_patterns: List[LearnedPattern]
    concept_patterns: List[LearnedPattern] = field(default_factory=list)
    semantic_clusters: List[Dict[str, Any]] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    source_word_count: int = 0
    processing_time: float = 0.0
    extraction_confidence: float = 0.8  # Overall extraction confidence score

    def get_top_patterns(
        self, pattern_type: str, limit: int = 10
    ) -> List[LearnedPattern]:
        """Get top patterns of specified type by relevance"""
        pattern_list = getattr(self, f"{pattern_type}_patterns", [])
        return sorted(
            pattern_list, key=lambda p: p.calculate_relevance_score(), reverse=True
        )[:limit]

    def get_all_patterns(self) -> List[LearnedPattern]:
        """Get all patterns regardless of type"""
        all_patterns = []
        all_patterns.extend(self.entity_patterns)
        all_patterns.extend(self.action_patterns)
        all_patterns.extend(self.relationship_patterns)
        all_patterns.extend(self.temporal_patterns)
        return all_patterns


class PatternEngine:
    """
    Unified pattern engine consolidating all pattern-related functionality.

    This engine combines:
    - Pattern extraction from raw content
    - Statistical pattern learning and evolution
    - Dynamic pattern matching and indexing
    - Continuous learning from user interactions
    - Domain-specific pattern adaptation
    """

    def __init__(self, config: Dict[str, Any] = None, cache_dir: Optional[Path] = None):
        self.config = config or {}
        self.cache_dir = (
            cache_dir or Path(__file__).parent.parent.parent / "cache" / "patterns"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get centralized configuration
        self.pattern_config = get_pattern_engine_config()

        # Pattern storage and indexing
        self.learned_patterns: Dict[str, LearnedPattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(
            set
        )  # word -> pattern_ids
        self.domain_patterns: Dict[str, Set[str]] = defaultdict(
            set
        )  # domain -> pattern_ids

        # Learning configuration (from centralized config)
        self.min_pattern_frequency = self.pattern_config.min_pattern_frequency
        self.min_confidence_threshold = self.pattern_config.min_confidence_threshold
        self.max_patterns_per_type = self.pattern_config.max_patterns_per_type

        # Pattern extraction rules (learned from data, not hardcoded)
        self.entity_extractors = self._initialize_entity_extractors()
        self.action_extractors = self._initialize_action_extractors()
        self.relationship_extractors = self._initialize_relationship_extractors()

        # Performance tracking
        self.stats = {
            "patterns_learned": 0,
            "patterns_applied": 0,
            "total_extractions": 0,
            "avg_extraction_time": 0.0,
            "learning_accuracy": 0.0,
        }

        # Load existing patterns
        self._load_learned_patterns()

        logger.info(
            f"Pattern engine initialized: {len(self.learned_patterns)} patterns loaded"
        )

    def extract_domain_patterns(
        self, domain: str, content_analysis: Any, confidence: float
    ) -> ExtractedPatterns:
        """
        Extract patterns from domain content using statistical analysis.

        This method consolidates the functionality from multiple pattern extractors
        into a single, efficient extraction process.
        """
        start_time = time.time()

        try:
            # Get content for analysis
            text_content = self._get_text_from_analysis(content_analysis)

            # Extract different types of patterns
            entity_patterns = self._extract_entity_patterns(
                text_content, domain, confidence
            )
            action_patterns = self._extract_action_patterns(
                text_content, domain, confidence
            )
            relationship_patterns = self._extract_relationship_patterns(
                text_content, domain, confidence
            )
            temporal_patterns = self._extract_temporal_patterns(
                text_content, domain, confidence
            )

            # Create semantic clusters
            semantic_clusters = self._create_semantic_clusters(
                entity_patterns + action_patterns, domain
            )

            # Update learned patterns
            all_new_patterns = (
                entity_patterns
                + action_patterns
                + relationship_patterns
                + temporal_patterns
            )
            self._update_learned_patterns(all_new_patterns, domain)

            processing_time = time.time() - start_time

            # Update statistics
            self.stats["total_extractions"] += 1
            self.stats["avg_extraction_time"] = (
                self.stats["avg_extraction_time"]
                * (self.stats["total_extractions"] - 1)
                + processing_time
            ) / self.stats["total_extractions"]

            return ExtractedPatterns(
                entity_patterns=entity_patterns,
                action_patterns=action_patterns,
                relationship_patterns=relationship_patterns,
                temporal_patterns=temporal_patterns,
                semantic_clusters=semantic_clusters,
                extraction_metadata={
                    "domain": domain,
                    "extraction_method": "statistical_analysis",
                    "confidence": confidence,
                    "patterns_found": len(all_new_patterns),
                },
                source_word_count=len(text_content.split()) if text_content else 0,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Pattern extraction failed for domain {domain}: {e}")
            # Return empty patterns on failure
            return ExtractedPatterns(
                entity_patterns=[],
                action_patterns=[],
                relationship_patterns=[],
                temporal_patterns=[],
                semantic_clusters=[],
                extraction_metadata={"error": str(e)},
                source_word_count=0,
                processing_time=time.time() - start_time,
            )

    def _get_text_from_analysis(self, content_analysis: Any) -> str:
        """Extract text content from analysis object"""
        if hasattr(content_analysis, "raw_text"):
            return content_analysis.raw_text
        elif hasattr(content_analysis, "content"):
            return content_analysis.content
        elif hasattr(content_analysis, "text"):
            return content_analysis.text
        else:
            # Fallback: construct text from available analysis data
            text_parts = []
            if hasattr(content_analysis, "concept_frequency"):
                text_parts.extend(content_analysis.concept_frequency.keys())
            if hasattr(content_analysis, "entity_candidates"):
                text_parts.extend(content_analysis.entity_candidates)
            return " ".join(text_parts)

    def _extract_entity_patterns(
        self, text: str, domain: str, confidence: float
    ) -> List[LearnedPattern]:
        """Extract entity patterns using statistical analysis"""
        entities = []
        current_time = time.time()

        for extractor_name, extractor in self.entity_extractors.items():
            try:
                matches = extractor.findall(text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)

                    if len(match.strip()) > self.pattern_config.min_pattern_length_entities:
                        pattern_id = self._generate_pattern_id(match, "entity")

                        entities.append(
                            LearnedPattern(
                                pattern_id=pattern_id,
                                pattern_text=match.strip(),
                                pattern_type="entity",
                                confidence=confidence * self.pattern_config.entity_pattern_confidence_multiplier,
                                frequency=text.lower().count(match.lower()),
                                domains=[domain],
                                learned_from=[f"extraction_{extractor_name}"],
                                first_seen=current_time,
                                last_updated=current_time,
                            )
                        )
            except Exception as e:
                logger.debug(f"Entity extractor {extractor_name} failed: {e}")

        # Apply statistical filtering
        return self._filter_patterns_by_statistics(entities)[
            : self.max_patterns_per_type
        ]

    def _extract_action_patterns(
        self, text: str, domain: str, confidence: float
    ) -> List[LearnedPattern]:
        """Extract action patterns using statistical analysis"""
        actions = []
        current_time = time.time()

        for extractor_name, extractor in self.action_extractors.items():
            try:
                matches = extractor.findall(text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)

                    if len(match.strip()) > self.pattern_config.min_pattern_length_entities:
                        pattern_id = self._generate_pattern_id(match, "action")

                        actions.append(
                            LearnedPattern(
                                pattern_id=pattern_id,
                                pattern_text=match.strip(),
                                pattern_type="action",
                                confidence=confidence * self.pattern_config.action_pattern_confidence_multiplier,
                                frequency=text.lower().count(match.lower()),
                                domains=[domain],
                                learned_from=[f"extraction_{extractor_name}"],
                                first_seen=current_time,
                                last_updated=current_time,
                            )
                        )
            except Exception as e:
                logger.debug(f"Action extractor {extractor_name} failed: {e}")

        return self._filter_patterns_by_statistics(actions)[
            : self.max_patterns_per_type
        ]

    def _extract_relationship_patterns(
        self, text: str, domain: str, confidence: float
    ) -> List[LearnedPattern]:
        """Extract relationship patterns using statistical analysis"""
        relationships = []
        current_time = time.time()

        for extractor_name, extractor in self.relationship_extractors.items():
            try:
                matches = extractor.findall(text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)

                    if len(match.strip()) > self.pattern_config.min_pattern_length_relationships:
                        pattern_id = self._generate_pattern_id(match, "relationship")

                        relationships.append(
                            LearnedPattern(
                                pattern_id=pattern_id,
                                pattern_text=match.strip(),
                                pattern_type="relationship",
                                confidence=confidence * self.pattern_config.relationship_pattern_confidence_multiplier,
                                frequency=text.lower().count(match.lower()),
                                domains=[domain],
                                learned_from=[f"extraction_{extractor_name}"],
                                first_seen=current_time,
                                last_updated=current_time,
                            )
                        )
            except Exception as e:
                logger.debug(f"Relationship extractor {extractor_name} failed: {e}")

        return self._filter_patterns_by_statistics(relationships)[
            : self.max_patterns_per_type
        ]

    def _extract_temporal_patterns(
        self, text: str, domain: str, confidence: float
    ) -> List[LearnedPattern]:
        """Extract temporal patterns using statistical analysis"""
        temporal_patterns = []
        current_time = time.time()

        # Temporal pattern extractors
        temporal_extractors = {
            "time_expressions": re.compile(
                self.pattern_config.time_expressions_pattern,
                re.IGNORECASE,
            ),
            "sequences": re.compile(
                self.pattern_config.sequences_pattern, re.IGNORECASE
            ),
            "durations": re.compile(
                self.pattern_config.durations_pattern,
                re.IGNORECASE,
            ),
        }

        for extractor_name, extractor in temporal_extractors.items():
            try:
                matches = extractor.findall(text)
                for match in matches:
                    if len(match.strip()) > self.pattern_config.min_pattern_length_temporal:
                        pattern_id = self._generate_pattern_id(match, "temporal")

                        temporal_patterns.append(
                            LearnedPattern(
                                pattern_id=pattern_id,
                                pattern_text=match.strip(),
                                pattern_type="temporal",
                                confidence=confidence * self.pattern_config.temporal_pattern_confidence_multiplier,
                                frequency=text.lower().count(match.lower()),
                                domains=[domain],
                                learned_from=[f"extraction_{extractor_name}"],
                                first_seen=current_time,
                                last_updated=current_time,
                            )
                        )
            except Exception as e:
                logger.debug(f"Temporal extractor {extractor_name} failed: {e}")

        return self._filter_patterns_by_statistics(temporal_patterns)[
            : self.max_patterns_per_type
        ]

    def _create_semantic_clusters(
        self, patterns: List[LearnedPattern], domain: str
    ) -> List[Dict[str, Any]]:
        """Create semantic clusters from patterns"""
        if not patterns:
            return []

        # Simple clustering based on word overlap
        clusters = []
        processed_patterns = set()

        for pattern in patterns:
            if pattern.pattern_id in processed_patterns:
                continue

            # Create new cluster
            cluster = {
                "cluster_id": f"cluster_{len(clusters)}_{domain}",
                "domain": domain,
                "central_pattern": pattern.pattern_text,
                "patterns": [pattern.pattern_text],
                "confidence": pattern.confidence,
                "pattern_count": 1,
            }

            # Find similar patterns
            pattern_words = set(pattern.pattern_text.lower().split())

            for other_pattern in patterns:
                if (
                    other_pattern.pattern_id != pattern.pattern_id
                    and other_pattern.pattern_id not in processed_patterns
                ):
                    other_words = set(other_pattern.pattern_text.lower().split())
                    overlap = len(pattern_words & other_words) / len(
                        pattern_words | other_words
                    )

                    if overlap > self.pattern_config.word_overlap_threshold:
                        cluster["patterns"].append(other_pattern.pattern_text)
                        cluster["confidence"] = max(
                            cluster["confidence"], other_pattern.confidence
                        )
                        cluster["pattern_count"] += 1
                        processed_patterns.add(other_pattern.pattern_id)

            clusters.append(cluster)
            processed_patterns.add(pattern.pattern_id)

        # Sort clusters by pattern count and confidence
        clusters.sort(key=lambda c: (c["pattern_count"], c["confidence"]), reverse=True)

        return clusters[:self.pattern_config.max_clusters_returned]

    def _filter_patterns_by_statistics(
        self, patterns: List[LearnedPattern]
    ) -> List[LearnedPattern]:
        """Filter patterns based on statistical significance"""
        if not patterns:
            return []

        # Filter by minimum frequency and confidence
        filtered_patterns = [
            p
            for p in patterns
            if p.frequency >= self.min_pattern_frequency
            and p.confidence >= self.min_confidence_threshold
        ]

        # Remove duplicates (same pattern text)
        seen_texts = set()
        unique_patterns = []

        for pattern in filtered_patterns:
            normalized_text = pattern.pattern_text.lower().strip()
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_patterns.append(pattern)

        # Sort by relevance score
        unique_patterns.sort(key=lambda p: p.calculate_relevance_score(), reverse=True)

        return unique_patterns

    def _update_learned_patterns(self, new_patterns: List[LearnedPattern], domain: str):
        """Update the learned pattern database with new patterns"""
        for pattern in new_patterns:
            if pattern.pattern_id in self.learned_patterns:
                # Update existing pattern
                existing = self.learned_patterns[pattern.pattern_id]
                existing.frequency += pattern.frequency
                existing.confidence = max(existing.confidence, pattern.confidence)
                existing.last_updated = time.time()

                if domain not in existing.domains:
                    existing.domains.append(domain)
            else:
                # Add new pattern
                self.learned_patterns[pattern.pattern_id] = pattern
                self.stats["patterns_learned"] += 1

                # Update indexes
                for word in pattern.pattern_text.lower().split():
                    if len(word) > 2:
                        self.pattern_index[word].add(pattern.pattern_id)

                self.domain_patterns[domain].add(pattern.pattern_id)

        # Persist patterns
        self._save_learned_patterns()

    def _generate_pattern_id(self, pattern_text: str, pattern_type: str) -> str:
        """Generate unique pattern ID"""
        content = f"{pattern_type}:{pattern_text.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _initialize_entity_extractors(self) -> Dict[str, re.Pattern]:
        """Initialize entity extraction patterns"""
        return {
            "technical_terms": re.compile(self.pattern_config.technical_terms_pattern),
            "model_names": re.compile(
                self.pattern_config.model_names_pattern,
                re.IGNORECASE,
            ),
            "identifiers": re.compile(self.pattern_config.identifiers_pattern),
            "measurements": re.compile(
                self.pattern_config.measurements_pattern,
                re.IGNORECASE,
            ),
            "codes": re.compile(self.pattern_config.codes_pattern),
            "proper_nouns": re.compile(self.pattern_config.proper_nouns_pattern),
        }

    def _initialize_action_extractors(self) -> Dict[str, re.Pattern]:
        """Initialize action extraction patterns"""
        return {
            "instructions": re.compile(
                self.pattern_config.instructions_pattern,
                re.IGNORECASE,
            ),
            "operations": re.compile(
                self.pattern_config.operations_pattern,
                re.IGNORECASE,
            ),
            "maintenance": re.compile(
                self.pattern_config.maintenance_pattern,
                re.IGNORECASE,
            ),
            "monitoring": re.compile(
                self.pattern_config.monitoring_pattern, re.IGNORECASE
            ),
            "management": re.compile(
                self.pattern_config.management_pattern,
                re.IGNORECASE,
            ),
        }

    def _initialize_relationship_extractors(self) -> Dict[str, re.Pattern]:
        """Initialize relationship extraction patterns"""
        return {
            "causation": re.compile(
                self.pattern_config.causation_pattern,
                re.IGNORECASE,
            ),
            "dependency": re.compile(
                self.pattern_config.dependency_pattern,
                re.IGNORECASE,
            ),
            "composition": re.compile(
                self.pattern_config.composition_pattern,
                re.IGNORECASE,
            ),
            "association": re.compile(
                self.pattern_config.association_pattern,
                re.IGNORECASE,
            ),
        }

    def find_matching_patterns(
        self, query: str, domain: Optional[str] = None, limit: int = 20
    ) -> List[LearnedPattern]:
        """Find patterns matching a query with optional domain filtering"""
        query_words = set(word.lower() for word in query.split() if len(word) > self.pattern_config.min_word_length_for_matching)
        pattern_scores = defaultdict(float)

        # Score patterns based on word matches
        for word in query_words:
            if word in self.pattern_index:
                for pattern_id in self.pattern_index[word]:
                    pattern = self.learned_patterns.get(pattern_id)
                    if pattern:
                        # Calculate match score
                        pattern_words = set(pattern.pattern_text.lower().split())
                        word_overlap = len(query_words & pattern_words) / len(
                            query_words | pattern_words
                        )

                        # Base score from word overlap and pattern relevance
                        score = word_overlap * pattern.calculate_relevance_score(domain)

                        # Boost domain-specific patterns
                        if domain and domain in pattern.domains:
                            score *= self.pattern_config.domain_pattern_score_boost

                        pattern_scores[pattern_id] = max(
                            pattern_scores[pattern_id], score
                        )

        # Sort by score and return top matches
        sorted_patterns = sorted(
            [(pid, score) for pid, score in pattern_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply usage tracking
        matching_patterns = []
        for pattern_id, score in sorted_patterns[:limit]:
            pattern = self.learned_patterns[pattern_id]
            pattern.update_usage()
            matching_patterns.append(pattern)
            self.stats["patterns_applied"] += 1

        return matching_patterns

    def get_domain_patterns(
        self, domain: str, pattern_type: Optional[str] = None, limit: int = 50
    ) -> List[LearnedPattern]:
        """Get patterns for a specific domain"""
        if domain not in self.domain_patterns:
            return []

        domain_pattern_list = []
        for pattern_id in self.domain_patterns[domain]:
            pattern = self.learned_patterns.get(pattern_id)
            if pattern and (not pattern_type or pattern.pattern_type == pattern_type):
                domain_pattern_list.append(pattern)

        # Sort by relevance
        domain_pattern_list.sort(
            key=lambda p: p.calculate_relevance_score(domain), reverse=True
        )

        return domain_pattern_list[:limit]

    def _save_learned_patterns(self):
        """Save learned patterns to persistent storage"""
        try:
            patterns_file = self.cache_dir / "learned_patterns.json"

            # Convert patterns to serializable format
            serializable_patterns = {}
            for pattern_id, pattern in self.learned_patterns.items():
                serializable_patterns[pattern_id] = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_text": pattern.pattern_text,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "frequency": pattern.frequency,
                    "domains": pattern.domains,
                    "learned_from": pattern.learned_from,
                    "first_seen": pattern.first_seen,
                    "last_updated": pattern.last_updated,
                    "usage_count": pattern.usage_count,
                }

            with open(patterns_file, "w", encoding="utf-8") as f:
                json.dump(serializable_patterns, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save learned patterns: {e}")

    def _load_learned_patterns(self):
        """Load learned patterns from persistent storage"""
        try:
            patterns_file = self.cache_dir / "learned_patterns.json"

            if patterns_file.exists():
                with open(patterns_file, "r", encoding="utf-8") as f:
                    serializable_patterns = json.load(f)

                # Convert back to LearnedPattern objects
                for pattern_id, pattern_data in serializable_patterns.items():
                    pattern = LearnedPattern(**pattern_data)
                    self.learned_patterns[pattern_id] = pattern

                    # Rebuild indexes
                    for word in pattern.pattern_text.lower().split():
                        if len(word) > 2:
                            self.pattern_index[word].add(pattern_id)

                    for domain in pattern.domains:
                        self.domain_patterns[domain].add(pattern_id)

                logger.info(
                    f"Loaded {len(self.learned_patterns)} learned patterns from cache"
                )

        except Exception as e:
            logger.warning(f"Failed to load learned patterns: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get pattern engine statistics"""
        return {
            "learned_patterns": {
                "total_patterns": len(self.learned_patterns),
                "by_type": {
                    pattern_type: len(
                        [
                            p
                            for p in self.learned_patterns.values()
                            if p.pattern_type == pattern_type
                        ]
                    )
                    for pattern_type in ["entity", "action", "relationship", "temporal"]
                },
                "by_domain": {
                    domain: len(pattern_ids)
                    for domain, pattern_ids in self.domain_patterns.items()
                },
            },
            "performance_stats": self.stats,
            "index_stats": {
                "indexed_words": len(self.pattern_index),
                "avg_patterns_per_word": statistics.mean(
                    [len(pattern_ids) for pattern_ids in self.pattern_index.values()]
                )
                if self.pattern_index
                else 0,
            },
            "configuration": {
                "min_pattern_frequency": self.min_pattern_frequency,
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_patterns_per_type": self.max_patterns_per_type,
            },
        }
