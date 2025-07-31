"""
Pattern Learning System - Advanced semantic pattern extraction and continuous learning.
Implements sophisticated learning algorithms for domain pattern discovery and evolution.
"""

import asyncio
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
import json
import statistics
import numpy as np

from .domain_pattern_engine import PatternType, DiscoveredPattern
from .constants import (
    StatisticalConfidenceCalculator,
    SearchWeightCalculator, 
    DataDrivenConfigurationManager,
    create_data_driven_configuration
)

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for pattern extraction"""
    SUPERVISED = "supervised"        # Learning with labeled examples
    UNSUPERVISED = "unsupervised"   # Learning without labels
    SEMI_SUPERVISED = "semi_supervised"  # Learning with some labels
    REINFORCEMENT = "reinforcement"  # Learning from feedback


class PatternEvolution(Enum):
    """Types of pattern evolution"""
    EMERGENCE = "emergence"          # New pattern discovered
    STRENGTHENING = "strengthening"  # Pattern confidence increased
    WEAKENING = "weakening"         # Pattern confidence decreased
    SPECIALIZATION = "specialization"  # Pattern became more specific
    GENERALIZATION = "generalization"  # Pattern became more general
    OBSOLESCENCE = "obsolescence"   # Pattern is no longer relevant


@dataclass
class LearningExample:
    """Example for pattern learning"""
    example_id: str
    text: str
    labels: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternEvolutionEvent:
    """Event representing pattern evolution"""
    event_id: str
    pattern_id: str
    evolution_type: PatternEvolution
    old_pattern: Optional[DiscoveredPattern]
    new_pattern: DiscoveredPattern
    confidence_change: float
    evidence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCluster:
    """Cluster of semantically related patterns"""
    cluster_id: str
    cluster_label: str
    patterns: List[str]  # Pattern IDs
    centroid_features: Dict[str, float]
    coherence_score: float
    size: int
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_pattern(self, pattern_id: str) -> None:
        """Add pattern to cluster"""
        if pattern_id not in self.patterns:
            self.patterns.append(pattern_id)
            self.size = len(self.patterns)
            self.updated_at = time.time()


@dataclass
class LearningSession:
    """Session for pattern learning"""
    session_id: str
    learning_mode: LearningMode
    start_time: float
    examples_processed: int = 0
    patterns_learned: int = 0
    patterns_evolved: int = 0
    clusters_created: int = 0
    avg_confidence_improvement: float = 0.0
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternLearningSystem:
    """
    Advanced pattern learning system for semantic extraction and continuous improvement.
    
    Implements sophisticated learning algorithms:
    - Unsupervised pattern discovery from text
    - Supervised learning from labeled examples
    - Reinforcement learning from user feedback
    - Pattern evolution tracking and optimization
    - Semantic clustering and organization
    - Continuous learning and adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pattern learning system.
        
        Args:
            config: Learning system configuration
                - learning_modes: Enabled learning modes
                - pattern_evolution_tracking: Enable pattern evolution tracking
                - semantic_clustering: Enable semantic clustering
                - confidence_learning_rate: Learning rate for confidence updates
                - cluster_similarity_threshold: Threshold for cluster formation
                - pattern_decay_rate: Rate of pattern confidence decay
                - max_learning_examples: Maximum examples to store
        """
        self.config = config
        self.enabled_learning_modes = [
            LearningMode(mode) for mode in config.get("learning_modes", ["unsupervised", "reinforcement"])
        ]
        self.pattern_evolution_tracking = config.get("pattern_evolution_tracking", True)
        self.semantic_clustering = config.get("semantic_clustering", True)
        # Initialize data-driven learning configuration
        self.learning_config = self._initialize_learning_configuration(config)
        
        self.confidence_learning_rate = config.get("confidence_learning_rate", 
            self.learning_config.get("confidence_adjustment", 0.1))  # Bootstrap fallback
        self.cluster_similarity_threshold = config.get("cluster_similarity_threshold", 
            self.learning_config.get("similarity_threshold", 0.7))  # Bootstrap fallback
        self.pattern_decay_rate = config.get("pattern_decay_rate", 
            self.learning_config.get("pattern_decay_rate", 0.05))  # Bootstrap fallback
        self.max_learning_examples = config.get("max_learning_examples", 
            self.learning_config.get("max_learning_examples", 1000))  # Bootstrap fallback
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Learning state
        self.learned_patterns: Dict[str, DiscoveredPattern] = {}
        self.pattern_evolution_history: List[PatternEvolutionEvent] = []
        self.semantic_clusters: Dict[str, SemanticCluster] = {}
        self.learning_examples: List[LearningExample] = []
        
        # Learning sessions
        self.active_sessions: Dict[str, LearningSession] = {}
        self.completed_sessions: List[LearningSession] = []
        
        # Pattern features for clustering
        self.pattern_features: Dict[str, Dict[str, float]] = {}
        
        # Performance metrics
        self.metrics = {
            "learning_sessions": 0,
            "patterns_learned": 0,
            "patterns_evolved": 0,
            "examples_processed": 0,
            "clusters_created": 0,
            "avg_learning_time": 0.0,
            "confidence_improvements": 0,
            "pattern_discoveries": 0
        }
    
    def _initialize_learning_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize data-driven learning configuration.
        
        This method replaces hardcoded learning parameters with values derived
        from actual learning performance data.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Data-driven learning configuration
        """
        learning_performance_data = config.get("learning_performance_data")
        
        if learning_performance_data:
            # Use real performance data
            logger.info("Creating data-driven learning configuration from performance data")
            config_manager = DataDrivenConfigurationManager()
            return config_manager.calculate_learning_parameters(learning_performance_data)
        else:
            # Bootstrap fallback with documentation
            logger.warning(
                "No learning performance data provided - using bootstrap values. "
                "For production, provide learning_performance_data to enable data-driven learning parameters."
            )
            
            return {
                "configuration_type": "bootstrap_learning",
                "created_from_real_data": False,
                "confidence_adjustment": 0.1,      # Bootstrap: TODO replace with performance data
                "similarity_threshold": 0.7,      # Bootstrap: TODO replace with performance data
                "pattern_decay_rate": 0.05,       # Bootstrap: TODO replace with performance data
                "max_learning_examples": 1000,    # Bootstrap: TODO replace with performance data
                "positive_feedback_multiplier": 1.2,  # Bootstrap: TODO replace with performance data
                "pattern_evolution_threshold": 0.1,   # Bootstrap: TODO replace with performance data
                "confidence_change_threshold": 0.05,  # Bootstrap: TODO replace with performance data
                "analysis_window_hours": 24.0,        # Bootstrap: TODO replace with performance data
                "requires_real_data": True
            }
    
    async def start_learning_session(
        self,
        learning_mode: LearningMode,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new pattern learning session.
        
        Args:
            learning_mode: Mode for this learning session
            session_metadata: Optional metadata for the session
            
        Returns:
            Session ID for the started session
        """
        if learning_mode not in self.enabled_learning_modes:
            raise ValueError(f"Learning mode {learning_mode.value} not enabled")
        
        session_id = f"session_{int(time.time())}_{learning_mode.value}"
        
        session = LearningSession(
            session_id=session_id,
            learning_mode=learning_mode,
            start_time=time.time(),
            metadata=session_metadata or {}
        )
        
        self.active_sessions[session_id] = session
        
        self.logger.info(f"Started learning session {session_id} in {learning_mode.value} mode")
        
        return session_id
    
    async def learn_patterns_from_examples(
        self,
        session_id: str,
        examples: List[LearningExample]
    ) -> Dict[str, Any]:
        """
        Learn patterns from provided examples.
        
        Args:
            session_id: Active learning session ID
            examples: Learning examples to process
            
        Returns:
            Learning results and statistics
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Learning session {session_id} not found")
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        # Add examples to learning history
        self.learning_examples.extend(examples)
        
        # Limit example storage
        if len(self.learning_examples) > self.max_learning_examples:
            self.learning_examples = self.learning_examples[-self.max_learning_examples:]
        
        # Process examples based on learning mode
        learning_results = {}
        
        if session.learning_mode == LearningMode.UNSUPERVISED:
            learning_results = await self._unsupervised_pattern_learning(examples)
        elif session.learning_mode == LearningMode.SUPERVISED:
            learning_results = await self._supervised_pattern_learning(examples)
        elif session.learning_mode == LearningMode.SEMI_SUPERVISED:
            learning_results = await self._semi_supervised_pattern_learning(examples)
        elif session.learning_mode == LearningMode.REINFORCEMENT:
            learning_results = await self._reinforcement_pattern_learning(examples)
        
        # Update session statistics
        session.examples_processed += len(examples)
        session.patterns_learned += learning_results.get("new_patterns", 0)
        session.patterns_evolved += learning_results.get("evolved_patterns", 0)
        session.clusters_created += learning_results.get("new_clusters", 0)
        
        # Calculate confidence improvement
        confidence_improvements = learning_results.get("confidence_improvements", [])
        if confidence_improvements:
            session.avg_confidence_improvement = statistics.mean(confidence_improvements)
        
        # Update global metrics
        self.metrics["examples_processed"] += len(examples)
        self.metrics["patterns_learned"] += learning_results.get("new_patterns", 0)
        self.metrics["patterns_evolved"] += learning_results.get("evolved_patterns", 0)
        self.metrics["clusters_created"] += learning_results.get("new_clusters", 0)
        
        learning_time = time.time() - start_time
        
        self.logger.info(
            f"Learning session {session_id}: processed {len(examples)} examples, "
            f"learned {learning_results.get('new_patterns', 0)} patterns in {learning_time:.2f}s"
        )
        
        return {
            "session_id": session_id,
            "learning_time_seconds": learning_time,
            **learning_results
        }
    
    async def apply_feedback_learning(
        self,
        pattern_id: str,
        feedback: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply feedback to improve pattern learning.
        
        Args:
            pattern_id: ID of pattern to apply feedback to
            feedback: Feedback information (success, confidence, corrections)
            context: Optional context for the feedback
            
        Returns:
            Results of feedback application
        """
        if pattern_id not in self.learned_patterns:
            self.logger.warning(f"Pattern {pattern_id} not found for feedback application")
            return {"success": False, "reason": "pattern_not_found"}
        
        pattern = self.learned_patterns[pattern_id]
        old_confidence = pattern.confidence
        
        # Extract feedback components
        success = feedback.get("success", True)
        user_confidence = feedback.get("confidence")
        corrections = feedback.get("corrections", {})
        
        # Apply feedback-based learning
        confidence_adjustment = 0.0
        
        if success:
            # Positive feedback - increase confidence
            positive_feedback_multiplier = self.learning_config.get("positive_feedback_multiplier", 1.2)
            confidence_adjustment = self.confidence_learning_rate * (positive_feedback_multiplier - pattern.confidence)
        else:
            # Negative feedback - decrease confidence
            confidence_adjustment = -self.confidence_learning_rate * pattern.confidence
        
        # Apply user confidence if provided
        if user_confidence is not None:
            target_confidence = float(user_confidence)
            confidence_adjustment = self.confidence_learning_rate * (target_confidence - pattern.confidence)
        
        # Update pattern confidence
        new_confidence = max(0.0, min(1.0, pattern.confidence + confidence_adjustment))
        pattern.confidence = new_confidence
        
        # Apply corrections if provided
        if corrections:
            for correction_type, correction_value in corrections.items():
                if correction_type == "pattern_value":
                    pattern.pattern_value = correction_value
                elif correction_type == "pattern_type":
                    try:
                        pattern.pattern_type = PatternType(correction_value)
                    except ValueError:
                        self.logger.warning(f"Invalid pattern type: {correction_value}")
        
        # Track pattern evolution
        evolution_threshold = self.learning_config.get("pattern_evolution_threshold", 0.1)
        if self.pattern_evolution_tracking and abs(new_confidence - old_confidence) > evolution_threshold:
            evolution_type = PatternEvolution.STRENGTHENING if new_confidence > old_confidence else PatternEvolution.WEAKENING
            
            evolution_event = PatternEvolutionEvent(
                event_id=f"evolution_{int(time.time())}_{pattern_id}",
                pattern_id=pattern_id,
                evolution_type=evolution_type,
                old_pattern=None,  # Could store old pattern copy
                new_pattern=pattern,
                confidence_change=new_confidence - old_confidence,
                evidence=[f"user_feedback: {feedback}"],
                metadata=context or {}
            )
            
            self.pattern_evolution_history.append(evolution_event)
            self.metrics["patterns_evolved"] += 1
        
        # Update metrics
        confidence_change_threshold = self.learning_config.get("confidence_change_threshold", 0.05)
        if abs(confidence_adjustment) > confidence_change_threshold:
            self.metrics["confidence_improvements"] += 1
        
        self.logger.debug(
            f"Applied feedback to pattern {pattern_id}: "
            f"confidence {old_confidence:.3f} -> {new_confidence:.3f}"
        )
        
        return {
            "success": True,
            "pattern_id": pattern_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "confidence_change": confidence_adjustment,
            "evolution_tracked": self.pattern_evolution_tracking
        }
    
    async def discover_semantic_clusters(
        self,
        min_cluster_size: int = 3,
        max_clusters: int = 20
    ) -> List[SemanticCluster]:
        """
        Discover semantic clusters from learned patterns.
        
        Args:
            min_cluster_size: Minimum size for cluster formation
            max_clusters: Maximum number of clusters to create
            
        Returns:
            List of discovered semantic clusters
        """
        if not self.semantic_clustering or len(self.learned_patterns) < min_cluster_size:
            return []
        
        # Extract features for clustering
        pattern_features = await self._extract_pattern_features()
        
        if len(pattern_features) < min_cluster_size:
            return []
        
        # Simple clustering algorithm (in production would use more sophisticated methods)
        clusters = await self._perform_semantic_clustering(
            pattern_features, min_cluster_size, max_clusters
        )
        
        # Store clusters
        new_clusters = []
        for cluster in clusters:
            cluster_id = f"cluster_{len(self.semantic_clusters)}_{int(time.time())}"
            
            semantic_cluster = SemanticCluster(
                cluster_id=cluster_id,
                cluster_label=cluster.get("label", f"Cluster {len(self.semantic_clusters)}"),
                patterns=cluster["patterns"],
                centroid_features=cluster["centroid"],
                coherence_score=cluster["coherence"],
                size=len(cluster["patterns"])
            )
            
            self.semantic_clusters[cluster_id] = semantic_cluster
            new_clusters.append(semantic_cluster)
        
        self.metrics["clusters_created"] += len(new_clusters)
        
        self.logger.info(f"Discovered {len(new_clusters)} semantic clusters")
        
        return new_clusters
    
    async def get_pattern_evolution_insights(
        self,
        time_window_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get insights about pattern evolution within time window.
        
        Args:
            time_window_hours: Time window for analysis
            
        Returns:
            Evolution insights and statistics
        """
        if not self.pattern_evolution_tracking:
            return {"evolution_tracking_disabled": True}
        
        # Use data-driven time window or default
        if time_window_hours is None:
            time_window_hours = self.learning_config.get("analysis_window_hours", 24.0)
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_events = [
            event for event in self.pattern_evolution_history
            if event.timestamp >= cutoff_time
        ]
        
        if not recent_events:
            return {"events_in_window": 0}
        
        # Analyze evolution types
        evolution_counts = Counter(event.evolution_type.value for event in recent_events)
        
        # Calculate confidence changes
        confidence_changes = [event.confidence_change for event in recent_events]
        
        # Find most evolved patterns
        pattern_evolution_counts = Counter(event.pattern_id for event in recent_events)
        most_evolved_patterns = pattern_evolution_counts.most_common(5)
        
        return {
            "events_in_window": len(recent_events),
            "time_window_hours": time_window_hours,
            "evolution_type_distribution": dict(evolution_counts),
            "confidence_change_stats": {
                "mean": statistics.mean(confidence_changes),
                "median": statistics.median(confidence_changes),
                "total_positive": len([c for c in confidence_changes if c > 0]),
                "total_negative": len([c for c in confidence_changes if c < 0])
            },
            "most_evolved_patterns": [
                {"pattern_id": pid, "evolution_count": count}
                for pid, count in most_evolved_patterns
            ],
            "patterns_emerged": len([e for e in recent_events if e.evolution_type == PatternEvolution.EMERGENCE]),
            "patterns_strengthened": len([e for e in recent_events if e.evolution_type == PatternEvolution.STRENGTHENING]),
            "patterns_weakened": len([e for e in recent_events if e.evolution_type == PatternEvolution.WEAKENING])
        }
    
    async def end_learning_session(self, session_id: str) -> LearningSession:
        """
        End an active learning session.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            Completed learning session
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Learning session {session_id} not found")
        
        session = self.active_sessions.pop(session_id)
        session.end_time = time.time()
        
        self.completed_sessions.append(session)
        self.metrics["learning_sessions"] += 1
        
        # Update average learning time
        session_duration = session.end_time - session.start_time
        current_avg = self.metrics["avg_learning_time"]
        total_sessions = self.metrics["learning_sessions"]
        
        self.metrics["avg_learning_time"] = (
            (current_avg * (total_sessions - 1) + session_duration) / total_sessions
        )
        
        self.logger.info(
            f"Ended learning session {session_id}: "
            f"processed {session.examples_processed} examples, "
            f"learned {session.patterns_learned} patterns"
        )
        
        return session
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_patterns = len(self.learned_patterns)
        
        # Calculate pattern confidence distribution
        if total_patterns > 0:
            confidences = [p.confidence for p in self.learned_patterns.values()]
            confidence_stats = {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "high_confidence_patterns": len([c for c in confidences if c > 0.7]),
                "low_confidence_patterns": len([c for c in confidences if c < 0.4])
            }
        else:
            confidence_stats = {}
        
        return {
            **self.metrics,
            "total_learned_patterns": total_patterns,
            "active_learning_sessions": len(self.active_sessions),
            "completed_learning_sessions": len(self.completed_sessions),
            "semantic_clusters": len(self.semantic_clusters),
            "pattern_evolution_events": len(self.pattern_evolution_history),
            "confidence_statistics": confidence_stats,
            "learning_examples_stored": len(self.learning_examples),
            "enabled_learning_modes": [mode.value for mode in self.enabled_learning_modes]
        }
    
    # Private implementation methods
    
    async def _unsupervised_pattern_learning(self, examples: List[LearningExample]) -> Dict[str, Any]:
        """Unsupervised pattern learning from examples"""
        new_patterns = 0
        evolved_patterns = 0
        confidence_improvements = []
        
        # Extract patterns from each example
        for example in examples:
            # Simple pattern extraction - would be more sophisticated in production
            words = re.findall(r'\b[a-zA-Z]{4,}\b', example.text.lower())
            word_counts = Counter(words)
            
            # Create patterns for frequent words
            for word, frequency in word_counts.items():
                if frequency >= 2:  # Minimum frequency
                    pattern_id = f"unsupervised_{hash(word) % 10000}"
                    
                    if pattern_id in self.learned_patterns:
                        # Update existing pattern
                        pattern = self.learned_patterns[pattern_id]
                        old_confidence = pattern.confidence
                        pattern.frequency += frequency
                        pattern.confidence = min(0.95, pattern.confidence + 0.05)
                        
                        confidence_improvements.append(pattern.confidence - old_confidence)
                        evolved_patterns += 1
                    else:
                        # Create new pattern
                        pattern = DiscoveredPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.SEMANTIC,
                            pattern_text=word,
                            frequency=frequency,
                            confidence=0.3 + min(0.4, frequency / 10.0),
                            metadata={"contexts": [example.text[:100]]}
                        )
                        
                        self.learned_patterns[pattern_id] = pattern
                        new_patterns += 1
        
        return {
            "new_patterns": new_patterns,
            "evolved_patterns": evolved_patterns,
            "confidence_improvements": confidence_improvements,
            "new_clusters": 0  # Clustering is separate step
        }
    
    async def _supervised_pattern_learning(self, examples: List[LearningExample]) -> Dict[str, Any]:
        """Supervised pattern learning from labeled examples"""
        new_patterns = 0
        evolved_patterns = 0
        confidence_improvements = []
        
        for example in examples:
            if not example.labels:
                continue
            
            # Learn from labeled patterns
            for label_type, label_value in example.labels.items():
                if isinstance(label_value, str):
                    pattern_id = f"supervised_{hash(f'{label_type}_{label_value}') % 10000}"
                    
                    if pattern_id in self.learned_patterns:
                        # Strengthen existing pattern
                        pattern = self.learned_patterns[pattern_id]
                        old_confidence = pattern.confidence
                        pattern.confidence = min(0.95, pattern.confidence + 0.1)  # Higher learning rate for supervised
                        pattern.frequency += 1
                        
                        confidence_improvements.append(pattern.confidence - old_confidence)
                        evolved_patterns += 1
                    else:
                        # Create new supervised pattern
                        try:
                            pattern_type = PatternType(label_type.lower()) if label_type.lower() in [t.value for t in PatternType] else PatternType.CONCEPT
                        except ValueError:
                            pattern_type = PatternType.CONCEPT
                        
                        pattern = DiscoveredPattern(
                            pattern_id=pattern_id,
                            pattern_type=pattern_type,
                            pattern_text=str(label_value),
                            frequency=1,
                            confidence=0.8,  # High initial confidence for supervised learning
                            metadata={"supervised": True, "label_type": label_type, "contexts": [example.text[:100]]}
                        )
                        
                        self.learned_patterns[pattern_id] = pattern
                        new_patterns += 1
        
        return {
            "new_patterns": new_patterns,
            "evolved_patterns": evolved_patterns,
            "confidence_improvements": confidence_improvements,
            "new_clusters": 0
        }
    
    async def _semi_supervised_pattern_learning(self, examples: List[LearningExample]) -> Dict[str, Any]:
        """Semi-supervised pattern learning combining labeled and unlabeled examples"""
        # Separate labeled and unlabeled examples
        labeled_examples = [ex for ex in examples if ex.labels]
        unlabeled_examples = [ex for ex in examples if not ex.labels]
        
        # Learn from labeled examples first
        supervised_results = await self._supervised_pattern_learning(labeled_examples)
        
        # Then learn from unlabeled examples using learned patterns as guidance
        unsupervised_results = await self._unsupervised_pattern_learning(unlabeled_examples)
        
        return {
            "new_patterns": supervised_results["new_patterns"] + unsupervised_results["new_patterns"],
            "evolved_patterns": supervised_results["evolved_patterns"] + unsupervised_results["evolved_patterns"],
            "confidence_improvements": supervised_results["confidence_improvements"] + unsupervised_results["confidence_improvements"],
            "new_clusters": 0,
            "labeled_examples": len(labeled_examples),
            "unlabeled_examples": len(unlabeled_examples)
        }
    
    async def _reinforcement_pattern_learning(self, examples: List[LearningExample]) -> Dict[str, Any]:
        """Reinforcement learning from examples with feedback"""
        new_patterns = 0
        evolved_patterns = 0
        confidence_improvements = []
        
        for example in examples:
            if not example.feedback:
                continue
            
            # Apply reinforcement learning based on feedback
            feedback = example.feedback
            reward = feedback.get("reward", 0.0)  # Positive or negative reward
            
            # Extract patterns and apply reward-based learning
            words = re.findall(r'\b[a-zA-Z]{4,}\b', example.text.lower())
            
            for word in set(words):  # Unique words
                pattern_id = f"reinforcement_{hash(word) % 10000}"
                
                if pattern_id in self.learned_patterns:
                    # Apply reward to existing pattern
                    pattern = self.learned_patterns[pattern_id]
                    old_confidence = pattern.confidence
                    
                    # Adjust confidence based on reward
                    confidence_adjustment = self.confidence_learning_rate * reward
                    pattern.confidence = max(0.0, min(1.0, pattern.confidence + confidence_adjustment))
                    
                    confidence_improvements.append(pattern.confidence - old_confidence)
                    evolved_patterns += 1
                elif reward > 0:  # Only create new patterns for positive rewards
                    # Create new pattern with reward-based confidence
                    initial_confidence = 0.5 + (reward * 0.3)  # Scale reward to confidence
                    
                    pattern = DiscoveredPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.SEMANTIC,
                        pattern_text=word,
                        frequency=1,
                        confidence=max(0.1, min(0.9, initial_confidence)),
                        metadata={"reinforcement": True, "initial_reward": reward, "contexts": [example.text[:100]]}
                    )
                    
                    self.learned_patterns[pattern_id] = pattern
                    new_patterns += 1
        
        return {
            "new_patterns": new_patterns,
            "evolved_patterns": evolved_patterns,
            "confidence_improvements": confidence_improvements,
            "new_clusters": 0
        }
    
    async def _extract_pattern_features(self) -> Dict[str, Dict[str, float]]:
        """Extract features from learned patterns for clustering"""
        features = {}
        
        for pattern_id, pattern in self.learned_patterns.items():
            pattern_features = {
                "confidence": pattern.confidence,
                "frequency": min(1.0, pattern.frequency / 10.0),  # Normalized frequency
                "pattern_length": len(pattern.pattern_value) / 20.0,  # Normalized length
                "pattern_type_entity": 1.0 if pattern.pattern_type == PatternType.ENTITY else 0.0,
                "pattern_type_concept": 1.0 if pattern.pattern_type == PatternType.CONCEPT else 0.0,
                "pattern_type_relationship": 1.0 if pattern.pattern_type == PatternType.RELATIONSHIP else 0.0,
                "pattern_type_temporal": 1.0 if pattern.pattern_type == PatternType.TEMPORAL else 0.0,
                "pattern_type_numerical": 1.0 if pattern.pattern_type == PatternType.NUMERICAL else 0.0,
                "has_contexts": 1.0 if pattern.contexts else 0.0,
                "context_count": min(1.0, len(pattern.contexts) / 5.0)  # Normalized context count
            }
            
            features[pattern_id] = pattern_features
            self.pattern_features[pattern_id] = pattern_features
        
        return features
    
    async def _perform_semantic_clustering(
        self,
        pattern_features: Dict[str, Dict[str, float]],
        min_cluster_size: int,
        max_clusters: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic clustering on pattern features"""
        if len(pattern_features) < min_cluster_size:
            return []
        
        # Simple clustering algorithm (in production would use k-means, hierarchical clustering, etc.)
        clusters = []
        unassigned_patterns = list(pattern_features.keys())
        
        cluster_count = 0
        while unassigned_patterns and cluster_count < max_clusters:
            # Start new cluster with first unassigned pattern
            seed_pattern = unassigned_patterns[0]
            cluster_patterns = [seed_pattern]
            unassigned_patterns.remove(seed_pattern)
            
            # Find similar patterns for this cluster
            seed_features = pattern_features[seed_pattern]
            
            similar_patterns = []
            for pattern_id in unassigned_patterns[:]:  # Copy list for safe iteration
                similarity = self._calculate_feature_similarity(seed_features, pattern_features[pattern_id])
                
                if similarity >= self.cluster_similarity_threshold:
                    similar_patterns.append(pattern_id)
                    cluster_patterns.append(pattern_id)
                    unassigned_patterns.remove(pattern_id)
            
            # Only create cluster if it meets minimum size
            if len(cluster_patterns) >= min_cluster_size:
                # Calculate cluster centroid
                centroid = self._calculate_centroid([pattern_features[pid] for pid in cluster_patterns])
                
                # Calculate coherence score
                coherence = self._calculate_cluster_coherence(cluster_patterns, pattern_features)
                
                clusters.append({
                    "patterns": cluster_patterns,
                    "centroid": centroid,
                    "coherence": coherence,
                    "label": f"Semantic_Cluster_{cluster_count}"
                })
                
                cluster_count += 1
            else:
                # Return patterns to unassigned if cluster too small
                unassigned_patterns.extend(cluster_patterns[1:])  # Keep seed out
        
        return clusters
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature vectors"""
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(features1[feat] * features2[feat] for feat in common_features)
        norm1 = sum(v ** 2 for v in features1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in features2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_centroid(self, feature_vectors: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate centroid of feature vectors"""
        if not feature_vectors:
            return {}
        
        all_features = set()
        for features in feature_vectors:
            all_features.update(features.keys())
        
        centroid = {}
        for feature in all_features:
            values = [features.get(feature, 0.0) for features in feature_vectors]
            centroid[feature] = statistics.mean(values)
        
        return centroid
    
    def _calculate_cluster_coherence(
        self,
        cluster_patterns: List[str],
        all_features: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate coherence score for a cluster"""
        if len(cluster_patterns) < 2:
            return 1.0
        
        # Calculate average pairwise similarity within cluster
        similarities = []
        
        for i in range(len(cluster_patterns)):
            for j in range(i + 1, len(cluster_patterns)):
                pattern1_features = all_features[cluster_patterns[i]]
                pattern2_features = all_features[cluster_patterns[j]]
                similarity = self._calculate_feature_similarity(pattern1_features, pattern2_features)
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0


__all__ = [
    'PatternLearningSystem',
    'LearningExample',
    'PatternEvolutionEvent',
    'SemanticCluster',
    'LearningSession',
    'LearningMode',
    'PatternEvolution'
]