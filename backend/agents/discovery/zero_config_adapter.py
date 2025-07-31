"""
Zero-Configuration Domain Adapter - Automatic domain detection and agent adaptation.
Enables agents to work with any domain without manual configuration.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import statistics

from .domain_pattern_engine import DomainPatternEngine, DomainFingerprint, PatternType
from .constants import (
    StatisticalConfidenceCalculator, 
    SearchWeightCalculator,
    DataDrivenConfigurationManager,
    create_data_driven_configuration
)
from ..base import AgentContext, AgentCapability

logger = logging.getLogger(__name__)


class DomainAdaptationStrategy(Enum):
    """Strategies for domain adaptation"""
    CONSERVATIVE = "conservative"      # High confidence required for adaptation
    BALANCED = "balanced"             # Moderate confidence thresholds
    AGGRESSIVE = "aggressive"         # Low confidence, rapid adaptation
    LEARNING = "learning"             # Continuous learning mode


class AdaptationConfidence(Enum):
    """Confidence levels for domain adaptation - determined from real data"""
    UNKNOWN = "unknown"               # Cannot determine domain
    LOW = "low"                      # Low quartile of confidence distribution
    MEDIUM = "medium"                # Median confidence level
    HIGH = "high"                    # Upper quartile confidence level  
    VERY_HIGH = "very_high"          # Top decile confidence level


@dataclass
class DomainDetectionResult:
    """Result of domain detection analysis"""
    detected_domain: Optional[str]
    confidence: float
    confidence_level: AdaptationConfidence
    domain_fingerprint: Optional[DomainFingerprint]
    similar_domains: List[Tuple[str, float]] = field(default_factory=list)
    adaptation_recommendations: Dict[str, Any] = field(default_factory=dict)
    detection_time_ms: float = 0.0
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set confidence level based on statistical analysis of confidence distribution
        # This will be replaced with real data-driven calculation in production
        self.confidence_level = self._calculate_confidence_level_from_data(self.confidence)
    
    def _calculate_confidence_level_from_data(self, confidence: float) -> AdaptationConfidence:
        """
        Calculate confidence level from actual confidence distribution data.
        
        In production, this should use real confidence samples from system operations
        to determine statistical quartiles for confidence classification.
        """
        # TODO: Replace with real confidence sample data from production system
        # For now, using bootstrap estimation until real data is available
        
        # This is a temporary implementation that needs real data
        # The proper implementation would call:
        # confidence_calc = StatisticalConfidenceCalculator()
        # levels = confidence_calc.calculate_confidence_levels(real_confidence_samples)
        
        if confidence >= 0.9:  # Temporary - should be levels["VERY_HIGH"]
            return AdaptationConfidence.VERY_HIGH
        elif confidence >= 0.7:  # Temporary - should be levels["HIGH"] 
            return AdaptationConfidence.HIGH
        elif confidence >= 0.5:  # Temporary - should be levels["MEDIUM"]
            return AdaptationConfidence.MEDIUM
        elif confidence >= 0.3:  # Temporary - should be levels["LOW"]
            return AdaptationConfidence.LOW
        else:
            return AdaptationConfidence.UNKNOWN


@dataclass
class AgentAdaptationProfile:
    """Profile for agent adaptation to specific domain"""
    domain_id: str
    domain_name: Optional[str]
    recommended_capabilities: List[AgentCapability]
    configuration_overrides: Dict[str, Any]
    search_strategy_preferences: Dict[str, float]
    reasoning_pattern_weights: Dict[str, float]
    context_priorities: Dict[str, float]
    performance_targets: Dict[str, float]
    tool_recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZeroConfigAdapter:
    """
    Zero-configuration domain adapter that automatically detects domains and adapts agents.
    
    Provides seamless domain adaptation without manual configuration:
    - Automatic domain detection from queries and context
    - Agent capability adjustment based on domain patterns
    - Dynamic configuration optimization
    - Continuous learning and improvement
    - Integration with existing agent architecture
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize zero-configuration domain adapter.
        
        Args:
            config: Adapter configuration
                - pattern_engine: DomainPatternEngine instance
                - adaptation_strategy: Default adaptation strategy
                - confidence_thresholds: Confidence thresholds for adaptation
                - enable_continuous_learning: Enable learning from interactions
                - similarity_threshold: Threshold for domain similarity matching
                - cache_adaptation_profiles: Cache domain adaptation profiles
        """
        self.config = config
        self.pattern_engine = config.get("pattern_engine") or DomainPatternEngine({})
        self.adaptation_strategy = DomainAdaptationStrategy(
            config.get("adaptation_strategy", "balanced")
        )
        
        # Initialize logger first
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize data-driven configuration system
        self.data_driven_config = self._initialize_data_driven_configuration(config)
        self.confidence_thresholds = self.data_driven_config.get("strategy_thresholds", {
            # Temporary fallbacks until real performance data is available
            "conservative": 0.8,  # TODO: Replace with calculated value from performance data
            "balanced": 0.6,      # TODO: Replace with calculated value from performance data  
            "aggressive": 0.4,    # TODO: Replace with calculated value from performance data
            "learning": 0.3       # TODO: Replace with calculated value from performance data
        })
        self.enable_continuous_learning = config.get("enable_continuous_learning", True)
        self.similarity_threshold = self.data_driven_config.get("similarity_threshold", 0.7)  # TODO: Replace with data-driven calculation
        self.cache_adaptation_profiles = config.get("cache_adaptation_profiles", True)
        
        # Adaptation state
        self.known_domains: Dict[str, DomainFingerprint] = {}
        self.adaptation_profiles: Dict[str, AgentAdaptationProfile] = {}
        self.query_domain_cache: Dict[str, DomainDetectionResult] = {}
        
        # Learning history
        self.interaction_history: List[Dict[str, Any]] = []
        self.adaptation_successes: Dict[str, int] = {}
        self.adaptation_failures: Dict[str, int] = {}
        
        # Performance metrics
        self.metrics = {
            "domains_detected": 0,
            "adaptations_performed": 0,
            "successful_adaptations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_detection_time": 0.0,
            "learning_iterations": 0
        }
    
    def _initialize_data_driven_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize data-driven configuration from real system data.
        
        This method replaces hardcoded constants with statistically derived values
        based on actual system performance and user interaction data.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Data-driven configuration dictionary
        """
        # Check if real performance data is provided
        performance_data = config.get("historical_performance_data")
        confidence_samples = config.get("confidence_samples")
        
        if performance_data and confidence_samples:
            # Use real data to create configuration
            self.logger.info("Creating data-driven configuration from real system data")
            return create_data_driven_configuration(
                confidence_samples=confidence_samples,
                performance_data=performance_data,
                context_data=config.get("context_effectiveness_data"),
                quality_data=config.get("quality_impact_data")
            )
        else:
            # Log that we're using temporary values
            self.logger.warning(
                "No real performance data provided - using temporary fallback values. "
                "For production, provide historical_performance_data and confidence_samples "
                "to enable data-driven configuration."
            )
            
            # Return minimal configuration structure for bootstrap
            return {
                "configuration_type": "bootstrap_fallback",
                "created_from_real_data": False,
                "similarity_threshold": 0.7,  # Bootstrap value
                "strategy_thresholds": {
                    "conservative": 0.8,
                    "balanced": 0.6, 
                    "aggressive": 0.4,
                    "learning": 0.3
                },
                "requires_real_data": True
            }
    
    async def detect_domain_from_query(
        self,
        query: str,
        context: Optional[AgentContext] = None,
        additional_text: Optional[List[str]] = None
    ) -> DomainDetectionResult:
        """
        Detect domain from user query and optional context.
        
        Args:
            query: User query to analyze
            context: Optional agent context for additional information
            additional_text: Optional additional text for domain analysis
            
        Returns:
            DomainDetectionResult with detected domain and confidence
        """
        start_time = time.time()
        
        # Check cache first
        query_key = self._generate_query_key(query, context, additional_text)
        if query_key in self.query_domain_cache:
            self.metrics["cache_hits"] += 1
            cached_result = self.query_domain_cache[query_key]
            self.logger.debug(f"Using cached domain detection for query: {query[:50]}...")
            return cached_result
        
        self.metrics["cache_misses"] += 1
        
        # Prepare text corpus for analysis
        text_corpus = [query]
        if additional_text:
            text_corpus.extend(additional_text)
        if context and context.conversation_history:
            # Add recent conversation context
            for turn in context.conversation_history[-3:]:  # Last 3 turns
                if isinstance(turn, dict) and "query" in turn:
                    text_corpus.append(turn["query"])
        
        # Analyze corpus for domain patterns
        try:
            domain_fingerprint = await self.pattern_engine.analyze_text_corpus(
                text_corpus=text_corpus,
                corpus_metadata={"source": "query_analysis", "query": query}
            )
            
            # Find similar known domains
            similar_domains = await self._find_similar_known_domains(domain_fingerprint)
            
            # Determine detected domain and confidence
            detected_domain, confidence = await self._determine_domain_from_patterns(
                domain_fingerprint, similar_domains
            )
            
            # Generate adaptation recommendations
            recommendations = await self._generate_adaptation_recommendations(
                domain_fingerprint, detected_domain, confidence
            )
            
            detection_time = (time.time() - start_time) * 1000
            
            # Create detection result
            result = DomainDetectionResult(
                detected_domain=detected_domain,
                confidence=confidence,
                confidence_level=AdaptationConfidence.UNKNOWN,  # Will be set in __post_init__
                domain_fingerprint=domain_fingerprint,
                similar_domains=similar_domains,
                adaptation_recommendations=recommendations,
                detection_time_ms=detection_time,
                analysis_details={
                    "pattern_count": len([p for patterns in domain_fingerprint.primary_patterns.values() for p in patterns]),
                    "vocabulary_size": domain_fingerprint.vocabulary_size,
                    "concept_density": domain_fingerprint.concept_density,
                    "confidence_distribution": self._analyze_pattern_confidence_distribution(domain_fingerprint)
                }
            )
            
            # Cache the result
            if self.cache_adaptation_profiles:
                self.query_domain_cache[query_key] = result
            
            # Update metrics
            self.metrics["domains_detected"] += 1
            self.metrics["avg_detection_time"] = (
                (self.metrics["avg_detection_time"] * (self.metrics["domains_detected"] - 1) + detection_time) /
                self.metrics["domains_detected"]
            )
            
            self.logger.info(
                f"Domain detected: {detected_domain or 'unknown'} "
                f"(confidence: {confidence:.3f}, time: {detection_time:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Domain detection failed: {e}")
            
            # Return unknown domain result
            return DomainDetectionResult(
                detected_domain=None,
                confidence=0.0,
                confidence_level=AdaptationConfidence.UNKNOWN,
                domain_fingerprint=None,
                detection_time_ms=(time.time() - start_time) * 1000,
                analysis_details={"error": str(e)}
            )
    
    async def adapt_agent_to_domain(
        self,
        detection_result: DomainDetectionResult,
        base_agent_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], AgentAdaptationProfile]:
        """
        Adapt agent configuration to detected domain.
        
        Args:
            detection_result: Result from domain detection
            base_agent_config: Base agent configuration to adapt
            
        Returns:
            Tuple of (adapted_config, adaptation_profile)
        """
        if not detection_result.detected_domain or detection_result.confidence < self._get_confidence_threshold():
            # Return base configuration if domain detection is insufficient
            self.logger.debug("Insufficient domain confidence, using base configuration")
            return base_agent_config, self._create_default_adaptation_profile()
        
        domain_id = detection_result.detected_domain
        
        # Check for existing adaptation profile
        if domain_id in self.adaptation_profiles:
            existing_profile = self.adaptation_profiles[domain_id]
            adapted_config = await self._apply_adaptation_profile(base_agent_config, existing_profile)
            self.logger.debug(f"Using existing adaptation profile for domain: {domain_id}")
            return adapted_config, existing_profile
        
        # Create new adaptation profile
        adaptation_profile = await self._create_adaptation_profile(detection_result, base_agent_config)
        
        # Apply the profile to configuration
        adapted_config = await self._apply_adaptation_profile(base_agent_config, adaptation_profile)
        
        # Store the profile for future use
        if self.cache_adaptation_profiles:
            self.adaptation_profiles[domain_id] = adaptation_profile
            self.known_domains[domain_id] = detection_result.domain_fingerprint
        
        # Update metrics
        self.metrics["adaptations_performed"] += 1
        
        self.logger.info(f"Created new adaptation profile for domain: {domain_id}")
        
        return adapted_config, adaptation_profile
    
    async def learn_from_interaction(
        self,
        query: str,
        detected_domain: Optional[str],
        adaptation_profile: Optional[AgentAdaptationProfile],
        interaction_result: Dict[str, Any]
    ) -> None:
        """
        Learn from agent interaction to improve future adaptations.
        
        Args:
            query: Original user query
            detected_domain: Domain that was detected
            adaptation_profile: Adaptation profile that was used
            interaction_result: Result of the agent interaction
        """
        if not self.enable_continuous_learning:
            return
        
        # Extract success metrics from interaction result
        success = interaction_result.get("success", False)
        confidence = interaction_result.get("confidence", 0.0)
        response_time = interaction_result.get("response_time_ms", 0.0)
        user_satisfaction = interaction_result.get("user_satisfaction", 0.5)
        
        # Record interaction
        interaction_record = {
            "timestamp": time.time(),
            "query": query,
            "detected_domain": detected_domain,
            "adaptation_used": adaptation_profile.domain_id if adaptation_profile else None,
            "success": success,
            "confidence": confidence,
            "response_time_ms": response_time,
            "user_satisfaction": user_satisfaction,
            "metadata": interaction_result.get("metadata", {})
        }
        
        self.interaction_history.append(interaction_record)
        
        # Update success/failure counters
        if detected_domain:
            if success and confidence > 0.6:
                self.adaptation_successes[detected_domain] = self.adaptation_successes.get(detected_domain, 0) + 1
                self.metrics["successful_adaptations"] += 1
            elif not success or confidence < 0.4:
                self.adaptation_failures[detected_domain] = self.adaptation_failures.get(detected_domain, 0) + 1
        
        # Trigger adaptation improvement if we have enough data
        if len(self.interaction_history) % 10 == 0:  # Every 10 interactions
            await self._improve_adaptations()
        
        self.metrics["learning_iterations"] += 1
        
        self.logger.debug(f"Learned from interaction: domain={detected_domain}, success={success}")
    
    async def get_domain_recommendations(
        self,
        query: str,
        num_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get domain recommendations for a query based on learned patterns.
        
        Args:
            query: Query to get recommendations for
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of domain recommendations with confidence scores
        """
        if not self.known_domains:
            return []
        
        # Analyze query for patterns
        query_fingerprint = await self.pattern_engine.analyze_text_corpus([query])
        
        # Compare with known domains
        domain_similarities = []
        
        for domain_id, domain_fingerprint in self.known_domains.items():
            similarity = query_fingerprint.get_similarity_score(domain_fingerprint)
            
            if similarity > 0.1:  # Minimum similarity threshold
                success_rate = self._calculate_domain_success_rate(domain_id)
                
                # Combine similarity with historical success rate using data-driven weights
                quality_weights = self._get_quality_weights()
                recommendation_score = (
                    similarity * quality_weights["similarity"] + 
                    success_rate * quality_weights["success_rate"]
                )
                
                domain_similarities.append({
                    "domain_id": domain_id,
                    "domain_name": domain_fingerprint.domain_name,
                    "similarity": similarity,
                    "success_rate": success_rate,
                    "recommendation_score": recommendation_score,
                    "interaction_count": (
                        self.adaptation_successes.get(domain_id, 0) + 
                        self.adaptation_failures.get(domain_id, 0)
                    )
                })
        
        # Sort by recommendation score and return top recommendations
        domain_similarities.sort(key=lambda x: x["recommendation_score"], reverse=True)
        
        return domain_similarities[:num_recommendations]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_adaptations = self.metrics["adaptations_performed"]
        success_rate = 0.0
        if total_adaptations > 0:
            success_rate = self.metrics["successful_adaptations"] / total_adaptations
        
        return {
            **self.metrics,
            "adaptation_success_rate": success_rate,
            "known_domains_count": len(self.known_domains),
            "cached_profiles_count": len(self.adaptation_profiles),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
            ),
            "domain_success_rates": {
                domain: self._calculate_domain_success_rate(domain)
                for domain in self.known_domains.keys()
            },
            "interaction_history_size": len(self.interaction_history)
        }
    
    # Private implementation methods
    
    def _generate_query_key(
        self, 
        query: str, 
        context: Optional[AgentContext],
        additional_text: Optional[List[str]]
    ) -> str:
        """Generate cache key for query"""
        key_components = [query]
        
        if context and context.domain:
            key_components.append(f"domain:{context.domain}")
        
        if additional_text:
            key_components.extend(additional_text[:2])  # Limit for key size
        
        key_string = "|".join(key_components)
        return f"query_{hash(key_string) % 100000}"
    
    async def _find_similar_known_domains(
        self, 
        fingerprint: DomainFingerprint
    ) -> List[Tuple[str, float]]:
        """Find similar domains in known domain database"""
        similar_domains = []
        
        for domain_id, known_fingerprint in self.known_domains.items():
            similarity = fingerprint.get_similarity_score(known_fingerprint)
            
            if similarity >= self.similarity_threshold:
                similar_domains.append((domain_id, similarity))
        
        # Sort by similarity
        similar_domains.sort(key=lambda x: x[1], reverse=True)
        
        return similar_domains
    
    async def _determine_domain_from_patterns(
        self,
        fingerprint: DomainFingerprint,
        similar_domains: List[Tuple[str, float]]
    ) -> Tuple[Optional[str], float]:
        """Determine domain and confidence from patterns and similarities"""
        
        # If we have a highly similar known domain, use it
        if similar_domains and similar_domains[0][1] > 0.8:
            return similar_domains[0][0], similar_domains[0][1]
        
        # Otherwise, assess if this is a new domain based on pattern strength
        pattern_confidence = fingerprint.confidence
        
        if pattern_confidence > 0.7:
            # Strong patterns suggest a distinct domain
            new_domain_id = f"domain_{fingerprint.domain_id}"
            return new_domain_id, pattern_confidence
        elif similar_domains and similar_domains[0][1] > 0.6:
            # Moderate similarity to known domain
            return similar_domains[0][0], similar_domains[0][1] * 0.8  # Reduce confidence
        else:
            # Insufficient evidence for domain determination
            return None, pattern_confidence
    
    async def _generate_adaptation_recommendations(
        self,
        fingerprint: DomainFingerprint,
        detected_domain: Optional[str],
        confidence: float
    ) -> Dict[str, Any]:
        """Generate recommendations for agent adaptation"""
        recommendations = {
            "adaptation_strategy": self.adaptation_strategy.value,
            "confidence_sufficient": confidence >= self._get_confidence_threshold(),
            "recommended_capabilities": [],
            "search_preferences": {},
            "reasoning_adjustments": {},
            "performance_targets": {}
        }
        
        if not detected_domain or confidence < self._get_confidence_threshold("low"):
            recommendations["recommendation"] = "use_default_configuration"
            return recommendations
        
        # Analyze patterns to recommend capabilities
        pattern_types = set(fingerprint.primary_patterns.keys())
        
        if PatternType.RELATIONSHIP in pattern_types:
            recommendations["recommended_capabilities"].append("REASONING_SYNTHESIS")
            recommendations["search_preferences"]["graph_search_weight"] = 0.4
        
        if PatternType.TEMPORAL in pattern_types:
            recommendations["recommended_capabilities"].append("CONTEXT_MANAGEMENT")
            recommendations["reasoning_adjustments"]["temporal_awareness"] = True
        
        if PatternType.NUMERICAL in pattern_types:
            recommendations["recommended_capabilities"].append("TOOL_DISCOVERY")
            recommendations["search_preferences"]["analytical_tools"] = True
        
        if fingerprint.concept_density > 0.5:
            recommendations["search_preferences"]["vector_search_weight"] = 0.5
            recommendations["reasoning_adjustments"]["semantic_depth"] = "high"
        
        # Set performance targets based on domain complexity
        if fingerprint.relationship_complexity > CONFIDENCE.LOW:
            recommendations["performance_targets"]["max_response_time"] = 4.0  # More time for complex domains
        else:
            recommendations["performance_targets"]["max_response_time"] = 2.0  # Fast response for simple domains
        
        return recommendations
    
    def _get_confidence_threshold(self) -> float:
        """Get confidence threshold for current adaptation strategy"""
        return self.confidence_thresholds[self.adaptation_strategy.value]
    
    async def _create_adaptation_profile(
        self,
        detection_result: DomainDetectionResult,
        base_config: Dict[str, Any]
    ) -> AgentAdaptationProfile:
        """Create new adaptation profile from detection result"""
        
        domain_id = detection_result.detected_domain
        fingerprint = detection_result.domain_fingerprint
        recommendations = detection_result.adaptation_recommendations
        
        # Extract capabilities from recommendations
        recommended_capabilities = []
        for cap_name in recommendations.get("recommended_capabilities", []):
            try:
                capability = AgentCapability(cap_name.lower())
                recommended_capabilities.append(capability)
            except ValueError:
                self.logger.warning(f"Unknown capability: {cap_name}")
        
        # Create configuration overrides
        config_overrides = {}
        if recommendations.get("reasoning_adjustments"):
            config_overrides["reasoning"] = recommendations["reasoning_adjustments"]
        
        # Set search strategy preferences
        search_preferences = {
            "vector_search": recommendations.get("search_preferences", {}).get("vector_search_weight", SEARCH_WEIGHTS.DEFAULT_VECTOR),
            "graph_search": recommendations.get("search_preferences", {}).get("graph_search_weight", SEARCH_WEIGHTS.DEFAULT_GRAPH),
            "gnn_search": SEARCH_WEIGHTS.DEFAULT_GNN
        }
        
        # Set reasoning pattern weights based on domain characteristics
        reasoning_weights = {
            "chain_of_thought": SEARCH_WEIGHTS.CHAIN_OF_THOUGHT,
            "evidence_synthesis": SEARCH_WEIGHTS.EVIDENCE_SYNTHESIS,
            "multi_perspective": SEARCH_WEIGHTS.MULTI_PERSPECTIVE
        }
        
        if fingerprint and fingerprint.relationship_complexity > CONFIDENCE.LOW:
            reasoning_weights["evidence_synthesis"] = 0.5
            reasoning_weights["chain_of_thought"] = SEARCH_WEIGHTS.COMPLEX_CHAIN_OF_THOUGHT
        
        # Set context priorities
        context_priorities = {
            "conversation_context": CONTEXT.CONVERSATION_CONTEXT,
            "domain_context": 0.4,
            "temporal_context": 0.2,
            "performance_context": 0.1
        }
        
        if fingerprint and PatternType.TEMPORAL in fingerprint.primary_patterns:
            context_priorities["temporal_context"] = CONTEXT.TEMPORAL_DOMAIN_TEMPORAL
            context_priorities["domain_context"] = CONTEXT.TEMPORAL_DOMAIN_DOMAIN
        
        # Set performance targets
        performance_targets = recommendations.get("performance_targets", {
            "max_response_time": PERFORMANCE.DEFAULT_MAX_RESPONSE_TIME,
            "min_confidence": PERFORMANCE.DEFAULT_MIN_CONFIDENCE
        })
        
        return AgentAdaptationProfile(
            domain_id=domain_id,
            domain_name=fingerprint.domain_name if fingerprint else None,
            recommended_capabilities=recommended_capabilities,
            configuration_overrides=config_overrides,
            search_strategy_preferences=search_preferences,
            reasoning_pattern_weights=reasoning_weights,
            context_priorities=context_priorities,
            performance_targets=performance_targets,
            confidence=detection_result.confidence,
            metadata={
                "created_from": "domain_detection",
                "pattern_count": len([p for patterns in fingerprint.primary_patterns.values() for p in patterns]) if fingerprint else 0,
                "adaptation_strategy": self.adaptation_strategy.value
            }
        )
    
    def _create_default_adaptation_profile(self) -> AgentAdaptationProfile:
        """Create default adaptation profile for unknown domains"""
        return AgentAdaptationProfile(
            domain_id="default",
            domain_name="Unknown Domain",
            recommended_capabilities=[
                AgentCapability.SEARCH_ORCHESTRATION,
                AgentCapability.REASONING_SYNTHESIS
            ],
            configuration_overrides={},
            search_strategy_preferences={"vector_search": SEARCH_WEIGHTS.DEFAULT_VECTOR, "graph_search": SEARCH_WEIGHTS.DEFAULT_GRAPH, "gnn_search": SEARCH_WEIGHTS.DEFAULT_GNN},
            reasoning_pattern_weights={"chain_of_thought": 0.5, "evidence_synthesis": 0.5},
            context_priorities={"conversation_context": 0.4, "domain_context": 0.6},
            performance_targets={"max_response_time": PERFORMANCE.DEFAULT_MAX_RESPONSE_TIME, "min_confidence": PERFORMANCE.DEFAULT_MIN_CONFIDENCE},
            confidence=CACHE.NEUTRAL_SUCCESS_RATE
        )
    
    async def _apply_adaptation_profile(
        self,
        base_config: Dict[str, Any],
        profile: AgentAdaptationProfile
    ) -> Dict[str, Any]:
        """Apply adaptation profile to base configuration"""
        adapted_config = base_config.copy()
        
        # Apply configuration overrides
        for key, value in profile.configuration_overrides.items():
            adapted_config[key] = value
        
        # Set search preferences
        adapted_config["search_preferences"] = profile.search_strategy_preferences
        
        # Set reasoning weights
        adapted_config["reasoning_weights"] = profile.reasoning_pattern_weights
        
        # Set context priorities
        adapted_config["context_priorities"] = profile.context_priorities
        
        # Set performance targets
        adapted_config["performance_targets"] = profile.performance_targets
        
        # Add domain metadata
        adapted_config["domain_adaptation"] = {
            "domain_id": profile.domain_id,
            "domain_name": profile.domain_name,
            "adaptation_confidence": profile.confidence,
            "adapted_at": time.time()
        }
        
        return adapted_config
    
    def _analyze_pattern_confidence_distribution(self, fingerprint: DomainFingerprint) -> Dict[str, Any]:
        """Analyze confidence distribution of patterns in fingerprint"""
        all_patterns = [p for patterns in fingerprint.primary_patterns.values() for p in patterns]
        
        if not all_patterns:
            return {"count": 0}
        
        confidences = [p.confidence for p in all_patterns]
        
        return {
            "count": len(confidences),
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "high_confidence_patterns": len([c for c in confidences if c > 0.7]),
            "low_confidence_patterns": len([c for c in confidences if c < 0.4])
        }
    
    def _calculate_domain_success_rate(self, domain_id: str) -> float:
        """Calculate success rate for a specific domain"""
        successes = self.adaptation_successes.get(domain_id, 0)
        failures = self.adaptation_failures.get(domain_id, 0)
        total = successes + failures
        
        if total == 0:
            return 0.5  # Neutral success rate for new domains
        
        return successes / total
    
    async def _improve_adaptations(self) -> None:
        """Improve adaptation profiles based on interaction history"""
        if len(self.interaction_history) < 5:
            return
        
        # Analyze recent interactions for improvement opportunities
        recent_interactions = self.interaction_history[-20:]  # Last 20 interactions
        
        # Group by domain
        domain_interactions = defaultdict(list)
        for interaction in recent_interactions:
            domain = interaction.get("detected_domain")
            if domain:
                domain_interactions[domain].append(interaction)
        
        # Update adaptation profiles based on performance
        for domain_id, interactions in domain_interactions.items():
            if domain_id in self.adaptation_profiles and len(interactions) >= 3:
                await self._update_adaptation_profile(domain_id, interactions)
        
        self.logger.info(f"Improved adaptations based on {len(recent_interactions)} recent interactions")
    
    async def _update_adaptation_profile(
        self,
        domain_id: str,
        interactions: List[Dict[str, Any]]
    ) -> None:
        """Update adaptation profile based on interaction feedback"""
        profile = self.adaptation_profiles[domain_id]
        
        # Calculate performance metrics
        success_rate = sum(1 for i in interactions if i["success"]) / len(interactions)
        avg_confidence = statistics.mean([i["confidence"] for i in interactions])
        avg_response_time = statistics.mean([i["response_time_ms"] for i in interactions])
        
        # Adjust profile based on performance
        if success_rate < 0.6:
            # Poor success rate - be more conservative
            profile.performance_targets["min_confidence"] = min(0.8, profile.performance_targets.get("min_confidence", 0.7) + 0.1)
        elif success_rate > 0.8:
            # Good success rate - can be more aggressive
            profile.performance_targets["min_confidence"] = max(0.6, profile.performance_targets.get("min_confidence", 0.7) - 0.05)
        
        if avg_response_time > profile.performance_targets.get("max_response_time", 3.0) * 1000:
            # Slow responses - adjust time targets
            profile.performance_targets["max_response_time"] = min(5.0, profile.performance_targets.get("max_response_time", 3.0) + 0.5)
        
        # Update confidence based on recent performance
        profile.confidence = profile.confidence * 0.8 + avg_confidence * 0.2
        
        self.logger.debug(f"Updated adaptation profile for {domain_id}: success_rate={success_rate:.3f}")
    
    def _get_quality_weights(self) -> Dict[str, float]:
        """
        Get quality assessment weights from data-driven configuration.
        
        Returns:
            Dict with similarity and success_rate weights
        """
        quality_config = self.data_driven_config.get("quality_weights")
        if quality_config and quality_config.get("calculation_source") == "quality_impact_analysis":
            return {
                "similarity": quality_config.get("SIMILARITY_WEIGHT", 0.6),
                "success_rate": quality_config.get("SUCCESS_RATE_WEIGHT", 0.4)
            }
        else:
            # Bootstrap fallback - should be replaced with real data
            return {
                "similarity": 0.6,  # TODO: Replace with data-driven calculation
                "success_rate": 0.4  # TODO: Replace with data-driven calculation
            }
    
    def _get_confidence_threshold(self, threshold_name: str = "low") -> float:
        """
        Get confidence threshold from data-driven configuration.
        
        Args:
            threshold_name: Name of threshold level (low, medium, high, very_high)
            
        Returns:
            Statistically derived confidence threshold
        """
        confidence_config = self.data_driven_config.get("confidence_levels")
        if confidence_config and confidence_config.get("data_source"):
            threshold_key = threshold_name.upper()
            return confidence_config.get(threshold_key, 0.3)  # Fallback to bootstrap
        else:
            # Bootstrap fallback values - should be replaced with real data  
            bootstrap_thresholds = {
                "low": 0.3,      # TODO: Replace with statistical calculation
                "medium": 0.5,   # TODO: Replace with statistical calculation  
                "high": 0.7,     # TODO: Replace with statistical calculation
                "very_high": 0.9 # TODO: Replace with statistical calculation
            }
            return bootstrap_thresholds.get(threshold_name, 0.3)


__all__ = [
    'ZeroConfigAdapter',
    'DomainDetectionResult',
    'AgentAdaptationProfile',
    'DomainAdaptationStrategy',
    'AdaptationConfidence'
]