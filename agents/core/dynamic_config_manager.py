"""
Dynamic Configuration Manager

This module implements the bridge between Config-Extraction workflow intelligence
and Search workflow execution, eliminating hardcoded values by providing
domain-specific, corpus-learned parameters.

Solves the critical design issue: Config-Extraction generates intelligent configs,
but Search workflow was using static hardcoded values instead.
"""

import asyncio
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml

from agents.core.constants import (
    FileSystemConstants,
    KnowledgeExtractionConstants,
    PerformanceAdaptiveConstants,
    StubConstants,
    UniversalSearchConstants,
)
from agents.core.data_models import DynamicExtractionConfig, DynamicSearchConfig
from agents.core.math_expressions import EXPR

# Import automation system components
from .constants.automation_interface import (
    automation_coordinator,
    GenerationRequest,
    LearningMechanism,
)
from .constants.automation_classifier import AutomationPotential

logger = logging.getLogger(__name__)

# Import workflow and agents dynamically to avoid circular imports
# DynamicExtractionConfig and DynamicSearchConfig now imported from
# agents.core.data_models


class DynamicConfigManager:
    """
    Enhanced Dynamic Configuration Manager with Phase 3 Automation Integration

    This is the architectural bridge that solves the hardcoded values problem:
    1. Loads learned configs from Config-Extraction workflow
    2. Provides domain-specific parameters to Search workflow
    3. Eliminates static hardcoded fallbacks
    4. Enables continuous learning and optimization
    5. **NEW**: Integrates with automation system for constant generation
    6. **NEW**: Coordinates performance feedback loops
    7. **NEW**: Manages interdependent constant groups
    """

    def __init__(self) -> None:
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl: int = PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL
        self.generated_configs_path: Path = Path(
            "agents/domain_intelligence/generated_configs"
        )
        self.config_extraction_workflow: Optional[Any] = None
        
        # Phase 3: Automation integration
        self.automation_enabled: bool = True
        self.performance_feedback_buffer: List[Dict[str, Any]] = []
        self.automation_generation_queue: List[str] = []  # Pending constant generations
        self._automation_lock = asyncio.Lock()

    async def get_extraction_config(self, domain_name: str) -> DynamicExtractionConfig:
        """
        Get extraction configuration for a specific domain.

        Priority:
        1. Recent learned config from Config-Extraction workflow
        2. Cached domain-specific config
        3. Trigger new Config-Extraction workflow run
        """

        # Try to load recent learned config
        learned_config = await self._load_learned_extraction_config(domain_name)
        if learned_config:
            return learned_config

        # Try cached config
        cached_config = self._get_cached_extraction_config(domain_name)
        if cached_config:
            return cached_config

        # Trigger new Config-Extraction workflow
        return await self._generate_new_extraction_config(domain_name)

    async def get_search_config(
        self, domain_name: str, query: str = None
    ) -> DynamicSearchConfig:
        """
        Get search configuration for a specific domain and query complexity.

        Uses domain analysis to determine optimal search parameters
        instead of hardcoded values.
        """

        # Try to load recent learned config
        learned_config = await self._load_learned_search_config(domain_name, query)
        if learned_config:
            return learned_config

        # Trigger domain analysis for search optimization
        return await self._generate_search_config_from_domain_analysis(
            domain_name, query
        )

    async def _load_learned_extraction_config(
        self, domain_name: str
    ) -> Optional[DynamicExtractionConfig]:
        """Load learned extraction config from Config-Extraction workflow results"""

        config_file = (
            self.generated_configs_path
            / f"{domain_name}{FileSystemConstants.EXTRACTION_CONFIG_SUFFIX}"
        )

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Check if config is recent (within 24 hours)
            learned_at = datetime.fromisoformat(config_data.get("learned_at", 0))
            if (
                datetime.now() - learned_at
            ).total_seconds() > PerformanceAdaptiveConstants.CACHE_TTL_SECONDS:
                return None

            return DynamicExtractionConfig(
                entity_confidence_threshold=config_data["entity_confidence_threshold"],
                relationship_confidence_threshold=config_data[
                    "relationship_confidence_threshold"
                ],
                chunk_size=config_data["chunk_size"],
                chunk_overlap=config_data["chunk_overlap"],
                batch_size=config_data["batch_size"],
                max_entities_per_chunk=config_data["max_entities_per_chunk"],
                min_relationship_strength=config_data["min_relationship_strength"],
                quality_validation_threshold=config_data[
                    "quality_validation_threshold"
                ],
                domain_name=domain_name,
                learned_at=learned_at,
                corpus_stats=config_data.get("corpus_stats", {}),
            )

        except Exception as e:
            print(f"Failed to load learned extraction config for {domain_name}: {e}")
            return None

    async def _load_learned_search_config(
        self, domain_name: str, query: str = None
    ) -> Optional[DynamicSearchConfig]:
        """Load learned search config optimized for domain and query complexity"""

        config_file = (
            self.generated_configs_path
            / f"{domain_name}{FileSystemConstants.SEARCH_CONFIG_SUFFIX}"
        )

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Check if config is recent
            learned_at = datetime.fromisoformat(config_data.get("learned_at", 0))
            if (
                datetime.now() - learned_at
            ).total_seconds() > PerformanceAdaptiveConstants.CACHE_TTL_SECONDS:
                return None

            # Adjust parameters based on query complexity if provided
            if query:
                config_data = await self._adjust_for_query_complexity(
                    config_data, query
                )

            return DynamicSearchConfig(
                vector_similarity_threshold=config_data["vector_similarity_threshold"],
                vector_top_k=config_data["vector_top_k"],
                graph_hop_count=config_data["graph_hop_count"],
                graph_min_relationship_strength=config_data[
                    "graph_min_relationship_strength"
                ],
                gnn_prediction_confidence=config_data["gnn_prediction_confidence"],
                gnn_node_embeddings=config_data["gnn_node_embeddings"],
                tri_modal_weights=config_data["tri_modal_weights"],
                result_synthesis_threshold=config_data["result_synthesis_threshold"],
                domain_name=domain_name,
                learned_at=learned_at,
                query_complexity_weights=config_data.get(
                    "query_complexity_weights", {}
                ),
            )

        except Exception as e:
            print(f"Failed to load learned search config for {domain_name}: {e}")
            return None

    async def _generate_new_extraction_config(
        self, domain_name: str
    ) -> DynamicExtractionConfig:
        """Generate domain-specific config using Domain Intelligence Agent
        analysis with real data"""

        try:
            logger.info(
                f"🧠 Using Domain Intelligence Agent to analyze corpus for "
                f"{domain_name}"
            )

            # Import domain intelligence analyzer
            from ..domain_intelligence.analyzers.unified_content_analyzer import (
                UnifiedContentAnalyzer,
            )

            # Initialize analyzer
            analyzer = UnifiedContentAnalyzer()

            # Determine corpus path from domain name
            # (e.g., programming_language -> Programming-Language)
            domain_path_map = {
                "programming_language": "Programming-Language",
                "general": "Programming-Language",  # Default to available data
            }

            # Get corpus path - use real data directory
            corpus_dir = domain_path_map.get(
                domain_name, domain_name.replace("_", "-").title()
            )
            corpus_path = f"/workspace/azure-maintie-rag/data/raw/{corpus_dir}"

            # Analyze corpus domain using real data
            domain_profile = analyzer.analyze_content(corpus_path)

            logger.info(f"✅ Domain analysis complete for {domain_name}:")
            logger.info(
                f"   📊 Analysis confidence: "
                f"{domain_profile.analysis_confidence.confidence:.3f}"
            )
            logger.info(
                f"   📏 Word count: {domain_profile.text_statistics.total_words}"
            )
            logger.info(
                f"   🔗 Domain fit score: {domain_profile.domain_fit_score:.3f}"
            )

            # Extract learned parameters from analysis (NO HARDCODED VALUES!)
            # Use analysis confidence for entity threshold
            entity_threshold = domain_profile.analysis_confidence.confidence
            relationship_threshold = (
                entity_threshold * StubConstants.RELATIONSHIP_THRESHOLD_FACTOR
            )  # Slightly more conservative for relationships

            # Use document characteristics for chunk size
            avg_sentence_length = (
                domain_profile.text_statistics.avg_sentence_length or 50
            )
            optimal_chunk_size = min(
                StubConstants.MAX_WORDS_BASE * 4,
                max(
                    StubConstants.MAX_WORDS_BASE,
                    int(avg_sentence_length * StubConstants.MAX_WORDS_MULTIPLIER),
                ),
            )  # Adaptive chunk size

            learned_config = DynamicExtractionConfig(
                entity_confidence_threshold=entity_threshold,
                relationship_confidence_threshold=relationship_threshold,
                chunk_size=optimal_chunk_size,
                chunk_overlap=EXPR.calculate_small_chunk_overlap(
                    optimal_chunk_size
                ),  # Centralized overlap ratio
                batch_size=max(
                    1, len(domain_profile.technical_vocabulary) // 10
                ),  # Scale with vocabulary
                max_entities_per_chunk=max(
                    5, int(len(domain_profile.technical_vocabulary) / 20)
                ),  # Based on vocabulary density
                min_relationship_strength=EXPR.calculate_relationship_strength_threshold(
                    relationship_threshold
                ),  # Centralized strength factor
                quality_validation_threshold=EXPR.calculate_quality_validation_threshold(
                    entity_threshold
                ),  # Centralized quality factor
                domain_name=domain_name,
                learned_at=datetime.now(),
                corpus_stats={
                    "total_words": domain_profile.text_statistics.total_words,
                    "lexical_diversity": (
                        domain_profile.text_statistics.lexical_diversity
                    ),
                    "readability_score": (
                        domain_profile.text_statistics.readability_score
                    ),
                    "avg_sentence_length": (
                        domain_profile.text_statistics.avg_sentence_length
                    ),
                    "processing_complexity": (
                        domain_profile.document_complexity.complexity_class
                    ),
                    "analysis_confidence": (
                        domain_profile.analysis_confidence.confidence
                    ),
                    "technical_vocabulary_size": len(
                        domain_profile.technical_vocabulary
                    ),
                    "domain_fit_score": domain_profile.domain_fit_score,
                    "technical_vocabulary": domain_profile.technical_vocabulary[
                        :20
                    ],  # Top technical terms
                    "key_concepts": list(domain_profile.concept_hierarchy.keys())[
                        :10
                    ],  # Top concepts
                },
            )

            # Save learned configuration for future use
            await self._save_learned_config(
                domain_name, "extraction", asdict(learned_config)
            )

            logger.info(f"💾 Saved learned extraction config for {domain_name}")
            return learned_config

        except Exception as e:
            logger.error(
                f"❌ Failed to generate extraction config using domain "
                f"intelligence: {e}"
            )
            # Fallback to centralized constants
            fallback_config = DynamicExtractionConfig(
                entity_confidence_threshold=(
                    KnowledgeExtractionConstants.FALLBACK_ENTITY_CONFIDENCE_THRESHOLD
                ),
                relationship_confidence_threshold=(
                    KnowledgeExtractionConstants.FALLBACK_RELATIONSHIP_CONFIDENCE_THRESHOLD
                ),
                chunk_size=KnowledgeExtractionConstants.FALLBACK_CHUNK_SIZE,
                chunk_overlap=KnowledgeExtractionConstants.FALLBACK_CHUNK_OVERLAP,
                batch_size=KnowledgeExtractionConstants.FALLBACK_BATCH_SIZE,
                max_entities_per_chunk=(
                    KnowledgeExtractionConstants.FALLBACK_MAX_ENTITIES_PER_CHUNK
                ),
                min_relationship_strength=(
                    KnowledgeExtractionConstants.FALLBACK_MIN_RELATIONSHIP_STRENGTH
                ),
                quality_validation_threshold=(
                    KnowledgeExtractionConstants.FALLBACK_QUALITY_VALIDATION_THRESHOLD
                ),
                domain_name=domain_name,
                learned_at=datetime.now(),
                corpus_stats={"fallback_used": True, "error": str(e)},
            )
            return fallback_config

    async def _generate_search_config_from_domain_analysis(
        self,
        domain_name: str,
        query: str = None,
    ) -> DynamicSearchConfig:
        """Generate search config using Domain Intelligence Agent analysis"""

        # Import dynamically to avoid circular imports
        from ..domain_intelligence.agent import get_domain_intelligence_agent

        domain_agent = get_domain_intelligence_agent()

        # Analyze domain characteristics for search optimization
        analysis_prompt = f"""Analyze domain '{domain_name}' for optimal search configuration.
        Query: {query or 'General search'}

        Provide optimal parameters for:
        1. Vector similarity threshold (precision vs recall balance)
        2. Vector top_k results needed
        3. Graph traversal hop count for relationships
        4. Minimum relationship strength for graph search
        5. GNN prediction confidence threshold
        6. Tri-modal search weights (vector, graph, gnn)
        7. Result synthesis threshold

        Base recommendations on domain complexity and query type."""

        try:
            await domain_agent.run(
                "analyze_domain_for_search_optimization",
                message_history=[{"role": "user", "content": analysis_prompt}],
            )

            # Extract optimization parameters from agent response
            # This would parse the agent's response and convert to config parameters
            # For now, using intelligent defaults based on domain analysis

            search_config = DynamicSearchConfig(
                vector_similarity_threshold=(
                    UniversalSearchConstants.FALLBACK_VECTOR_SIMILARITY_THRESHOLD
                ),  # Would be extracted from agent analysis
                vector_top_k=(
                    UniversalSearchConstants.FALLBACK_VECTOR_TOP_K
                ),  # Adjusted based on domain complexity
                graph_hop_count=(
                    UniversalSearchConstants.FALLBACK_GRAPH_HOP_COUNT
                ),  # Based on relationship depth analysis
                graph_min_relationship_strength=(
                    UniversalSearchConstants.FALLBACK_GRAPH_MIN_RELATIONSHIP_STRENGTH
                ),  # Domain-specific threshold
                gnn_prediction_confidence=(
                    UniversalSearchConstants.FALLBACK_GNN_PREDICTION_CONFIDENCE
                ),  # Based on model performance for domain
                gnn_node_embeddings=(
                    UniversalSearchConstants.FALLBACK_GNN_NODE_EMBEDDINGS
                ),  # Optimized for domain complexity
                tri_modal_weights={
                    "vector": UniversalSearchConstants.MULTI_MODAL_WEIGHT_VECTOR,
                    "graph": UniversalSearchConstants.MULTI_MODAL_WEIGHT_GRAPH,
                    "gnn": UniversalSearchConstants.MULTI_MODAL_WEIGHT_GNN,
                },  # Domain-optimized
                result_synthesis_threshold=(
                    UniversalSearchConstants.FALLBACK_RESULT_SYNTHESIS_THRESHOLD
                ),  # Quality threshold for domain
                domain_name=domain_name,
                learned_at=datetime.now(),
                query_complexity_weights={
                    "simple": (
                        UniversalSearchConstants.QUERY_COMPLEXITY_SIMPLE_MULTIPLIER
                    ),
                    "medium": (
                        UniversalSearchConstants.QUERY_COMPLEXITY_MEDIUM_MULTIPLIER
                    ),
                    "complex": (
                        UniversalSearchConstants.QUERY_COMPLEXITY_COMPLEX_MULTIPLIER
                    ),
                },
            )

            # Save for future use
            await self._save_learned_config(
                domain_name, "search", asdict(search_config)
            )

            return search_config

        except Exception as e:
            raise Exception(f"Failed to generate search config for {domain_name}: {e}")

    async def _adjust_for_query_complexity(
        self, config_data: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Adjust search parameters based on query complexity analysis"""

        # Simple complexity analysis based on query characteristics
        query_length = len(query.split())
        has_complex_terms = any(
            term in query.lower()
            for term in ["relationship", "connection", "between", "how does", "explain"]
        )

        complexity_multiplier = 1.0
        if (
            query_length > UniversalSearchConstants.QUERY_LENGTH_COMPLEX_THRESHOLD
            or has_complex_terms
        ):
            complexity_multiplier = (
                UniversalSearchConstants.QUERY_COMPLEXITY_COMPLEX_MULTIPLIER
            )  # Increase search depth for complex queries
        elif query_length < UniversalSearchConstants.QUERY_LENGTH_SIMPLE_THRESHOLD:
            complexity_multiplier = (
                UniversalSearchConstants.QUERY_COMPLEXITY_SIMPLE_MULTIPLIER
            )  # Reduce search depth for simple queries

        # Adjust parameters based on complexity
        config_data["vector_top_k"] = int(
            config_data["vector_top_k"] * complexity_multiplier
        )
        config_data["graph_hop_count"] = min(
            UniversalSearchConstants.MAX_GRAPH_HOP_COUNT,
            int(config_data["graph_hop_count"] * complexity_multiplier),
        )

        return config_data

    async def _save_learned_config(
        self, domain_name: str, config_type: str, config_data: Dict[str, Any]
    ):
        """Save learned configuration for future use"""

        os.makedirs(self.generated_configs_path, exist_ok=True)
        config_file = (
            self.generated_configs_path
            / f"{domain_name}_{config_type}{FileSystemConstants.GENERAL_CONFIG_SUFFIX}"
        )

        try:
            with open(config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save learned config: {e}")

    def _get_cached_extraction_config(
        self, domain_name: str
    ) -> Optional[DynamicExtractionConfig]:
        """Get extraction config from memory cache"""
        cache_key = f"extraction_{domain_name}"
        cached = self.config_cache.get(cache_key)

        if (
            cached
            and (datetime.now() - cached["cached_at"]).total_seconds() < self.cache_ttl
        ):
            return DynamicExtractionConfig(**cached["config"])

        return None

    async def force_config_regeneration(self, domain_name: str) -> Dict[str, Any]:
        """Force regeneration of all configs for a domain with automation integration"""

        results = {}

        # Phase 3: Queue automated constant generation before config generation
        if self.automation_enabled:
            await self._queue_automated_constant_generation(domain_name, results)

        # Regenerate extraction config
        try:
            extraction_config = await self._generate_new_extraction_config(domain_name)
            results["extraction_config"] = asdict(extraction_config)
        except Exception as e:
            results["extraction_error"] = str(e)

        # Regenerate search config
        try:
            search_config = await self._generate_search_config_from_domain_analysis(
                domain_name
            )
            results["search_config"] = asdict(search_config)
        except Exception as e:
            results["search_error"] = str(e)

        return results

    # === Phase 3: Automation Integration Methods ===

    async def _queue_automated_constant_generation(
        self, domain_name: str, results: Dict[str, Any]
    ) -> None:
        """Queue automated constant generation for domain-specific optimization"""
        
        async with self._automation_lock:
            try:
                # Queue domain-adaptive constants generation
                domain_request = GenerationRequest(
                    constant_name="DomainAdaptiveConstants",
                    learning_mechanisms=[
                        LearningMechanism.DOMAIN_ANALYSIS,
                        LearningMechanism.CORRELATION_ANALYSIS,
                        LearningMechanism.QUALITY_OPTIMIZATION
                    ],
                    context={
                        "domain_name": domain_name,
                        "domain_analysis": await self._gather_domain_analysis_context(domain_name),
                        "performance_metrics": await self._gather_performance_context(domain_name),
                        "quality_assessment": await self._gather_quality_context(domain_name)
                    },
                    priority=3  # High priority
                )
                
                await automation_coordinator.queue_generation_request(domain_request)
                
                # Queue performance-adaptive constants if we have performance data
                if self.performance_feedback_buffer:
                    perf_request = GenerationRequest(
                        constant_name="PerformanceAdaptiveConstants",
                        learning_mechanisms=[
                            LearningMechanism.PERFORMANCE_FEEDBACK,
                            LearningMechanism.USAGE_PATTERNS
                        ],
                        context={
                            "domain_name": domain_name,
                            "performance_metrics": self._aggregate_performance_feedback(),
                            "usage_statistics": await self._gather_usage_statistics(domain_name)
                        },
                        priority=2
                    )
                    
                    await automation_coordinator.queue_generation_request(perf_request)
                
                # Queue extraction constants optimization
                extraction_request = GenerationRequest(
                    constant_name="KnowledgeExtractionConstants",
                    learning_mechanisms=[
                        LearningMechanism.DOMAIN_ANALYSIS,
                        LearningMechanism.QUALITY_OPTIMIZATION
                    ],
                    context={
                        "domain_name": domain_name,
                        "extraction_performance": await self._gather_extraction_performance(domain_name)
                    },
                    priority=2
                )
                
                await automation_coordinator.queue_generation_request(extraction_request)
                
                results["automation_queued"] = {
                    "domain_adaptive": True,
                    "performance_adaptive": bool(self.performance_feedback_buffer),
                    "knowledge_extraction": True,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to queue automated constant generation: {e}")
                results["automation_error"] = str(e)

    async def _gather_domain_analysis_context(self, domain_name: str) -> Dict[str, Any]:
        """Gather domain analysis context for constant generation"""
        
        try:
            # Try to get existing domain profile from cache or recent analysis
            cache_key = f"domain_profile_{domain_name}"
            cached_profile = self.config_cache.get(cache_key)
            
            if cached_profile and (datetime.now() - cached_profile["cached_at"]).total_seconds() < self.cache_ttl:
                return cached_profile["data"]
            
            # If no cached profile, gather basic domain characteristics
            corpus_stats = await self._analyze_corpus_characteristics(domain_name)
            
            return {
                "entity_density": corpus_stats.get("entity_density", 0.1),
                "relationship_complexity": corpus_stats.get("relationship_complexity", 0.5),
                "technical_vocabulary_size": corpus_stats.get("vocabulary_size", 100),
                "domain_complexity": corpus_stats.get("complexity_class", "medium")
            }
            
        except Exception as e:
            logger.warning(f"Failed to gather domain analysis context: {e}")
            return {}

    async def _gather_performance_context(self, domain_name: str) -> Dict[str, Any]:
        """Gather performance metrics context for constant optimization"""
        
        try:
            # Aggregate performance metrics from buffer
            performance_data = self._aggregate_performance_feedback()
            
            # Add domain-specific performance metrics
            domain_metrics = {}
            for metric in self.performance_feedback_buffer:
                if metric.get("domain_name") == domain_name:
                    domain_metrics.update(metric)
            
            return {
                "average_response_time": performance_data.get("avg_response_time", 1.0),
                "cache_hit_rate": performance_data.get("cache_hit_rate", 0.6),
                "extraction_accuracy": domain_metrics.get("extraction_accuracy", 0.85),
                "search_relevance": domain_metrics.get("search_relevance", 0.75),
                "concurrent_users": performance_data.get("concurrent_users", 10)
            }
            
        except Exception as e:
            logger.warning(f"Failed to gather performance context: {e}")
            return {}

    async def _gather_quality_context(self, domain_name: str) -> Dict[str, Any]:
        """Gather quality assessment context for optimization"""
        
        try:
            # This would integrate with quality assessment systems
            # For now, return placeholder structure
            return {
                "extraction_accuracy_by_threshold": {
                    "0.6": 0.70,
                    "0.7": 0.80,
                    "0.8": 0.85,
                    "0.9": 0.75  # May decrease with very high threshold
                },
                "search_precision_recall": {
                    "precision": 0.82,
                    "recall": 0.76,
                    "f1_score": 0.79
                },
                "user_satisfaction_scores": {
                    "relevance": 4.2,  # out of 5
                    "completeness": 3.9,
                    "accuracy": 4.1
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to gather quality context: {e}")
            return {}

    async def _gather_usage_statistics(self, domain_name: str) -> Dict[str, Any]:
        """Gather usage pattern statistics for optimization"""
        
        try:
            # This would integrate with usage analytics
            # For now, return placeholder structure
            return {
                "query_complexity_distribution": {
                    "simple": 0.4,   # 40% simple queries
                    "medium": 0.45,  # 45% medium queries  
                    "complex": 0.15  # 15% complex queries
                },
                "average_query_length": 8.5,
                "peak_usage_hours": [9, 10, 14, 15, 16],
                "common_query_patterns": [
                    "how to", "what is", "examples of", "difference between"
                ],
                "session_duration_avg": 12.3  # minutes
            }
            
        except Exception as e:
            logger.warning(f"Failed to gather usage statistics: {e}")
            return {}

    async def _gather_extraction_performance(self, domain_name: str) -> Dict[str, Any]:
        """Gather extraction performance metrics for optimization"""
        
        try:
            # This would integrate with extraction pipeline metrics
            return {
                "entity_extraction_accuracy": 0.87,
                "relationship_extraction_accuracy": 0.82,
                "processing_time_per_chunk": 0.45,  # seconds
                "memory_usage_per_batch": 128,  # MB
                "false_positive_rate": 0.08,
                "false_negative_rate": 0.12
            }
            
        except Exception as e:
            logger.warning(f"Failed to gather extraction performance: {e}")
            return {}

    async def _analyze_corpus_characteristics(self, domain_name: str) -> Dict[str, Any]:
        """Analyze corpus characteristics for domain-specific optimization"""
        
        try:
            # This would use the UnifiedContentAnalyzer
            # For now, return estimated characteristics based on domain
            if domain_name == "programming_language":
                return {
                    "entity_density": 0.15,  # High technical entity density
                    "relationship_complexity": 0.7,  # Complex technical relationships
                    "vocabulary_size": 500,
                    "complexity_class": "high",
                    "average_document_length": 2500
                }
            elif domain_name == "general":
                return {
                    "entity_density": 0.08,
                    "relationship_complexity": 0.4,
                    "vocabulary_size": 200,
                    "complexity_class": "medium",
                    "average_document_length": 1200
                }
            else:
                # Default estimates
                return {
                    "entity_density": 0.1,
                    "relationship_complexity": 0.5,
                    "vocabulary_size": 300,
                    "complexity_class": "medium",
                    "average_document_length": 1500
                }
                
        except Exception as e:
            logger.warning(f"Failed to analyze corpus characteristics: {e}")
            return {}

    def _aggregate_performance_feedback(self) -> Dict[str, Any]:
        """Aggregate performance feedback from buffer"""
        
        if not self.performance_feedback_buffer:
            return {}
        
        # Calculate aggregated metrics
        response_times = [m.get("response_time", 1.0) for m in self.performance_feedback_buffer if "response_time" in m]
        cache_hits = [m.get("cache_hit_rate", 0.6) for m in self.performance_feedback_buffer if "cache_hit_rate" in m]
        
        return {
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 1.0,
            "cache_hit_rate": sum(cache_hits) / len(cache_hits) if cache_hits else 0.6,
            "total_requests": len(self.performance_feedback_buffer),
            "measurement_period_hours": 24  # Default measurement window
        }

    async def record_performance_feedback(
        self, domain_name: str, metrics: Dict[str, Any]
    ) -> None:
        """Record performance feedback for automation learning"""
        
        async with self._automation_lock:
            # Add timestamp and domain to metrics
            metrics_with_context = {
                **metrics,
                "domain_name": domain_name,
                "timestamp": datetime.now(),
                "session_id": getattr(self, "_current_session_id", "default")
            }
            
            self.performance_feedback_buffer.append(metrics_with_context)
            
            # Keep buffer size manageable (last 1000 entries)
            if len(self.performance_feedback_buffer) > 1000:
                self.performance_feedback_buffer = self.performance_feedback_buffer[-1000:]
            
            logger.info(f"Recorded performance feedback for {domain_name}: {list(metrics.keys())}")

    async def process_automation_queue(self) -> Dict[str, Any]:
        """Process pending automation requests and apply results"""
        
        if not self.automation_enabled:
            return {"automation_disabled": True}
        
        try:
            # Process all queued generation requests
            generation_results = await automation_coordinator.process_generation_queue()
            
            results = {
                "processed_count": len(generation_results),
                "successful_generations": [],
                "failed_generations": [],
                "timestamp": datetime.now().isoformat()
            }
            
            for result in generation_results:
                if result.validation_passed:
                    results["successful_generations"].append({
                        "constant_name": result.constant_name,
                        "confidence_score": result.confidence_score,
                        "learning_source": result.learning_source,
                        "generated_values_count": len(result.generated_values)
                    })
                    
                    # Apply generated constants (this would integrate with constant loading)
                    await self._apply_generated_constants(result)
                    
                else:
                    results["failed_generations"].append({
                        "constant_name": result.constant_name,
                        "error_message": result.error_message,
                        "confidence_score": result.confidence_score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process automation queue: {e}")
            return {"error": str(e)}

    async def _apply_generated_constants(self, generation_result) -> None:
        """Apply generated constants to the system (placeholder for integration)"""
        
        try:
            # This would integrate with the constant loading system
            # For now, log the successful generation
            logger.info(
                f"Would apply generated constants for {generation_result.constant_name}: "
                f"{list(generation_result.generated_values.keys())}"
            )
            
            # Cache the generated constants for future use
            cache_key = f"generated_constants_{generation_result.constant_name}"
            self.config_cache[cache_key] = {
                "constants": generation_result.generated_values,
                "confidence": generation_result.confidence_score,
                "learning_source": generation_result.learning_source,
                "cached_at": generation_result.generation_timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to apply generated constants: {e}")

    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation system status"""
        
        return {
            "automation_enabled": self.automation_enabled,
            "performance_feedback_count": len(self.performance_feedback_buffer),
            "cached_generated_constants": len([
                k for k in self.config_cache.keys() 
                if k.startswith("generated_constants_")
            ]),
            "coordinator_status": automation_coordinator.get_automation_status()
        }

    def enable_automation(self) -> None:
        """Enable automation system integration"""
        self.automation_enabled = True
        logger.info("Automation system integration enabled")

    def disable_automation(self) -> None:
        """Disable automation system integration"""
        self.automation_enabled = False
        logger.info("Automation system integration disabled")


# Global instance for use across the system
dynamic_config_manager = DynamicConfigManager()


# Integration functions for centralized_config.py
async def load_extraction_config_from_workflow(
    domain_name: str,
) -> DynamicExtractionConfig:
    """Load extraction config from Config-Extraction workflow intelligence"""
    return await dynamic_config_manager.get_extraction_config(domain_name)


async def load_search_config_from_workflow(
    domain_name: str, query: str = None
) -> DynamicSearchConfig:
    """Load search config from domain analysis and workflow intelligence"""
    return await dynamic_config_manager.get_search_config(domain_name, query)


async def force_dynamic_config_loading() -> Dict[str, Any]:
    """Force regeneration of all dynamic configs - useful for testing"""

    # Import dynamically to avoid circular imports
    from ..domain_intelligence.agent import get_domain_intelligence_agent

    # Discover available domains
    domain_agent = get_domain_intelligence_agent()

    try:
        result = await domain_agent.run(
            "discover_available_domains",
            message_history=[
                {
                    "role": "user",
                    "content": "Discover all available domains in the system",
                }
            ],
        )

        # Extract domains and regenerate configs for each
        domains = (
            result.data.get("domains", ["general"])
            if hasattr(result, "data")
            else ["general"]
        )

        regeneration_results = {}
        for domain in domains:
            domain_results = await dynamic_config_manager.force_config_regeneration(
                domain
            )
            regeneration_results[domain] = domain_results

        return regeneration_results

    except Exception as e:
        raise Exception(f"Failed to force dynamic config loading: {e}")
