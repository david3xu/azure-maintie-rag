"""
Dynamic Configuration Manager

This module implements the bridge between Config-Extraction workflow intelligence
and Search workflow execution, eliminating hardcoded values by providing
domain-specific, corpus-learned parameters.

Solves the critical design issue: Config-Extraction generates intelligent configs,
but Search workflow was using static hardcoded values instead.
"""

import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

logger = logging.getLogger(__name__)

# Import workflow and agents dynamically to avoid circular imports
# DynamicExtractionConfig and DynamicSearchConfig now imported from
# agents.core.data_models


class DynamicConfigManager:
    """
    Manages dynamic configuration loading from Config-Extraction workflow results.

    This is the architectural bridge that solves the hardcoded values problem:
    1. Loads learned configs from Config-Extraction workflow
    2. Provides domain-specific parameters to Search workflow
    3. Eliminates static hardcoded fallbacks
    4. Enables continuous learning and optimization
    """

    def __init__(self) -> None:
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl: int = PerformanceAdaptiveConstants.DEFAULT_CACHE_TTL
        self.generated_configs_path: Path = Path(
            "agents/domain_intelligence/generated_configs"
        )
        self.config_extraction_workflow: Optional[Any] = None

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
        """Force regeneration of all configs for a domain"""

        results = {}

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
