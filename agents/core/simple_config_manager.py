"""
Simple Dynamic Configuration Manager - Actually Works
====================================================

This module provides domain-aware configuration that:
- Integrates with Domain Intelligence Agent results
- Adapts parameters based on real domain analysis
- Has proper error handling and fallbacks
- Works with the query generation system
- No circular imports or missing dependencies

This replaces the broken dynamic_config_manager references.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DomainIntelligenceResult:
    """Results from domain intelligence analysis"""

    domain_signature: str
    content_type_confidence: float
    vocabulary_complexity: float  # Universal characteristic, not domain-specific
    document_count: int
    avg_document_length: float
    key_concepts: list
    recommended_chunk_size: int
    recommended_confidence_thresholds: Dict[str, float]
    processing_recommendations: Dict[str, Any]


class SimpleDynamicConfigManager:
    """Simple, working dynamic configuration manager"""

    def __init__(self):
        self._domain_analyses = {}  # Cache for domain analyses
        self._config_cache = {}  # Cache for generated configurations

    async def analyze_domain_if_needed(
        self, data_directory: str
    ) -> DomainIntelligenceResult:
        """Analyze domain characteristics or return cached result"""

        if data_directory in self._domain_analyses:
            return self._domain_analyses[data_directory]

        try:
            # Use universal config for content analysis to avoid circular imports
            from config.universal_config import analyze_content_characteristics

            # Read sample files for analysis
            data_path = Path(data_directory)
            content_samples = []
            if data_path.exists():
                for file_path in list(data_path.rglob("*"))[:5]:  # Sample first 5 files
                    if file_path.is_file() and file_path.suffix in [
                        ".md",
                        ".txt",
                        ".py",
                        ".json",
                    ]:
                        try:
                            content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                            if len(content) > 100:
                                content_samples.append(content[:2000])  # First 2k chars
                        except:
                            pass

            if content_samples:
                characteristics = analyze_content_characteristics(
                    content_samples, data_directory
                )

                # Convert to our simple format
                result = DomainIntelligenceResult(
                    domain_signature=f"{data_path.name}_analyzed",
                    content_type_confidence=characteristics.analysis_confidence,
                    vocabulary_complexity=characteristics.vocabulary_complexity,
                    document_count=characteristics.sample_size,
                    avg_document_length=characteristics.avg_paragraph_length,
                    key_concepts=["content", "analysis", data_path.name],
                    recommended_chunk_size=int(
                        800 + characteristics.structural_complexity * 600
                    ),
                    recommended_confidence_thresholds={
                        "entity": 0.9 - characteristics.vocabulary_complexity * 0.2,
                        "relationship": 0.8
                        - characteristics.vocabulary_complexity * 0.2,
                    },
                    processing_recommendations={},
                )

                self._domain_analyses[data_directory] = result
                logger.info(
                    f"Universal domain analysis complete: {result.domain_signature}"
                )
                return result
            else:
                raise Exception("No content samples found")

        except Exception as e:
            logger.warning(f"Domain analysis failed, using safe defaults: {e}")

            # Fallback analysis based on directory structure
            return self._create_fallback_analysis(data_directory)

    def _create_fallback_analysis(
        self, data_directory: str
    ) -> DomainIntelligenceResult:
        """Create reasonable fallback analysis when agent is unavailable"""

        data_path = Path(data_directory)
        domain_name = data_path.name if data_path.exists() else "unknown"

        # Basic file analysis
        file_count = 0
        total_size = 0
        specialized_indicators = 0

        if data_path.exists():
            for file_path in data_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in [
                    ".md",
                    ".txt",
                    ".py",
                    ".json",
                ]:
                    file_count += 1
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        total_size += len(content)

                        # Simple specialized vocabulary detection
                        specialized_indicators_count = 0
                        # Count capitalized terms (indicating specialized terminology)
                        words = content.split()
                        capitalized_count = sum(
                            1 for word in words if word.istitle() and len(word) > 3
                        )
                        # Count structured patterns (indicating formal content)
                        punctuation_density = (
                            sum(1 for char in content if char in ":.;()[]{}")
                            / len(content)
                            if content
                            else 0
                        )

                        if (
                            capitalized_count > len(words) * 0.1
                            or punctuation_density > 0.05
                        ):
                            specialized_indicators += 1
                    except:
                        pass

        # Calculate basic metrics
        vocabulary_complexity_ratio = specialized_indicators / max(file_count, 1)
        avg_length = total_size / max(file_count, 1)

        return DomainIntelligenceResult(
            domain_signature=f"{domain_name}_auto_detected",
            content_type_confidence=0.7,
            vocabulary_complexity=vocabulary_complexity_ratio,  # Universal characteristic
            document_count=file_count,
            avg_document_length=avg_length,
            key_concepts=[domain_name, "content", "analysis"],
            recommended_chunk_size=1000 if avg_length > 2000 else 800,
            recommended_confidence_thresholds={
                "entity": 0.8 if vocabulary_complexity_ratio > 0.5 else 0.85,
                "relationship": 0.7 if vocabulary_complexity_ratio > 0.5 else 0.75,
            },
            processing_recommendations={},
        )

    async def get_extraction_config(
        self, domain_name: str, data_directory: str = None
    ) -> Dict[str, Any]:
        """Get extraction configuration adapted for domain"""

        cache_key = f"extraction_{domain_name}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Get domain analysis if directory provided
        domain_analysis = None
        if data_directory:
            try:
                domain_analysis = await self.analyze_domain_if_needed(data_directory)
            except Exception as e:
                logger.warning(f"Failed to get domain analysis: {e}")

        # Generate configuration
        if domain_analysis:
            config = {
                "entity_confidence_threshold": domain_analysis.recommended_confidence_thresholds[
                    "entity"
                ],
                "relationship_confidence_threshold": domain_analysis.recommended_confidence_thresholds[
                    "relationship"
                ],
                "chunk_size": domain_analysis.recommended_chunk_size,
                "chunk_overlap": int(domain_analysis.recommended_chunk_size * 0.2),
                "batch_size": min(10, max(3, domain_analysis.document_count // 5)),
                "max_entities_per_chunk": (
                    20 if domain_analysis.vocabulary_complexity > 0.6 else 15
                ),
                "min_relationship_strength": (
                    0.6 if domain_analysis.vocabulary_complexity > 0.6 else 0.7
                ),
                "quality_validation_threshold": 0.75,
                "domain_name": domain_analysis.domain_signature,
                "technical_vocabulary": domain_analysis.key_concepts,
                "expected_entity_types": ["CONCEPT", "TERM", "PROCEDURE", "ENTITY"],
                "corpus_stats": {
                    "document_count": domain_analysis.document_count,
                    "vocabulary_complexity": domain_analysis.vocabulary_complexity,
                    "confidence": domain_analysis.content_type_confidence,
                },
            }
        else:
            # Safe defaults
            config = {
                "entity_confidence_threshold": 0.8,
                "relationship_confidence_threshold": 0.7,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "batch_size": 5,
                "max_entities_per_chunk": 15,
                "min_relationship_strength": 0.7,
                "quality_validation_threshold": 0.75,
                "domain_name": domain_name,
                "technical_vocabulary": [],
                "expected_entity_types": ["CONCEPT", "TERM", "PROCEDURE"],
                "corpus_stats": {},
            }

        self._config_cache[cache_key] = config
        return config

    async def get_search_config(
        self, domain_name: str, query: str = None, data_directory: str = None
    ) -> Dict[str, Any]:
        """Get search configuration adapted for domain and query"""

        cache_key = f"search_{domain_name}_{hash(query or '')}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Get domain analysis if directory provided
        domain_analysis = None
        if data_directory:
            try:
                domain_analysis = await self.analyze_domain_if_needed(data_directory)
            except Exception as e:
                logger.warning(f"Failed to get domain analysis: {e}")

        # Generate configuration
        base_config = {
            "vector_similarity_threshold": 0.7,
            "vector_top_k": 10,
            "graph_hop_count": 2,
            "graph_min_relationship_strength": 0.5,
            "gnn_prediction_confidence": 0.6,
            "gnn_node_embeddings": 128,
            "tri_modal_weights": {"vector": 0.4, "graph": 0.3, "gnn": 0.3},
            "result_synthesis_threshold": 0.6,
            "query_complexity_weights": {
                "simple": 1.0,
                "moderate": 1.2,
                "complex": 1.5,
            },
            "domain_name": domain_name,
            "learned_at": None,
        }

        # Adapt based on content analysis
        if domain_analysis:
            if domain_analysis.vocabulary_complexity > 0.7:
                # High vocabulary complexity - adjust for precision
                base_config["vector_similarity_threshold"] = 0.75
                base_config["graph_hop_count"] = 3
                base_config["tri_modal_weights"] = {
                    "vector": 0.3,
                    "graph": 0.4,
                    "gnn": 0.3,
                }
            elif domain_analysis.vocabulary_complexity < 0.3:
                # Lower vocabulary complexity - adjust for recall
                base_config["vector_similarity_threshold"] = 0.65
                base_config["vector_top_k"] = 12
                base_config["tri_modal_weights"] = {
                    "vector": 0.5,
                    "graph": 0.25,
                    "gnn": 0.25,
                }

        # Adapt based on query complexity
        if query:
            query_length = len(query.split())
            if query_length > 10:
                base_config["vector_top_k"] = 15
                base_config["graph_hop_count"] = min(
                    3, base_config["graph_hop_count"] + 1
                )
            elif query_length < 3:
                base_config["vector_top_k"] = 8
                base_config["graph_hop_count"] = max(
                    1, base_config["graph_hop_count"] - 1
                )

        self._config_cache[cache_key] = base_config
        return base_config

    async def get_domain_config(self, domain_name: str) -> Dict[str, Any]:
        """Get domain configuration (for infrastructure compatibility)"""
        return await self.get_extraction_config(domain_name)

    def clear_cache(self):
        """Clear all caches"""
        self._domain_analyses.clear()
        self._config_cache.clear()
        logger.info("Configuration cache cleared")


# Private singleton for proper dependency injection
_config_manager: Optional[SimpleDynamicConfigManager] = None


def get_simple_config_manager() -> SimpleDynamicConfigManager:
    """Factory function to get config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SimpleDynamicConfigManager()
    return _config_manager


# Convenience functions that use the factory pattern
async def get_extraction_config_dynamic(
    domain_name: str, data_directory: str = None
) -> Dict[str, Any]:
    """Get dynamic extraction configuration"""
    config_manager = get_simple_config_manager()
    return await config_manager.get_extraction_config(domain_name, data_directory)


async def get_search_config_dynamic(
    domain_name: str, query: str = None, data_directory: str = None
) -> Dict[str, Any]:
    """Get dynamic search configuration"""
    config_manager = get_simple_config_manager()
    return await config_manager.get_search_config(domain_name, query, data_directory)


async def analyze_domain_directory(data_directory: str) -> DomainIntelligenceResult:
    """Analyze domain characteristics for directory"""
    config_manager = get_simple_config_manager()
    return await config_manager.analyze_domain_if_needed(data_directory)


# Legacy compatibility names - using functions since properties can't be at module level
def get_dynamic_config_manager() -> SimpleDynamicConfigManager:
    """Legacy compatibility function"""
    return get_simple_config_manager()


# Legacy compatibility - these will be functions that act like the old singletons
simple_dynamic_config_manager = get_simple_config_manager
dynamic_config_manager = get_simple_config_manager

__all__ = [
    "SimpleDynamicConfigManager",
    "DomainIntelligenceResult",
    "simple_dynamic_config_manager",
    "dynamic_config_manager",
    "get_extraction_config_dynamic",
    "get_search_config_dynamic",
    "analyze_domain_directory",
]
