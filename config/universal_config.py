"""
Universal Configuration System - Zero Domain Bias
================================================

This module provides truly universal configuration that:
- DISCOVERS characteristics from content analysis
- ADAPTS parameters based on measured properties
- AVOIDS predetermined domain categories
- USES content-agnostic patterns that work for ANY domain
- LEARNS from data, never assumes domain knowledge

No hardcoded domain types, no predetermined categories, no domain assumptions.
Pure content-driven adaptation following Universal RAG philosophy.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Environment-based Azure configuration (infrastructure only)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
OPENAI_MODEL_DEPLOYMENT = os.getenv("OPENAI_MODEL_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini"))


@dataclass
class SystemConfig:
    """Infrastructure configuration - no domain assumptions"""

    # Resource limits (infrastructure constraints)
    max_workers: int = 4
    max_concurrent_requests: int = 5
    max_batch_size: int = 10

    # Timeouts (network infrastructure)
    openai_timeout: int = 60
    search_timeout: int = 30
    cosmos_timeout: int = 45
    max_retries: int = 3

    # Security limits (infrastructure security)
    max_query_length: int = 1000
    max_execution_time: float = 300.0
    max_document_size_mb: int = 50

    # Performance (system performance)
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    def __post_init__(self):
        # Apply environment overrides
        self.max_workers = int(os.getenv("MAX_WORKERS", self.max_workers))
        self.max_concurrent_requests = int(
            os.getenv("MAX_CONCURRENT_REQUESTS", self.max_concurrent_requests)
        )
        self.openai_timeout = int(os.getenv("OPENAI_TIMEOUT", self.openai_timeout))


@dataclass
class ModelConfig:
    """Azure OpenAI model configuration - infrastructure only"""

    # Model deployments (infrastructure)
    primary_model: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"))
    # NO FALLBACK MODELS - Production uses primary model only
    embedding_model: str = "text-embedding-ada-002"

    # Azure configuration (infrastructure)
    endpoint: str = AZURE_OPENAI_ENDPOINT
    api_key: Optional[str] = AZURE_OPENAI_API_KEY
    api_version: str = AZURE_OPENAI_API_VERSION
    deployment_name: str = OPENAI_MODEL_DEPLOYMENT

    # Model parameters (universal)
    temperature: float = 0.0
    max_tokens: int = 4000

    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured"""
        return bool(self.endpoint and self.api_key)


@dataclass
class ContentCharacteristics:
    """Discovered characteristics from content analysis - no predetermined domains"""

    # Measured complexity (0.0 to 1.0)
    vocabulary_complexity: float = 0.5
    concept_density: float = 0.5
    relationship_complexity: float = 0.5
    structural_complexity: float = 0.5

    # Measured patterns (counts and ratios)
    avg_sentence_length: float = 20.0
    avg_paragraph_length: float = 100.0
    terminology_uniqueness: float = 0.5  # Ratio of unique terms
    cross_reference_ratio: float = 0.3  # Ratio of internal references

    # Content properties (discovered)
    content_signature: str = "unknown"
    primary_language: str = "en"
    measured_text_complexity_level: int = 8  # Measured from content analysis

    # Statistical confidence
    analysis_confidence: float = 0.0
    sample_size: int = 0


@dataclass
class UniversalExtractionConfig:
    """Knowledge extraction configuration based on discovered characteristics"""

    # Adaptive thresholds based on content complexity
    entity_confidence_threshold: float = 0.65
    relationship_confidence_threshold: float = 0.6

    # Adaptive chunking based on content structure
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_entities_per_chunk: int = 15
    max_relationships_per_chunk: int = 10

    # Quality thresholds (universal constraints)
    min_entity_length: int = 2
    max_entity_length: int = 100
    quality_threshold: float = 0.7

    # Content characteristics (discovered)
    characteristics: ContentCharacteristics = field(
        default_factory=ContentCharacteristics
    )

    def adapt_to_content(
        self, characteristics: ContentCharacteristics
    ) -> "UniversalExtractionConfig":
        """Adapt configuration based on discovered content characteristics"""
        adapted = UniversalExtractionConfig(**self.__dict__)
        adapted.characteristics = characteristics

        # Adapt thresholds based on vocabulary complexity
        complexity_factor = characteristics.vocabulary_complexity
        adapted.entity_confidence_threshold = 0.9 - (complexity_factor * 0.2)
        adapted.relationship_confidence_threshold = 0.8 - (complexity_factor * 0.2)

        # Adapt chunking based on structural complexity
        structure_factor = characteristics.structural_complexity
        base_chunk_size = 800 + int(structure_factor * 600)  # 800-1400 range
        adapted.chunk_size = base_chunk_size
        adapted.chunk_overlap = int(base_chunk_size * 0.15)  # 15% overlap

        # Adapt entity limits based on concept density
        density_factor = characteristics.concept_density
        adapted.max_entities_per_chunk = int(10 + (density_factor * 15))  # 10-25 range
        adapted.max_relationships_per_chunk = int(
            8 + (density_factor * 12)
        )  # 8-20 range

        return adapted


@dataclass
class UniversalSearchConfig:
    """Universal search configuration based on discovered characteristics"""

    # Adaptive similarity thresholds
    vector_similarity_threshold: float = 0.7
    graph_relationship_threshold: float = 0.5

    # Adaptive search parameters
    vector_top_k: int = 10
    graph_traversal_depth: int = 2
    max_results: int = 50

    # Multi-modal weights (adaptive)
    vector_weight: float = 0.4
    graph_weight: float = 0.3
    hybrid_weight: float = 0.3

    # Result synthesis
    result_synthesis_threshold: float = 0.6

    # Content characteristics (discovered)
    characteristics: ContentCharacteristics = field(
        default_factory=ContentCharacteristics
    )

    def adapt_to_query_and_content(
        self,
        query_characteristics: Dict[str, float],
        content_characteristics: ContentCharacteristics,
    ) -> "UniversalSearchConfig":
        """Adapt search configuration based on query and content characteristics"""
        adapted = UniversalSearchConfig(**self.__dict__)
        adapted.characteristics = content_characteristics

        # Adapt based on query complexity
        query_complexity = query_characteristics.get("complexity", 0.5)
        query_length_factor = min(query_characteristics.get("length_factor", 1.0), 2.0)

        # Adjust search depth based on query complexity
        adapted.vector_top_k = int(8 + (query_complexity * 12))  # 8-20 range
        adapted.graph_traversal_depth = min(
            1 + int(query_complexity * 3), 4
        )  # 1-4 range

        # Adjust thresholds based on content relationship complexity
        relationship_complexity = content_characteristics.relationship_complexity
        adapted.vector_similarity_threshold = 0.8 - (relationship_complexity * 0.2)
        adapted.graph_relationship_threshold = 0.6 - (relationship_complexity * 0.2)

        # Adjust multi-modal weights based on content structure
        structure_complexity = content_characteristics.structural_complexity
        # Adapt multi-modal weights based on measured structure complexity
        if structure_complexity > 0.7:  # Measured high structure complexity
            adapted.vector_weight = 0.3
            adapted.graph_weight = 0.4
            adapted.hybrid_weight = 0.3
        elif structure_complexity < 0.3:  # Measured low structure complexity
            adapted.vector_weight = 0.5
            adapted.graph_weight = 0.2
            adapted.hybrid_weight = 0.3

        return adapted


@dataclass
class UniversalQueryConfig:
    """Configuration for universal query generation"""

    # Query generation settings
    enable_gremlin_optimization: bool = True
    enable_search_optimization: bool = True
    enable_analysis_optimization: bool = True

    # Cache settings
    enable_query_caching: bool = True
    cache_ttl_minutes: int = 30
    max_cache_entries: int = 1000

    # Generation parameters (adaptive)
    max_complexity_level: str = "adaptive"  # Determined by content characteristics
    optimization_goal: str = "balanced"  # "speed", "accuracy", "balanced"

    # Batch processing
    max_batch_size: int = 10
    concurrent_generation: bool = True


class UniversalConfigManager:
    """Universal configuration manager - zero domain bias"""

    def __init__(self):
        self._system_config = None
        self._model_config = None
        self._query_config = None
        self._content_analyses = {}  # Cache for content analyses
        self._config_cache = {}  # Cache for generated configurations

    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        if self._system_config is None:
            self._system_config = SystemConfig()
        return self._system_config

    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        if self._model_config is None:
            self._model_config = ModelConfig()
        return self._model_config

    @classmethod
    def get_openai_config(cls) -> ModelConfig:
        """Class method for getting OpenAI config (compatibility with infrastructure layer)"""
        manager = cls()
        return manager.get_model_config()

    def get_query_config(self) -> UniversalQueryConfig:
        """Get query generation configuration"""
        if self._query_config is None:
            self._query_config = UniversalQueryConfig()
        return self._query_config

    async def analyze_content_characteristics(
        self, content_samples: List[str], content_signature: str = None
    ) -> ContentCharacteristics:
        """Analyze content to discover characteristics using REAL Azure OpenAI LLM analysis"""

        if not content_samples:
            return ContentCharacteristics()

        # Cache key based on content hash
        cache_key = hash(str(content_samples[:3]))  # Use first 3 samples for key
        if cache_key in self._content_analyses:
            return self._content_analyses[cache_key]

        # Use REAL Azure OpenAI LLM analysis via agent toolsets
        all_text = " ".join(content_samples)

        # Import and call the real LLM analysis functions from agent toolsets
        from agents.core.agent_toolsets import (
            _analyze_concept_density_via_llm,
            _analyze_lexical_diversity_via_llm,
            _analyze_sentence_complexity_via_llm,
            _analyze_structural_consistency_via_llm,
            _analyze_vocabulary_complexity_via_llm,
        )
        from agents.core.universal_deps import get_universal_deps

        # Create a mock RunContext for calling the LLM functions
        deps = await get_universal_deps()

        class MockRunContext:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockRunContext(deps)

        # Use REAL Azure OpenAI for all analysis
        vocabulary_complexity = await _analyze_vocabulary_complexity_via_llm(
            ctx, all_text
        )
        concept_density = await _analyze_concept_density_via_llm(ctx, all_text)

        # Use REAL Azure OpenAI for all remaining analysis
        sentence_complexity = await _analyze_sentence_complexity_via_llm(ctx, all_text)
        lexical_diversity = await _analyze_lexical_diversity_via_llm(ctx, all_text)
        structural_consistency = await _analyze_structural_consistency_via_llm(
            ctx, all_text
        )

        # Calculate relationship complexity using LLM-derived values
        relationship_complexity = min(
            concept_density * 1.2, 1.0
        )  # Derive from concept density

        # Calculate structural complexity using LLM analysis
        structural_complexity = (
            structural_consistency  # Use LLM-analyzed consistency as complexity
        )

        # Calculate terminology uniqueness using LLM analysis
        terminology_uniqueness = lexical_diversity  # Use LLM-analyzed diversity

        # Calculate cross-reference ratio using LLM-derived values
        cross_reference_ratio = min(
            structural_complexity * 0.5, 1.0
        )  # Derive from structure

        # Calculate text complexity level from LLM sentence complexity
        measured_text_complexity_level = min(12, max(6, int(sentence_complexity / 2)))

        # Calculate paragraph metrics using LLM-derived values
        paragraphs = all_text.split("\n\n")
        avg_paragraph_length = len(all_text) / max(len(paragraphs), 1)

        characteristics = ContentCharacteristics(
            vocabulary_complexity=vocabulary_complexity,
            concept_density=max(0.0, concept_density),
            relationship_complexity=max(0.0, relationship_complexity),
            structural_complexity=max(0.0, structural_complexity),
            avg_sentence_length=sentence_complexity,  # Use LLM-analyzed sentence complexity
            avg_paragraph_length=avg_paragraph_length,
            terminology_uniqueness=terminology_uniqueness,
            cross_reference_ratio=cross_reference_ratio,
            content_signature=content_signature or f"llm_analyzed_{cache_key}",
            measured_text_complexity_level=measured_text_complexity_level,
            analysis_confidence=0.9,  # Higher confidence with LLM analysis
            sample_size=len(content_samples),
        )

        self._content_analyses[cache_key] = characteristics
        logger.info(
            f"Analyzed content characteristics: vocab={vocabulary_complexity:.2f}, density={concept_density:.2f}"
        )

        return characteristics

    async def _analyze_query_characteristics_via_llm(
        self, query: str
    ) -> Dict[str, float]:
        """Analyze query characteristics using REAL Azure OpenAI LLM analysis"""

        # Import LLM analysis functions
        from agents.core.agent_toolsets import (
            _analyze_sentence_complexity_via_llm,
            _analyze_vocabulary_complexity_via_llm,
        )
        from agents.core.universal_deps import get_universal_deps

        # Create RunContext for LLM analysis
        deps = await get_universal_deps()

        class MockRunContext:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockRunContext(deps)

        # Analyze query with LLM
        complexity = await _analyze_vocabulary_complexity_via_llm(ctx, query)
        sentence_complexity = await _analyze_sentence_complexity_via_llm(ctx, query)

        # Calculate length factor from sentence complexity
        length_factor = min(sentence_complexity / 10.0, 2.0)  # Scale to 0-2 range

        return {
            "complexity": complexity,
            "length_factor": length_factor,
        }

    async def get_extraction_config(
        self, content_samples: List[str] = None, content_signature: str = None
    ) -> UniversalExtractionConfig:
        """Get extraction configuration adapted to discovered content characteristics"""

        if content_samples:
            characteristics = await self.analyze_content_characteristics(
                content_samples, content_signature
            )
            base_config = UniversalExtractionConfig()
            return base_config.adapt_to_content(characteristics)
        else:
            # Return base configuration with default characteristics
            return UniversalExtractionConfig()

    async def get_search_config(
        self,
        query: str = None,
        content_samples: List[str] = None,
        content_signature: str = None,
    ) -> UniversalSearchConfig:
        """Get search configuration adapted to query and content characteristics"""

        # Use REAL Azure OpenAI for query analysis
        query_characteristics = {}
        if query:
            query_characteristics = await self._analyze_query_characteristics_via_llm(
                query
            )

        # Get content characteristics
        if content_samples:
            content_characteristics = await self.analyze_content_characteristics(
                content_samples, content_signature
            )
        else:
            content_characteristics = ContentCharacteristics()

        base_config = UniversalSearchConfig()
        return base_config.adapt_to_query_and_content(
            query_characteristics, content_characteristics
        )

    def clear_cache(self):
        """Clear all analysis and configuration caches"""
        self._content_analyses.clear()
        self._config_cache.clear()
        logger.info("Universal configuration cache cleared")

    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure service configuration (infrastructure only)"""
        model_config = self.get_model_config()
        system_config = self.get_system_config()

        return {
            "openai_endpoint": model_config.endpoint,
            "openai_api_key": model_config.api_key,
            "openai_api_version": model_config.api_version,
            "openai_deployment": model_config.deployment_name,
            "search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            "cosmos_endpoint": os.getenv("AZURE_COSMOS_ENDPOINT", ""),
            "storage_account": os.getenv("AZURE_STORAGE_ACCOUNT", ""),
            "openai_timeout": system_config.openai_timeout,
            "search_timeout": system_config.search_timeout,
            "cosmos_timeout": system_config.cosmos_timeout,
            "max_retries": system_config.max_retries,
        }

    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all configuration components"""
        model_config = self.get_model_config()

        validation = {
            "azure_openai_configured": model_config.is_configured(),
            "azure_search_configured": bool(os.getenv("AZURE_SEARCH_ENDPOINT")),
            "azure_cosmos_configured": bool(os.getenv("AZURE_COSMOS_ENDPOINT")),
            "azure_storage_configured": bool(os.getenv("AZURE_STORAGE_ACCOUNT")),
        }

        all_configured = all(validation.values())
        validation["all_services_configured"] = all_configured

        if not all_configured:
            logger.warning(f"Configuration validation: {validation}")

        return validation


# Private singleton for proper dependency injection
_universal_config_manager: Optional[UniversalConfigManager] = None


def get_universal_config_manager() -> UniversalConfigManager:
    """Factory function to get universal config manager instance."""
    global _universal_config_manager
    if _universal_config_manager is None:
        _universal_config_manager = UniversalConfigManager()
    return _universal_config_manager


# Legacy compatibility
universal_config_manager = get_universal_config_manager

# Alias for infrastructure layer compatibility
UniversalConfig = UniversalConfigManager


# Universal convenience functions using factory pattern
def get_system_config() -> SystemConfig:
    """Get system configuration"""
    config_manager = get_universal_config_manager()
    return config_manager.get_system_config()


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    config_manager = get_universal_config_manager()
    return config_manager.get_model_config()


async def get_extraction_config_universal(
    content_samples: List[str] = None, content_signature: str = None
) -> UniversalExtractionConfig:
    """Get extraction configuration adapted to discovered content characteristics"""
    config_manager = get_universal_config_manager()
    return await config_manager.get_extraction_config(
        content_samples, content_signature
    )


async def get_search_config_universal(
    query: str = None, content_samples: List[str] = None, content_signature: str = None
) -> UniversalSearchConfig:
    """Get search configuration adapted to query and content characteristics"""
    config_manager = get_universal_config_manager()
    return await config_manager.get_search_config(
        query, content_samples, content_signature
    )


def get_query_config() -> UniversalQueryConfig:
    """Get query generation configuration"""
    config_manager = get_universal_config_manager()
    return config_manager.get_query_config()


def get_azure_config() -> Dict[str, Any]:
    """Get Azure service configuration"""
    config_manager = get_universal_config_manager()
    return config_manager.get_azure_config()


async def analyze_content_characteristics(
    content_samples: List[str], content_signature: str = None
) -> ContentCharacteristics:
    """Analyze content to discover characteristics"""
    config_manager = get_universal_config_manager()
    return await config_manager.analyze_content_characteristics(
        content_samples, content_signature
    )


def validate_configuration() -> Dict[str, bool]:
    """Validate all configuration"""
    config_manager = get_universal_config_manager()
    return config_manager.validate_configuration()


# Legacy compatibility functions (bias-free)
def get_model_config_bootstrap() -> ModelConfig:
    """Bootstrap model configuration (legacy compatibility)"""
    return get_model_config()


def initialize_configuration():
    """Initialize and validate configuration on startup"""
    validation = validate_configuration()

    if not validation["azure_openai_configured"]:
        logger.error(
            "Azure OpenAI not configured! Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
        )
        raise RuntimeError("Azure OpenAI configuration required")

    logger.info("Universal configuration system initialized successfully")
    return validation


# Export main interfaces
__all__ = [
    "SystemConfig",
    "ModelConfig",
    "UniversalExtractionConfig",
    "UniversalSearchConfig",
    "UniversalQueryConfig",
    "ContentCharacteristics",
    "get_system_config",
    "get_model_config",
    "get_extraction_config_universal",
    "get_search_config_universal",
    "get_query_config",
    "get_azure_config",
    "analyze_content_characteristics",
    "validate_configuration",
    "get_model_config_bootstrap",
    "initialize_configuration",
    "universal_config_manager",
]
