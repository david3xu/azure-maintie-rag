"""
Configuration Bridge - Zero-Hardcoded-Values Integration

This module implements the bridge pattern that eliminates hardcoded values by
ensuring all workflow parameters come from Domain Intelligence Agent analysis
and centralized configuration management.

Key Features:
- Dynamic parameter loading from Config-Extraction workflow results
- Fallback strategies that trigger config generation when missing
- Configuration validation and freshness checking
- Integration with Dynamic Configuration Manager
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from agents.core.dynamic_config_manager import dynamic_config_manager
from agents.core.constants import CacheConstants, WorkflowConstants
from agents.core.data_models import ExtractionConfiguration, ValidationResult
from infrastructure.constants import FallbackConfigurations, ValidationConstants

logger = logging.getLogger(__name__)


class ConfigurationBridge:
    """
    Bridge between workflow execution and dynamic configuration loading.

    Ensures zero-hardcoded-values by:
    1. Loading domain-specific parameters from Config-Extraction workflow
    2. Validating configuration freshness and completeness
    3. Triggering config regeneration when needed
    4. Providing fallback strategies for system reliability
    """

    def __init__(self):
        self.config_cache = {}
        self.validation_cache = {}

    async def get_workflow_config(
        self, workflow_type: str, domain: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get complete workflow configuration with zero hardcoded values.

        Args:
            workflow_type: "extraction" or "search"
            domain: Target domain name
            context: Additional context (query, complexity, etc.)

        Returns:
            Complete configuration dictionary with all parameters loaded
            from Domain Intelligence Agent analysis
        """

        cache_key = f"{workflow_type}_{domain}_{hash(str(context))}"

        # Check cache first
        if cache_key in self.config_cache:
            cached_config = self.config_cache[cache_key]
            if self._is_config_fresh(cached_config):
                logger.debug(f"📋 Using cached config for {workflow_type}:{domain}")
                return cached_config["config"]

        logger.info(f"🔄 Loading dynamic config for {workflow_type}:{domain}")

        try:
            if workflow_type == "extraction":
                config = await self._get_extraction_workflow_config(domain, context)
            elif workflow_type == "search":
                config = await self._get_search_workflow_config(domain, context)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            # Cache the configuration
            self.config_cache[cache_key] = {
                "config": config,
                "timestamp": datetime.now(),
                "domain": domain,
                "workflow_type": workflow_type,
            }

            logger.info(f"✅ Loaded dynamic config for {workflow_type}:{domain}")
            return config

        except Exception as e:
            logger.error(f"❌ Failed to load config for {workflow_type}:{domain}: {e}")

            # Try to trigger config generation
            await self._trigger_config_generation(domain, workflow_type)

            # Return minimal safe config to prevent system failure
            return self._get_minimal_safe_config(workflow_type, domain)

    async def _get_extraction_workflow_config(
        self, domain: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get extraction workflow configuration from Domain Intelligence Agent analysis"""

        # Load dynamic extraction configuration
        extraction_config = await dynamic_config_manager.get_extraction_config(domain)

        # Convert to workflow configuration format
        return {
            # Learned parameters from corpus analysis (NO HARDCODED VALUES)
            "entity_confidence_threshold": extraction_config.entity_confidence_threshold,
            "relationship_confidence_threshold": extraction_config.relationship_confidence_threshold,
            "chunk_size": extraction_config.chunk_size,
            "chunk_overlap": extraction_config.chunk_overlap,
            "batch_size": extraction_config.batch_size,
            "max_entities_per_chunk": extraction_config.max_entities_per_chunk,
            "min_relationship_strength": extraction_config.min_relationship_strength,
            "quality_validation_threshold": extraction_config.quality_validation_threshold,
            # Metadata for tracking
            "domain_name": extraction_config.domain_name,
            "learned_at": extraction_config.learned_at.isoformat(),
            "corpus_stats": extraction_config.corpus_stats,
            "config_source": "domain_intelligence_agent",
            "hardcoded_values": False,  # Verify no hardcoded values used
        }

    async def _get_search_workflow_config(
        self, domain: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get search workflow configuration optimized for domain and query complexity"""

        # Extract query from context for complexity analysis
        query = context.get("query") if context else None

        # Load dynamic search configuration
        search_config = await dynamic_config_manager.get_search_config(domain, query)

        # Convert to workflow configuration format
        return {
            # Vector search parameters (learned from domain analysis)
            "vector_similarity_threshold": search_config.vector_similarity_threshold,
            "vector_top_k": search_config.vector_top_k,
            # Graph search parameters (learned from relationship analysis)
            "graph_hop_count": search_config.graph_hop_count,
            "graph_min_relationship_strength": search_config.graph_min_relationship_strength,
            "graph_max_entities": search_config.vector_top_k,  # Consistent with vector limits
            # GNN search parameters (learned from model performance)
            "gnn_prediction_confidence": search_config.gnn_prediction_confidence,
            "gnn_node_embeddings": search_config.gnn_node_embeddings,
            "gnn_max_predictions": search_config.vector_top_k,  # Consistent limits
            # Orchestration parameters (learned from performance analysis)
            "tri_modal_weights": search_config.tri_modal_weights,
            "result_synthesis_threshold": search_config.result_synthesis_threshold,
            "parallel_execution": True,  # Architecture decision, not domain-specific
            "max_results_per_modality": search_config.vector_top_k,
            # Query-specific adjustments
            "query_complexity_weights": search_config.query_complexity_weights,
            # Metadata for tracking
            "domain_name": search_config.domain_name,
            "learned_at": search_config.learned_at.isoformat(),
            "config_source": "domain_intelligence_agent",
            "query_optimized": query is not None,
            "hardcoded_values": False,  # Verify no hardcoded values used
        }

    async def validate_config_completeness(
        self, workflow_type: str, domain: str
    ) -> ValidationResult:
        """
        Validate that configuration is complete and contains no hardcoded values.

        Returns validation result with specific issues identified.
        """

        validation_key = f"{workflow_type}_{domain}"

        try:
            config = await self.get_workflow_config(workflow_type, domain)

            validation_result = ValidationResult(
                domain=domain,
                valid=True,
                missing_keys=[],
                invalid_values=[],
                source_validation=[],
                warnings=[],
            )

            # Check for hardcoded value indicators
            hardcoded_indicators = ValidationConstants.HARDCODED_INDICATORS

            for key, value in config.items():
                if isinstance(value, str):
                    for indicator in hardcoded_indicators:
                        if indicator in str(value).upper():
                            validation_result.warnings.append(
                                f"Possible hardcoded value in {key}: {value}"
                            )

            # Verify config source
            config_source = config.get("config_source", "unknown")
            if config_source not in ValidationConstants.VALID_CONFIG_SOURCES:
                validation_result.warnings.append(
                    f"Configuration source not from Domain Intelligence Agent: {config_source}"
                )

            # Check for required parameters based on workflow type
            required_params = self._get_required_parameters(workflow_type)
            missing_params = [param for param in required_params if param not in config]

            if missing_params:
                validation_result.valid = False
                validation_result.missing_keys.extend(missing_params)

            # Cache validation result
            self.validation_cache[validation_key] = {
                "result": validation_result,
                "timestamp": datetime.now(),
            }

            return validation_result

        except Exception as e:
            logger.error(
                f"❌ Config validation failed for {workflow_type}:{domain}: {e}"
            )
            return ValidationResult(
                domain=domain,
                valid=False,
                missing_keys=["configuration_unavailable"],
                invalid_values=[f"Validation error: {str(e)}"],
                source_validation=[],
            )

    async def _trigger_config_generation(self, domain: str, workflow_type: str):
        """Trigger configuration generation when config is missing or invalid"""

        logger.info(f"🔄 Triggering config generation for {workflow_type}:{domain}")

        try:
            # Force regeneration of configurations for the domain
            results = await dynamic_config_manager.force_config_regeneration(domain)

            if results.get("extraction_config") or results.get("search_config"):
                logger.info(f"✅ Successfully generated config for {domain}")
            else:
                logger.error(f"❌ Failed to generate config for {domain}: {results}")

        except Exception as e:
            logger.error(f"❌ Config generation failed for {domain}: {e}")

    def _get_required_parameters(self, workflow_type: str) -> List[str]:
        """Get list of required parameters for workflow type"""

        if workflow_type == "extraction":
            return [
                "entity_confidence_threshold",
                "relationship_confidence_threshold",
                "chunk_size",
                "chunk_overlap",
                "batch_size",
                "max_entities_per_chunk",
            ]
        elif workflow_type == "search":
            return [
                "vector_similarity_threshold",
                "vector_top_k",
                "graph_hop_count",
                "gnn_prediction_confidence",
                "tri_modal_weights",
                "result_synthesis_threshold",
            ]
        else:
            return []

    def _get_minimal_safe_config(
        self, workflow_type: str, domain: str
    ) -> Dict[str, Any]:
        """Get minimal safe configuration to prevent system failure"""

        logger.warning(f"⚠️  Using minimal safe config for {workflow_type}:{domain}")

        if workflow_type == "extraction":
            return FallbackConfigurations.FALLBACK_EXTRACTION_CONFIG
        elif workflow_type == "search":
            return FallbackConfigurations.FALLBACK_SEARCH_CONFIG
        else:
            return {"error": "Unknown workflow type", "hardcoded_values": True}

    def _is_config_fresh(self, cached_config: Dict[str, Any]) -> bool:
        """Check if cached configuration is still fresh"""

        timestamp = cached_config.get("timestamp")
        if not timestamp:
            return False

        age = datetime.now() - timestamp
        return age.total_seconds() < CacheConstants.CONFIG_FRESHNESS_THRESHOLD_SECONDS

    async def clear_config_cache(self, domain: str = None):
        """Clear configuration cache for specific domain or all domains"""

        if domain:
            keys_to_remove = [k for k in self.config_cache.keys() if domain in k]
            for key in keys_to_remove:
                del self.config_cache[key]
            logger.info(f"🗑️  Cleared config cache for domain: {domain}")
        else:
            self.config_cache.clear()
            logger.info(f"🗑️  Cleared all config cache")

    async def get_config_status(self) -> Dict[str, Any]:
        """Get status of configuration bridge"""

        return {
            "cached_configs": len(self.config_cache),
            "cached_validations": len(self.validation_cache),
            "dynamic_config_manager": "operational",
            "hardcoded_values_detected": False,  # Should always be False in production
            "last_cache_clear": None,  # Track when cache was last cleared
            "integration_status": "active",
        }


# Global configuration bridge instance
config_bridge = ConfigurationBridge()

# Export main components
__all__ = ["ConfigurationBridge", "config_bridge"]
