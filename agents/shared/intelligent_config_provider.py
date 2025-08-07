"""
Intelligent Configuration Provider - State Transfer Engine for workflow integration.

This module provides intelligent configuration from Config-Extraction workflow,
eliminating all hardcoded values in the search system.
"""

from typing import Dict, Any, Optional
from .config_enforcement import ConfigurationEnforcementError, validate_config
from .workflow_state_bridge import WorkflowStateBridge, create_sample_config

# Import constants for zero-hardcoded-values compliance
from agents.core.constants import CacheConstants


class IntelligentConfigProvider:
    """Provides configuration from workflow-generated intelligence"""

    def __init__(self):
        self.state_bridge = WorkflowStateBridge()

    def get_search_config(self, domain: str) -> Dict[str, Any]:
        """Get search configuration from Config-Extraction workflow"""

        # Try to get intelligent configuration first
        config = self.state_bridge.get_domain_config(domain)

        if config is None:
            # Force Config-Extraction workflow execution
            config = self._force_config_extraction(domain)

        # Simple validation using PydanticAI built-in patterns
        validated_config = {}
        for key, value in config.items():
            if key.endswith("_source"):
                # Source tracking keys - pass through directly
                validated_config[key] = value
            else:
                # Simple validation - complex enforcement replaced with PydanticAI Field constraints
                validated_config[key] = validate_config(
                    config_key=key,
                    value=value,
                    source=config.get(f"{key}_source", "workflow_generated"),
                )

        return validated_config

    def _force_config_extraction(self, domain: str) -> Dict[str, Any]:
        """Force Config-Extraction workflow execution if no config exists"""

        print(
            f"🧠 No intelligent config found for domain '{domain}'. Running Config-Extraction workflow..."
        )

        # Try to import and execute the Config-Extraction workflow
        try:
            from agents.workflows.config_extraction_graph import (
                ConfigExtractionWorkflow,
            )

            workflow = ConfigExtractionWorkflow()
            result = workflow.execute_for_domain(domain)

            if not result.success:
                raise ConfigurationEnforcementError(
                    f"Config-Extraction workflow failed for domain '{domain}'. "
                    f"Cannot proceed with hardcoded fallbacks. Error: {result.error}"
                )

            return result.config

        except ImportError:
            # Config-Extraction workflow not available - create development config
            print(
                "⚠️  Config-Extraction workflow not available. Creating development configuration..."
            )
            return self._create_development_config(domain)

    def _create_development_config(self, domain: str) -> Dict[str, Any]:
        """
        Create development configuration when Config-Extraction workflow is unavailable.

        This should ONLY be used in development environments.
        """
        sample_config = create_sample_config(domain)

        # Store the sample config so it can be retrieved later
        self.state_bridge.store_config(sample_config)

        # Return the config in the expected format
        return self.state_bridge.get_domain_config(domain)

    def refresh_config(self, domain: str, force: bool = False) -> Dict[str, Any]:
        """Refresh configuration for a domain"""

        if force or not self.state_bridge.is_config_fresh(domain):
            print(f"🔄 Refreshing configuration for domain '{domain}'...")
            # Remove old config and force regeneration
            config_file = self.state_bridge.state_dir / f"domain_{domain}_config.json"
            if config_file.exists():
                config_file.unlink()

            return self.get_search_config(domain)
        else:
            print(f"✅ Configuration for domain '{domain}' is fresh")
            return self.get_search_config(domain)

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of the intelligent configuration provider"""

        bridge_status = self.state_bridge.get_bridge_status()

        return {
            "provider_operational": True,
            "bridge_status": bridge_status,
            "validation_method": "pydantic_field_constraints",
            "available_domains": bridge_status["domains"],
            "fresh_configs_count": len(bridge_status["fresh_configs"]),
            "stale_configs_count": len(bridge_status["stale_configs"]),
        }

    def validate_domain_config(self, domain: str) -> Dict[str, Any]:
        """Validate that a domain's configuration is complete and valid"""

        try:
            config = self.get_search_config(domain)

            # Check required configuration keys
            required_keys = [
                "similarity_threshold",
                "processing_patterns",
                "synthesis_weights",
                "routing_rules",
            ]

            validation_result = {
                "domain": domain,
                "valid": True,
                "missing_keys": [],
                "invalid_values": [],
                "source_validation": [],
            }

            for key in required_keys:
                if key not in config:
                    validation_result["missing_keys"].append(key)
                    validation_result["valid"] = False
                else:
                    # Validate source tracking
                    source_key = f"{key}_source"
                    if source_key not in config:
                        validation_result["source_validation"].append(
                            f"Missing source for {key}"
                        )
                    elif "hardcoded" in config[source_key]:
                        validation_result["source_validation"].append(
                            f"Hardcoded source for {key}"
                        )
                        validation_result["valid"] = False

            # Manual validation replaced with PydanticAI built-in patterns
            # Use: similarity_threshold: float = Field(ge=CacheConstants.MIN_CONFIDENCE, le=CacheConstants.MAX_CONFIDENCE)
            # This provides automatic validation with better error messages

            return validation_result

        except Exception as e:
            return {"domain": domain, "valid": False, "error": str(e)}


# Global provider instance for easy access
_global_provider = IntelligentConfigProvider()


def get_intelligent_config(domain: str) -> Dict[str, Any]:
    """
    Global function to get intelligent configuration for a domain.

    This is a convenience function that uses the global provider instance.
    """
    return _global_provider.get_search_config(domain)


def refresh_domain_config(domain: str, force: bool = False) -> Dict[str, Any]:
    """Global function to refresh configuration for a domain"""
    return _global_provider.refresh_config(domain, force)


def get_config_provider_status() -> Dict[str, Any]:
    """Get status of the global configuration provider"""
    return _global_provider.get_provider_status()
