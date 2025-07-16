"""Universal configuration loader for any domain."""

import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import os


class ConfigLoader:
    """Universal configuration loader that works with any domain.

    Loads domain-specific configurations and merges with universal
    templates to create complete domain configurations.
    """

    def __init__(self, config_dir: str = "config"):
        """Initialize config loader."""
        self.config_dir = Path(config_dir)
        self.domains_dir = self.config_dir / "domains"
        self.models_dir = self.config_dir / "models"
        self.pipeline_dir = self.config_dir / "pipeline"

    def load_domain_config(self, domain: str) -> Dict[str, Any]:
        """Load configuration for a specific domain."""
        domain_file = self.domains_dir / f"{domain}.yaml"

        if not domain_file.exists():
            raise FileNotFoundError(f"Domain config not found: {domain_file}")

        with open(domain_file, 'r') as f:
            config = yaml.safe_load(f)

        # Merge with universal template
        template = self.load_universal_template()
        merged_config = self._deep_merge(template, config)

        return merged_config

    def load_universal_template(self) -> Dict[str, Any]:
        """Load universal configuration template."""
        template_file = self.domains_dir / "universal_schema_template.yaml"

        if not template_file.exists():
            raise FileNotFoundError(f"Universal template not found: {template_file}")

        with open(template_file, 'r') as f:
            return yaml.safe_load(f)

    def load_model_config(self, model_type: str) -> Dict[str, Any]:
        """Load model configuration."""
        model_file = self.models_dir / f"{model_type}_config.yaml"

        if not model_file.exists():
            raise FileNotFoundError(f"Model config not found: {model_file}")

        with open(model_file, 'r') as f:
            return yaml.safe_load(f)

    def load_pipeline_config(self, pipeline_type: str) -> Dict[str, Any]:
        """Load pipeline configuration."""
        pipeline_file = self.pipeline_dir / f"{pipeline_type}_config.yaml"

        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline config not found: {pipeline_file}")

        with open(pipeline_file, 'r') as f:
            return yaml.safe_load(f)

    def save_domain_config(self, domain: str, config: Dict[str, Any]) -> None:
        """Save domain configuration."""
        domain_file = self.domains_dir / f"{domain}.yaml"

        # Ensure directory exists
        domain_file.parent.mkdir(parents=True, exist_ok=True)

        with open(domain_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)

    def list_domains(self) -> List[str]:
        """List available domains."""
        if not self.domains_dir.exists():
            return []

        domains = []
        for file in self.domains_dir.glob("*.yaml"):
            if file.name != "universal_schema_template.yaml":
                domains.append(file.stem)

        return sorted(domains)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        required_sections = [
            'domain', 'entities', 'relationships',
            'processing', 'query', 'performance'
        ]

        for section in required_sections:
            if section not in config:
                return False

        return True

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        env_config = {}

        # Load from environment variables
        env_vars = [
            'AZURE_API_KEY', 'AZURE_ENDPOINT', 'AZURE_DEPLOYMENT',
            'LOG_LEVEL', 'CACHE_DIR', 'DATA_DIR'
        ]

        for var in env_vars:
            value = os.getenv(var)
            if value:
                env_config[var.lower()] = value

        return env_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result