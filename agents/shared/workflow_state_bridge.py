"""
Workflow State Bridge - Manages state transfer between Config-Extraction and Search workflows.

This module provides the core infrastructure for transferring intelligent configurations
from the Config-Extraction workflow to the Search workflow, eliminating hardcoded values.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# Import models from centralized data models
from agents.core.data_models import WorkflowStateBridge as WorkflowConfig
from agents.core.constants import (
    StatisticalConstants,
    UniversalSearchConstants,
    CacheConstants,
    MLModelConstants,
)


class WorkflowStateBridge:
    """Manages state transfer between Config-Extraction and Search workflows"""

    def __init__(self):
        self.state_dir = Path("cache/workflow_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def store_config(self, config: WorkflowConfig) -> None:
        """Store configuration from Config-Extraction workflow"""
        config_file = self.state_dir / f"domain_{config.domain}_config.json"

        config_data = {
            "domain": config.domain,
            "similarity_threshold": config.similarity_threshold,
            "processing_patterns": config.processing_patterns,
            "synthesis_weights": config.synthesis_weights,
            "routing_rules": config.routing_rules,
            "generated_at": config.generated_at.isoformat(),
            "source_workflow": config.source_workflow,
            "metadata": {
                "similarity_threshold_source": config.similarity_threshold_source,
                "processing_patterns_source": config.processing_patterns_source,
                "synthesis_weights_source": config.synthesis_weights_source,
                "routing_rules_source": config.routing_rules_source,
            },
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    def get_domain_config(self, domain: str) -> Optional[Dict[str, Any]]:
        """Retrieve configuration for specific domain"""
        config_file = self.state_dir / f"domain_{domain}_config.json"

        if not config_file.exists():
            return None

        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Add source tracking to main config
        config = config_data.copy()
        for key, source in config_data.get("metadata", {}).items():
            config[key] = source

        return config

    def list_available_domains(self) -> list[str]:
        """List all domains with available configurations"""
        domains = []
        for config_file in self.state_dir.glob("domain_*_config.json"):
            domain = config_file.stem.replace("domain_", "").replace("_config", "")
            domains.append(domain)
        return sorted(domains)

    def get_config_age(self, domain: str) -> Optional[datetime]:
        """Get the age of a domain's configuration"""
        config = self.get_domain_config(domain)
        if config and "generated_at" in config:
            return datetime.fromisoformat(config["generated_at"])
        return None

    def is_config_fresh(self, domain: str, max_age_hours: int = 24) -> bool:
        """Check if a domain's configuration is fresh (within max_age_hours)"""
        config_age = self.get_config_age(domain)
        if not config_age:
            return False

        age_hours = (
            datetime.now() - config_age
        ).total_seconds() / CacheConstants.SECONDS_PER_HOUR
        return age_hours <= max_age_hours

    def cleanup_old_configs(self, max_age_hours: int = 168) -> int:  # 7 days default
        """Clean up old configuration files"""
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (
            max_age_hours * CacheConstants.SECONDS_PER_HOUR
        )

        for config_file in self.state_dir.glob("domain_*_config.json"):
            if config_file.stat().st_mtime < cutoff_time:
                config_file.unlink()
                cleaned_count += 1

        return cleaned_count

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get status information about the workflow bridge"""
        domains = self.list_available_domains()

        status = {
            "total_domains": len(domains),
            "domains": domains,
            "bridge_operational": True,
            "state_directory": str(self.state_dir),
            "fresh_configs": [],
            "stale_configs": [],
        }

        for domain in domains:
            if self.is_config_fresh(domain):
                status["fresh_configs"].append(domain)
            else:
                status["stale_configs"].append(domain)

        return status


def create_sample_config(domain: str = "test_domain") -> WorkflowConfig:
    """
    Create a sample configuration for testing purposes.

    This should ONLY be used for development and testing.
    Production configs must come from Config-Extraction workflow.
    """
    return WorkflowConfig(
        domain=domain,
        similarity_threshold=StatisticalConstants.MEDIUM_CONTENT_SIMILARITY_THRESHOLD,  # Learned from domain analysis
        processing_patterns={
            "vector_params": {
                "model": AzureServiceConstants.DEFAULT_EMBEDDING_MODEL,
                "dimensions": MLModelConstants.EMBEDDING_DIMENSION,
            },
            "graph_params": {
                "max_depth": 3,
                "min_score": UniversalSearchConstants.VECTOR_SIMILARITY_THRESHOLD,
            },
            "gnn_params": {
                "hidden_dim": MLModelConstants.GNN_HIDDEN_DIM,
                "num_layers": MLModelConstants.GNN_NUM_LAYERS,
            },
        },
        synthesis_weights={
            "vector_weight": MLModelConstants.DEFAULT_VECTOR_WEIGHT,
            "graph_weight": MLModelConstants.DEFAULT_GRAPH_WEIGHT,
            "gnn_weight": MLModelConstants.DEFAULT_GNN_WEIGHT,
        },
        routing_rules={
            "graph_params": {"traversal_strategy": "breadth_first"},
            "gnn_params": {"aggregation": "attention"},
        },
        generated_at=datetime.now(),
    )
