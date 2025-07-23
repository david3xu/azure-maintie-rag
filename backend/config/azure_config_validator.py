"""Azure Configuration Validation for Enterprise RAG Migration"""
import logging
from typing import Dict, Any
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureConfigValidator:
    """Enterprise Azure configuration validation service"""

    def __init__(self):
        self.settings = azure_settings

    def validate_cosmos_db_configuration(self) -> Dict[str, Any]:
        """Validate Cosmos DB configuration from azure_settings"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "configuration": {}
        }

        # Required Azure Cosmos DB settings
        required_settings = {
            "azure_cosmos_endpoint": self.settings.azure_cosmos_endpoint,
            "azure_cosmos_key": self.settings.azure_cosmos_key,
            "azure_cosmos_database": self.settings.azure_cosmos_database,
            "azure_cosmos_container": self.settings.azure_cosmos_container
        }

        for setting_name, setting_value in required_settings.items():
            if not setting_value:
                validation_result["errors"].append(f"Missing required setting: {setting_name}")
                validation_result["valid"] = False
            else:
                validation_result["configuration"][setting_name] = setting_value

        # Environment-specific validation
        environment = getattr(self.settings, 'azure_environment', 'dev')
        expected_database_suffix = f"-{environment}"
        expected_container_suffix = f"-{environment}"

        if not self.settings.azure_cosmos_database.endswith(expected_database_suffix):
            validation_result["warnings"].append(
                f"Database name should end with '{expected_database_suffix}' for {environment} environment"
            )

        if not self.settings.azure_cosmos_container.endswith(expected_container_suffix):
            validation_result["warnings"].append(
                f"Container name should end with '{expected_container_suffix}' for {environment} environment"
            )

        return validation_result

    def validate_azure_ml_configuration(self) -> Dict[str, Any]:
        """Validate Azure ML configuration from azure_settings"""
        validation_result = {
            "valid": True,
            "errors": [],
            "configuration": {}
        }

        required_ml_settings = {
            "azure_subscription_id": self.settings.azure_subscription_id,
            "azure_resource_group": self.settings.azure_resource_group,
            "azure_ml_workspace_name": self.settings.azure_ml_workspace_name,
            "gnn_training_trigger_threshold": self.settings.gnn_training_trigger_threshold,
            "gnn_model_deployment_tier": self.settings.gnn_model_deployment_tier
        }

        for setting_name, setting_value in required_ml_settings.items():
            if not setting_value:
                validation_result["errors"].append(f"Missing required ML setting: {setting_name}")
                validation_result["valid"] = False
            else:
                validation_result["configuration"][setting_name] = setting_value

        return validation_result