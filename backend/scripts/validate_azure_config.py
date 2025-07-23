#!/usr/bin/env python3
"""Validate Azure configuration before running GNN pipeline"""

from config.settings import azure_settings
import logging

logger = logging.getLogger(__name__)

def validate_azure_configuration():
    """Validate all required Azure configuration is present"""
    required_settings = [
        ('azure_cosmos_endpoint', azure_settings.azure_cosmos_endpoint),
        ('azure_cosmos_key', azure_settings.azure_cosmos_key),
        ('azure_cosmos_database', azure_settings.azure_cosmos_database),
        ('azure_cosmos_container', azure_settings.azure_cosmos_container),
        ('azure_subscription_id', azure_settings.azure_subscription_id),
        ('azure_resource_group', azure_settings.azure_resource_group),
        ('azure_ml_workspace_name', getattr(azure_settings, 'azure_ml_workspace_name', None))
    ]
    missing_settings = []
    for setting_name, setting_value in required_settings:
        if not setting_value or str(setting_value).strip() == "":
            missing_settings.append(setting_name)
    if missing_settings:
        logger.error(f"Missing required Azure settings: {', '.join(missing_settings)}")
        print(f"❌ Missing required Azure settings: {', '.join(missing_settings)}")
        return False
    logger.info("✅ Azure configuration validation passed")
    print("✅ Azure configuration validation passed")
    return True

if __name__ == "__main__":
    validate_azure_configuration()