"""
Azure-specific configuration management for Universal RAG
Extends main settings with Azure service configurations
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

from .settings import Settings


class AzureSettings(BaseSettings):
    """Azure service configuration settings - follows existing Settings pattern"""

    # Azure Storage Settings
    azure_storage_account: str = Field(default="", env="AZURE_STORAGE_ACCOUNT")
    azure_storage_key: str = Field(default="", env="AZURE_STORAGE_KEY")
    azure_blob_container: str = Field(default="universal-rag-data", env="AZURE_BLOB_CONTAINER")
    azure_storage_connection_string: str = Field(default="", env="AZURE_STORAGE_CONNECTION_STRING")

    # Azure Cognitive Search Settings
    azure_search_service: str = Field(default="", env="AZURE_SEARCH_SERVICE")
    azure_search_key: str = Field(default="", env="AZURE_SEARCH_KEY")
    azure_search_index: str = Field(default="universal-rag-index", env="AZURE_SEARCH_INDEX")
    azure_search_api_version: str = Field(default="2023-11-01", env="AZURE_SEARCH_API_VERSION")

    # Azure Cosmos DB Settings (Gremlin API)
    azure_cosmos_endpoint: str = Field(default="", env="AZURE_COSMOS_ENDPOINT")
    azure_cosmos_key: str = Field(default="", env="AZURE_COSMOS_KEY")
    azure_cosmos_database: str = Field(default="universal-rag-db", env="AZURE_COSMOS_DATABASE")
    azure_cosmos_container: str = Field(default="knowledge-graph", env="AZURE_COSMOS_CONTAINER")

    # Azure ML Settings
    azure_subscription_id: str = Field(default="", env="AZURE_SUBSCRIPTION_ID")
    azure_resource_group: str = Field(default="", env="AZURE_RESOURCE_GROUP")
    azure_ml_workspace: str = Field(default="", env="AZURE_ML_WORKSPACE")
    azure_tenant_id: str = Field(default="", env="AZURE_TENANT_ID")

    # Azure Resource Naming Convention (data-driven)
    azure_resource_prefix: str = Field(default="maintie", env="AZURE_RESOURCE_PREFIX")
    azure_environment: str = Field(default="dev", env="AZURE_ENVIRONMENT")
    azure_region: str = Field(default="eastus", env="AZURE_REGION")

    def get_resource_name(self, resource_type: str, suffix: str = "") -> str:
        """Generate Azure resource names following convention"""
        parts = [self.azure_resource_prefix, self.azure_environment, self.azure_region, resource_type]
        if suffix:
            parts.append(suffix)
        return "-".join(parts)

    def validate_azure_config(self) -> Dict[str, Any]:
        """Validate Azure configuration completeness"""
        return {
            "storage_configured": bool(self.azure_storage_account and self.azure_storage_key),
            "search_configured": bool(self.azure_search_service and self.azure_search_key),
            "cosmos_configured": bool(self.azure_cosmos_endpoint and self.azure_cosmos_key),
            "ml_configured": bool(self.azure_subscription_id and self.azure_resource_group),
        }


# Singleton instance following existing settings pattern
azure_settings = AzureSettings()