"""
Consolidated Azure Services - Unified Azure integration for agent architecture

This service consolidates azure_integration.py + unified_azure_services.py into a single,
clean Azure service container following clean architecture principles.

Features:
- Unified Azure AI Foundry integration for LLM services
- Consolidated non-LLM Azure services (Search, Cosmos, Storage, ML)
- DefaultAzureCredential for unified authentication
- Proper error handling and logging
- Dependency injection ready
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from azure.core.exceptions import AzureError

# Import constants for zero-hardcoded-values compliance
from agents.core.constants import MathematicalConstants, InfrastructureConstants, SystemBoundaryConstants

# Import consolidated data models
from agents.core.data_models import ConsolidatedAzureServices
from azure.identity import DefaultAzureCredential
from pydantic_ai.providers.azure import AzureProvider

# Import clean configuration (CODING_STANDARDS compliant)
from config.centralized_config import get_system_config, get_model_config_bootstrap, get_workflow_config

# Get configuration instances (cleaned)
_system_config = get_system_config()
# Use bootstrap config during initialization to avoid circular dependencies
_model_config = get_model_config_bootstrap()
_workflow_config = get_workflow_config()

# Import infrastructure services (respecting layer boundaries)
try:
    from infrastructure.azure_search.search_client import UnifiedSearchClient
except ImportError:
    UnifiedSearchClient = None

try:
    from infrastructure.azure_cosmos.cosmos_gremlin_client import (
        AzureCosmosGremlinClient,
    )
except ImportError:
    AzureCosmosGremlinClient = None

try:
    from infrastructure.azure_storage.storage_client import (
        UnifiedStorageClient as AzureStorageClient,
    )
except ImportError:
    AzureStorageClient = None

try:
    from infrastructure.azure_ml.ml_client import AzureMLClient
except ImportError:
    AzureMLClient = None

try:
    from config.settings import azure_settings
except ImportError:
    # Create minimal settings for testing
    class AzureSettings:
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_api_version = _model_config.api_version
        azure_search_endpoint = None
        azure_cosmos_endpoint = None
        azure_storage_account = None
        azure_ml_endpoint = None

    azure_settings = AzureSettings()

logger = logging.getLogger(__name__)

# ConsolidatedAzureServices now imported from agents.core.data_models

async def create_azure_service_container() -> ConsolidatedAzureServices:
    """
    Factory function to create and initialize consolidated Azure services.

    This function is used by dependency injection systems to provide
    properly initialized Azure services to agents and other components.
    """
    logger.info("🏭 Creating consolidated Azure service container...")

    # Create the container
    container = ConsolidatedAzureServices()

    # Initialize all services
    await container.initialize_all_services()

    return container


# Backward compatibility aliases
AzureServiceContainer = ConsolidatedAzureServices
create_azure_service_container = create_azure_service_container
