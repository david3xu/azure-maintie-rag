"""
Universal Dependencies for Azure Universal RAG Multi-Agent System
================================================================

Centralized dependency management following PydanticAI best practices.
All agents share the same dependencies to avoid duplication and ensure consistency.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from azure.identity import DefaultAzureCredential
from openai import AsyncAzureOpenAI

from agents.core.simple_config_manager import SimpleDynamicConfigManager
from agents.core.universal_models import (
    UniversalDomainAnalysis,
    UniversalProcessingConfiguration,
)
from infrastructure.azure_cosmos.cosmos_gremlin_client import SimpleCosmosGremlinClient
# Optional GNN import - use lazy loading to avoid PyTorch resource issues
def _get_gnn_client_class():
    """Lazy import of GNN client to avoid PyTorch resource exhaustion."""
    try:
        from infrastructure.azure_ml.gnn_inference_client import GNNInferenceClient
        return GNNInferenceClient
    except ImportError:
        return None

GNN_AVAILABLE = False  # Will be set to True when GNN client is successfully accessed
from infrastructure.azure_monitoring.app_insights_client import (
    AzureApplicationInsightsClient,
)
from infrastructure.azure_openai.openai_client import UnifiedAzureOpenAIClient
from infrastructure.azure_search.search_client import UnifiedSearchClient
from infrastructure.azure_storage.storage_client import SimpleStorageClient


@dataclass
class UniversalDeps:
    """
    Centralized dependencies for all Universal RAG agents.

    Following PydanticAI best practices:
    - Shared across all agents to avoid duplication
    - Initialized once and reused
    - Supports proper agent delegation patterns
    - Enables clean dependency injection
    """

    # Configuration Management
    config_manager: SimpleDynamicConfigManager = field(
        default_factory=SimpleDynamicConfigManager
    )

    # Azure Authentication (shared across all services)
    credential: DefaultAzureCredential = field(default_factory=DefaultAzureCredential)

    # Core Azure Services
    openai_client: Optional[UnifiedAzureOpenAIClient] = None
    cosmos_client: Optional[SimpleCosmosGremlinClient] = None
    search_client: Optional[UnifiedSearchClient] = None
    storage_client: Optional[SimpleStorageClient] = None

    # Advanced Services (GNN client is dynamically typed due to optional PyTorch)
    gnn_client: Optional[Any] = None
    monitoring_client: Optional[AzureApplicationInsightsClient] = None

    # Service Status
    _initialized: bool = field(default=False, init=False)
    _initialization_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def initialize_all_services(self) -> dict[str, bool]:
        """
        Initialize all Azure services with proper error handling.

        Returns:
            dict: Service initialization status
        """
        async with self._initialization_lock:
            if self._initialized:
                return await self._get_service_status()

            services_status = {}

            # Initialize OpenAI Client (required by all agents)
            try:
                if not self.openai_client:
                    self.openai_client = UnifiedAzureOpenAIClient()
                    self.openai_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["openai"] = True
            except Exception as e:
                print(f"Warning: OpenAI client initialization failed: {e}")
                services_status["openai"] = False

            # Initialize Cosmos DB (required by Knowledge Extraction)
            try:
                if not self.cosmos_client:
                    self.cosmos_client = SimpleCosmosGremlinClient()
                    self.cosmos_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["cosmos"] = True
            except Exception as e:
                print(f"Warning: Cosmos DB client initialization failed: {e}")
                services_status["cosmos"] = False

            # Initialize Search Client (required by Universal Search)
            try:
                if not self.search_client:
                    self.search_client = UnifiedSearchClient()
                    self.search_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["search"] = True
            except Exception as e:
                print(f"Warning: Search client initialization failed: {e}")
                services_status["search"] = False

            # Initialize Storage Client (optional)
            try:
                if not self.storage_client:
                    self.storage_client = SimpleStorageClient()
                    self.storage_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["storage"] = True
            except Exception as e:
                print(f"Warning: Storage client initialization failed: {e}")
                services_status["storage"] = False

            # Initialize GNN Client (optional, lazy-loaded to avoid PyTorch issues)
            try:
                if not self.gnn_client:
                    GNNInferenceClient = _get_gnn_client_class()
                    if GNNInferenceClient is not None:
                        self.gnn_client = GNNInferenceClient()
                        if hasattr(self.gnn_client, "ensure_initialized"):
                            self.gnn_client.ensure_initialized()  # Use synchronous if available
                        services_status["gnn"] = True
                        global GNN_AVAILABLE
                        GNN_AVAILABLE = True
                    else:
                        services_status["gnn"] = False
                else:
                    services_status["gnn"] = True
            except Exception as e:
                print(f"Warning: GNN client initialization failed: {e}")
                services_status["gnn"] = False

            # Initialize Monitoring (optional)
            try:
                if not self.monitoring_client:
                    self.monitoring_client = AzureApplicationInsightsClient()
                    if hasattr(self.monitoring_client, "ensure_initialized"):
                        self.monitoring_client.ensure_initialized()  # Use synchronous if available
                    # Monitoring client might not inherit from BaseAzureClient
                services_status["monitoring"] = True
            except Exception as e:
                print(f"Warning: Monitoring client initialization failed: {e}")
                services_status["monitoring"] = False

            self._initialized = True
            return services_status

    async def _get_service_status(self) -> dict[str, bool]:
        """Get current service status."""
        return {
            "openai": self.openai_client is not None,
            "cosmos": self.cosmos_client is not None,
            "search": self.search_client is not None,
            "storage": self.storage_client is not None,
            "gnn": self.gnn_client is not None,
            "monitoring": self.monitoring_client is not None,
        }

    def get_available_services(self) -> list[str]:
        """Get list of successfully initialized services."""
        # Use synchronous service status check to avoid nested event loops
        service_status = {
            "openai": self.openai_client is not None,
            "cosmos": self.cosmos_client is not None,
            "search": self.search_client is not None,
            "storage": self.storage_client is not None,
            "gnn": self.gnn_client is not None,
            "monitoring": self.monitoring_client is not None,
        }
        return [service for service, status in service_status.items() if status]

    def is_service_available(self, service_name: str) -> bool:
        """Check if a specific service is available."""
        service_map = {
            "openai": self.openai_client,
            "cosmos": self.cosmos_client,
            "search": self.search_client,
            "storage": self.storage_client,
            "gnn": self.gnn_client,
            "monitoring": self.monitoring_client,
        }
        return service_map.get(service_name) is not None


# Private singleton instance for proper dependency injection
_universal_deps: Optional[UniversalDeps] = None


async def get_universal_deps() -> UniversalDeps:
    """
    Factory function to get initialized universal dependencies (async version).

    This function ensures all services are properly initialized
    before returning the dependencies to agents.
    Implements proper dependency injection pattern.
    """
    global _universal_deps
    if _universal_deps is None:
        _universal_deps = UniversalDeps()
        await _universal_deps.initialize_all_services()
    elif not _universal_deps._initialized:
        await _universal_deps.initialize_all_services()
    return _universal_deps


def get_universal_deps_sync() -> UniversalDeps:
    """
    Factory function to get universal dependencies synchronously.

    This creates the UniversalDeps instance but does NOT initialize
    Azure services. Suitable for testing and synchronous contexts.
    For production use, prefer get_universal_deps() async version.
    """
    global _universal_deps
    if _universal_deps is None:
        _universal_deps = UniversalDeps()
        # Note: Services are not initialized - call initialize_all_services() separately
    return _universal_deps


def reset_universal_deps():
    """Reset dependencies for testing or reinitialization."""
    global _universal_deps
    _universal_deps = None
