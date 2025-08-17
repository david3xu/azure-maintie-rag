"""
Universal Dependencies for Azure Universal RAG Multi-Agent System
================================================================

Centralized dependency management following PydanticAI best practices.
All agents share the same dependencies to avoid duplication and ensure consistency.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.core.credentials import TokenCredential
import os
from openai import AsyncAzureOpenAI

# Load environment variables early with explicit path
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env", override=True)

from agents.core.simple_config_manager import SimpleDynamicConfigManager
from agents.core.universal_models import (
    UniversalDomainAnalysis,
    UniversalProcessingConfiguration,
)


def get_azure_credential():
    """
    Get appropriate Azure credential based on environment.
    
    Automatically detects environment and uses appropriate authentication:
    - Container Apps: ManagedIdentityCredential 
    - Local/azd deployment: DefaultAzureCredential (Azure CLI)
    - Explicit control via USE_MANAGED_IDENTITY environment variable
    """
    # Check for explicit control via environment variable
    use_managed_identity_env = os.getenv("USE_MANAGED_IDENTITY")
    
    # Auto-detect environment if not explicitly set
    if use_managed_identity_env is None:
        # Check if running in Azure Container Apps (has AZURE_CLIENT_ID)
        client_id = os.getenv('AZURE_CLIENT_ID')
        if client_id:
            # Running in Container App - use managed identity
            use_managed_identity = True
        else:
            # Local development or azd deployment - use Azure CLI
            use_managed_identity = False
    else:
        # Explicit setting provided
        use_managed_identity = use_managed_identity_env.lower() == "true"
    
    if use_managed_identity:
        # Use managed identity (Container Apps)
        client_id = os.getenv('AZURE_CLIENT_ID')
        if client_id:
            return ManagedIdentityCredential(client_id=client_id)
        else:
            return ManagedIdentityCredential()
    else:
        # Use DefaultAzureCredential for local/azd deployment (includes Azure CLI)
        return DefaultAzureCredential(
            # Exclude problematic credential types for deployment scenarios
            exclude_cli_credential=False,  # Keep CLI credential for azd deployments
            exclude_managed_identity_credential=True,  # Exclude MI for local development
            exclude_visual_studio_code_credential=True  # Exclude VS Code for deployment
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
    credential: TokenCredential = field(default_factory=get_azure_credential)

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
                    # Pass the managed identity credential to the client
                    self.openai_client.credential = self.credential
                    self.openai_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["openai"] = True
            except Exception as e:
                print(f"Warning: OpenAI client initialization failed: {e}")
                services_status["openai"] = False

            # Initialize Cosmos DB (required by Knowledge Extraction)
            try:
                if not self.cosmos_client:
                    self.cosmos_client = SimpleCosmosGremlinClient()
                    # Pass the managed identity credential to the client
                    self.cosmos_client.credential = self.credential
                    self.cosmos_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["cosmos"] = True
            except Exception as e:
                print(f"Warning: Cosmos DB client initialization failed: {e}")
                services_status["cosmos"] = False

            # Initialize Search Client (required by Universal Search)
            try:
                if not self.search_client:
                    self.search_client = UnifiedSearchClient()
                    # Pass the managed identity credential to the client
                    self.search_client.credential = self.credential
                    self.search_client.ensure_initialized()  # Use synchronous ensure_initialized
                services_status["search"] = True
            except Exception as e:
                print(f"Warning: Search client initialization failed: {e}")
                services_status["search"] = False

            # Initialize Storage Client (optional)
            try:
                if not self.storage_client:
                    self.storage_client = SimpleStorageClient()
                    # Pass the managed identity credential to the client
                    self.storage_client.credential = self.credential
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

    async def cleanup(self) -> None:
        """
        Properly close all service connections to prevent connection leaks.

        This method should be called when shutting down the application
        or when dependencies are no longer needed.
        """
        try:
            # Close Cosmos DB Gremlin connection
            if self.cosmos_client and hasattr(self.cosmos_client, "close"):
                try:
                    self.cosmos_client.close()
                except Exception as e:
                    print(f"Warning: Could not close Cosmos client: {e}")

            # Close Azure OpenAI client if it has close method
            if self.openai_client and hasattr(self.openai_client, "close"):
                try:
                    await self.openai_client.close()
                except Exception as e:
                    print(f"Warning: Could not close OpenAI client: {e}")

            # Close storage client
            if self.storage_client and hasattr(self.storage_client, "close"):
                try:
                    self.storage_client.close()
                except Exception as e:
                    print(f"Warning: Could not close Storage client: {e}")

            # Close search client
            if self.search_client and hasattr(self.search_client, "close"):
                try:
                    self.search_client.close()
                except Exception as e:
                    print(f"Warning: Could not close Search client: {e}")

        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            # Only do sync cleanup in destructor
            if self.cosmos_client and hasattr(self.cosmos_client, "close"):
                self.cosmos_client.close()
        except Exception:
            pass  # Ignore errors during garbage collection


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


async def cleanup_universal_deps():
    """
    Cleanup global universal dependencies and close all connections.

    This function should be called at the end of scripts or application shutdown
    to prevent connection leak warnings.
    """
    global _universal_deps
    if _universal_deps is not None:
        await _universal_deps.cleanup()


def reset_universal_deps():
    """Reset dependencies for testing or reinitialization."""
    global _universal_deps
    _universal_deps = None
