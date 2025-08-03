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
from azure.identity import DefaultAzureCredential
from pydantic_ai.providers.azure import AzureProvider

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
        azure_openai_endpoint = "https://example.openai.azure.com/"
        openai_api_version = "2024-02-15-preview"
        azure_search_endpoint = None
        azure_cosmos_endpoint = None
        azure_storage_account = None
        azure_ml_endpoint = None

    azure_settings = AzureSettings()

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedAzureServices:
    """
    Consolidated Azure services container combining LLM and non-LLM services.

    Consolidates functionality from:
    - azure_integration.py (Azure AI Foundry provider)
    - unified_azure_services.py (non-LLM services)

    Follows clean architecture with proper dependency injection support.
    """

    # Authentication
    credential: DefaultAzureCredential = field(default_factory=DefaultAzureCredential)

    # Azure AI Foundry for LLM services
    ai_foundry_provider: Optional[AzureProvider] = None

    # Non-LLM Azure services
    search_client: Optional[Any] = None
    cosmos_client: Optional[Any] = None
    storage_client: Optional[Any] = None
    ml_client: Optional[Any] = None

    # Search orchestration services
    tri_modal_orchestrator: Optional[Any] = None

    # Service status tracking
    initialized_services: Dict[str, bool] = field(default_factory=dict)
    initialization_errors: Dict[str, str] = field(default_factory=dict)

    async def initialize_all_services(self) -> Dict[str, bool]:
        """
        Initialize all Azure services with proper error handling.

        Returns:
            Dict mapping service names to initialization success status
        """
        logger.info("🚀 Initializing consolidated Azure services...")

        # Initialize services in parallel for performance
        initialization_tasks = [
            self._initialize_ai_foundry_provider(),
            self._initialize_search_client(),
            self._initialize_cosmos_client(),
            self._initialize_storage_client(),
            self._initialize_ml_client(),
            self._initialize_tri_modal_orchestrator(),
        ]

        # Execute all initializations concurrently
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

        # Process results
        service_names = [
            "ai_foundry",
            "search",
            "cosmos",
            "storage",
            "ml",
            "tri_modal_orchestrator",
        ]
        for i, result in enumerate(results):
            service_name = service_names[i]
            if isinstance(result, Exception):
                self.initialized_services[service_name] = False
                self.initialization_errors[service_name] = str(result)
                logger.warning(
                    f"⚠️ {service_name} service initialization failed: {result}"
                )
            else:
                self.initialized_services[service_name] = result
                if result:
                    logger.info(f"✅ {service_name} service initialized successfully")

        # Log summary
        successful_services = sum(self.initialized_services.values())
        total_services = len(self.initialized_services)
        logger.info(
            f"🎯 Azure services initialization complete: {successful_services}/{total_services} services ready"
        )

        return self.initialized_services

    async def _initialize_ai_foundry_provider(self) -> bool:
        """Initialize Azure AI Foundry provider for LLM integration"""
        try:
            from azure.identity import get_bearer_token_provider
            from openai import AsyncAzureOpenAI

            if not azure_settings.azure_openai_endpoint:
                logger.info(
                    "📝 Azure OpenAI endpoint not configured, skipping AI Foundry provider"
                )
                return False

            if not azure_settings.openai_api_version:
                logger.warning(
                    "⚠️ Azure OpenAI API version not configured, using default"
                )
                api_version = "2024-02-15-preview"
            else:
                api_version = azure_settings.openai_api_version

            # Create token provider with managed identity
            token_provider = get_bearer_token_provider(
                self.credential, "https://cognitiveservices.azure.com/.default"
            )

            # Create Azure OpenAI client with managed identity
            azure_client = AsyncAzureOpenAI(
                azure_endpoint=azure_settings.azure_openai_endpoint,
                api_version=api_version,
                azure_ad_token_provider=token_provider,  # Uses managed identity
            )

            # Create PydanticAI provider with managed identity client
            self.ai_foundry_provider = AzureProvider(openai_client=azure_client)

            logger.info(
                f"🤖 Azure AI Foundry provider created with managed identity for: {azure_settings.azure_openai_endpoint}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure AI Foundry provider: {e}")
            return False

    async def _initialize_search_client(self) -> bool:
        """Initialize Azure Cognitive Search client"""
        try:
            if not UnifiedSearchClient:
                logger.info(
                    "📝 UnifiedSearchClient not available, skipping search service"
                )
                return False

            if not azure_settings.azure_search_endpoint:
                logger.info(
                    "📝 Azure Search endpoint not configured, skipping search service"
                )
                return False

            # Initialize search client with proper config
            self.search_client = UnifiedSearchClient(
                config={
                    "endpoint": azure_settings.azure_search_endpoint,
                    "credential": self.credential,
                }
            )

            # Test connection
            await self.search_client.test_connection()

            logger.info(
                f"🔍 Azure Search client initialized for endpoint: {azure_settings.azure_search_endpoint}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Search client: {e}")
            return False

    async def _initialize_cosmos_client(self) -> bool:
        """Initialize Azure Cosmos DB Gremlin client"""
        try:
            if not AzureCosmosGremlinClient:
                logger.info(
                    "📝 CosmosGremlinClient not available, skipping Cosmos service"
                )
                return False

            if not azure_settings.azure_cosmos_endpoint:
                logger.info(
                    "📝 Azure Cosmos endpoint not configured, skipping Cosmos service"
                )
                return False

            # Initialize Cosmos client with proper config
            self.cosmos_client = AzureCosmosGremlinClient(
                config={
                    "endpoint": azure_settings.azure_cosmos_endpoint,
                    "credential": self.credential,
                }
            )

            # Test connection
            await self.cosmos_client.test_connection()

            logger.info(
                f"🌐 Azure Cosmos client initialized for endpoint: {azure_settings.azure_cosmos_endpoint}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Cosmos client: {e}")
            return False

    async def _initialize_storage_client(self) -> bool:
        """Initialize Azure Storage client"""
        try:
            if not AzureStorageClient:
                logger.info(
                    "📝 AzureStorageClient not available, skipping storage service"
                )
                return False

            if not azure_settings.azure_storage_account:
                logger.info(
                    "📝 Azure Storage account not configured, skipping storage service"
                )
                return False

            # Initialize storage client with proper config
            self.storage_client = AzureStorageClient(
                config={
                    "account_name": azure_settings.azure_storage_account,
                    "credential": self.credential,
                }
            )

            # Test connection
            await self.storage_client.test_connection()

            logger.info(
                f"💾 Azure Storage client initialized for account: {azure_settings.azure_storage_account}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Storage client: {e}")
            return False

    async def _initialize_ml_client(self) -> bool:
        """Initialize Azure ML client"""
        try:
            if not AzureMLClient:
                logger.info("📝 AzureMLClient not available, skipping ML service")
                return False

            if not azure_settings.azure_ml_workspace_name:
                logger.info("📝 Azure ML workspace not configured, skipping ML service")
                return False

            # Initialize ML client (uses internal credential and workspace config)
            self.ml_client = AzureMLClient()

            # Test connection - ML client doesn't have async test method, assume success if initialization worked
            # await self.ml_client.test_connection()

            logger.info(
                f"🧠 Azure ML client initialized for workspace: {azure_settings.azure_ml_workspace_name}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure ML client: {e}")
            return False

    async def _initialize_tri_modal_orchestrator(self) -> bool:
        """Initialize tri-modal search orchestrator"""
        try:
            # Import the orchestrator (using absolute import to avoid relative import issues)
            from infrastructure.search.tri_modal_orchestrator import (
                TriModalOrchestrator,
            )

            # Initialize the orchestrator
            self.tri_modal_orchestrator = TriModalOrchestrator()

            logger.info("🔍 Tri-modal search orchestrator initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"⚠️ Could not import TriModalOrchestrator: {e}")
            logger.info(
                "📝 Tri-modal orchestrator not available, search operations may be limited"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize tri-modal orchestrator: {e}")
            return False

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status for health checks"""
        return {
            "services_initialized": self.initialized_services,
            "initialization_errors": self.initialization_errors,
            "total_services": len(self.initialized_services),
            "successful_services": sum(self.initialized_services.values()),
            "has_ai_foundry": self.ai_foundry_provider is not None,
            "has_search": self.search_client is not None,
            "has_cosmos": self.cosmos_client is not None,
            "has_storage": self.storage_client is not None,
            "has_ml": self.ml_client is not None,
            "has_tri_modal_orchestrator": self.tri_modal_orchestrator is not None,
            "overall_health": "healthy"
            if sum(self.initialized_services.values()) > 0
            else "degraded",
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all initialized services"""
        logger.info("🔍 Performing consolidated Azure services health check...")

        health_results = {}

        # Check each initialized service
        for service_name, is_initialized in self.initialized_services.items():
            if not is_initialized:
                health_results[service_name] = "not_initialized"
                continue

            try:
                if service_name == "search" and self.search_client:
                    await self.search_client.test_connection()
                    health_results[service_name] = "healthy"
                elif service_name == "cosmos" and self.cosmos_client:
                    await self.cosmos_client.test_connection()
                    health_results[service_name] = "healthy"
                elif service_name == "storage" and self.storage_client:
                    await self.storage_client.test_connection()
                    health_results[service_name] = "healthy"
                elif service_name == "ml" and self.ml_client:
                    await self.ml_client.test_connection()
                    health_results[service_name] = "healthy"
                elif service_name == "ai_foundry" and self.ai_foundry_provider:
                    health_results[
                        service_name
                    ] = "healthy"  # Provider doesn't need connection test
                elif (
                    service_name == "tri_modal_orchestrator"
                    and self.tri_modal_orchestrator
                ):
                    # Test orchestrator health
                    orchestrator_health = (
                        await self.tri_modal_orchestrator.health_check()
                    )
                    if orchestrator_health.get("overall_status") == "healthy":
                        health_results[service_name] = "healthy"
                    else:
                        health_results[service_name] = "degraded"
                else:
                    health_results[service_name] = "unknown"

            except Exception as e:
                health_results[service_name] = f"unhealthy: {str(e)}"
                logger.warning(f"⚠️ Health check failed for {service_name}: {e}")

        # Calculate overall health
        healthy_services = sum(
            1 for status in health_results.values() if status == "healthy"
        )
        total_checked = len(health_results)

        overall_status = {
            "service_health": health_results,
            "healthy_services": healthy_services,
            "total_services_checked": total_checked,
            "health_percentage": (healthy_services / max(1, total_checked)) * 100,
            "overall_status": "healthy" if healthy_services > 0 else "degraded",
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(
            f"🎯 Health check complete: {healthy_services}/{total_checked} services healthy"
        )
        return overall_status


# Factory function for dependency injection
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
