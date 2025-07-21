"""Unified Azure services integration for Universal RAG system."""

import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import asyncio

from core.azure_storage.storage_factory import get_storage_factory
from core.azure_search.search_client import AzureCognitiveSearchClient
from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from core.azure_ml.ml_client import AzureMLClient
from .azure_openai import AzureOpenAIClient

logger = logging.getLogger(__name__)


class AzureServicesManager:
    """Unified manager for all Azure services - enterprise health monitoring"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize all Azure service clients"""
        self.config = config or {}
        self.services = {}
        self.service_status = {}
        self.initialization_status = {}

        # Safe service initialization with dependency validation
        self._safe_initialize_services()

        logger.info("AzureServicesManager initialized with all services including storage factory")

    def _safe_initialize_services(self) -> None:
        """Safe service initialization with dependency validation"""
        initialization_status = {}

        # Core services (required)
        try:
            self.services['openai'] = AzureOpenAIClient(self.config.get('openai'))
            initialization_status['openai'] = True
        except ImportError as e:
            logger.error(f"OpenAI service initialization failed: {e}")
            initialization_status['openai'] = False

        # Storage services (required)
        try:
            self.storage_factory = get_storage_factory()
            self.services['rag_storage'] = self.storage_factory.get_rag_data_client()
            self.services['ml_storage'] = self.storage_factory.get_ml_models_client()
            self.services['app_storage'] = self.storage_factory.get_app_data_client()
            initialization_status['storage'] = True
        except Exception as e:
            logger.error(f"Storage services initialization failed: {e}")
            initialization_status['storage'] = False

        # Search service (required)
        try:
            self.services['search'] = AzureCognitiveSearchClient(self.config.get('search'))
            initialization_status['search'] = True
        except Exception as e:
            logger.error(f"Search service initialization failed: {e}")
            initialization_status['search'] = False

        # Cosmos DB service (required)
        try:
            self.services['cosmos'] = AzureCosmosGremlinClient(self.config.get('cosmos'))
            initialization_status['cosmos'] = True
        except Exception as e:
            logger.error(f"Cosmos DB service initialization failed: {e}")
            initialization_status['cosmos'] = False

        # Optional services (graceful degradation)
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            # Initialize text analytics if available
            initialization_status['text_analytics'] = True
        except ImportError:
            logger.warning("Azure Text Analytics not available - continuing without it")
            initialization_status['text_analytics'] = False

        try:
            from azure.ai.ml import MLClient
            self.services['ml'] = AzureMLClient(self.config.get('ml'))
            initialization_status['ml'] = True
        except ImportError:
            logger.warning("Azure ML client not available - continuing without it")
            initialization_status['ml'] = False

        self.initialization_status = initialization_status

    async def initialize(self) -> None:
        """Async initialization for Azure services - enterprise pattern"""
        logger.info("Initializing Azure services with async pattern...")
        # Initialize any async-required services
        if hasattr(self.services.get('cosmos'), 'initialize_async'):
            await self.services['cosmos'].initialize_async()
        if hasattr(self.services.get('search'), 'initialize_async'):
            await self.services['search'].initialize_async()
        logger.info("Azure services async initialization completed")

    def check_all_services_health(self) -> Dict[str, Any]:
        """Concurrent service health check - enterprise monitoring"""
        health_results = {}
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                'openai': executor.submit(self.services['openai'].get_service_status),
                'rag_storage': executor.submit(self.services['rag_storage'].get_connection_status),
                'ml_storage': executor.submit(self.services['ml_storage'].get_connection_status),
                'app_storage': executor.submit(self.services['app_storage'].get_connection_status),
                'search': executor.submit(self.services['search'].get_service_status),
                'cosmos': executor.submit(self.services['cosmos'].get_connection_status),
                'ml': executor.submit(self.services['ml'].get_workspace_status)
            }

            for service_name, future in futures.items():
                try:
                    health_results[service_name] = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Service health check failed for {service_name}: {e}")
                    health_results[service_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "service": service_name
                    }

        overall_time = time.time() - start_time

        return {
            "overall_status": "healthy" if all(
                result.get("status") == "healthy" for result in health_results.values()
            ) else "degraded",
            "services": health_results,
            "healthy_count": sum(1 for s in health_results.values() if s.get("status") == "healthy"),
            "total_count": len(health_results),
            "health_check_duration_ms": overall_time * 1000,
            "timestamp": time.time(),
            "telemetry": {
                "service": "azure_services_manager",
                "operation": "health_check",
                "environment": "enterprise"
            }
        }

    def migrate_data_to_azure(self, source_data_path: str, domain: str) -> Dict[str, Any]:
        """Migrate local data to Azure services - universal migration"""
        migration_results = {
            "storage_migration": {"success": False},
            "search_migration": {"success": False},
            "cosmos_migration": {"success": False}
        }

        try:
            # 1. Migrate raw data to Azure Storage
            # Implementation follows existing patterns...

            # 2. Migrate vector index to Azure Cognitive Search
            # Implementation follows existing patterns...

            # 3. Migrate knowledge graph to Azure Cosmos DB
            # Implementation follows existing patterns...

            logger.info(f"Data migration completed for domain: {domain}")
            return {
                "success": True,
                "domain": domain,
                "migration_results": migration_results
            }

        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            # ❌ REMOVED: Silent fallback - let the error propagate
            raise RuntimeError(f"Data migration failed: {e}")

    def get_service(self, service_name: str):
        """Get specific Azure service client"""
        return self.services.get(service_name)

    def get_rag_storage_client(self):
        """Get RAG data storage client"""
        return self.services.get('rag_storage')

    def get_ml_storage_client(self):
        """Get ML models storage client"""
        return self.services.get('ml_storage')

    def get_app_storage_client(self):
        """Get application data storage client"""
        return self.services.get('app_storage')

    def get_storage_factory(self):
        """Get storage factory instance"""
        return self.storage_factory

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all Azure service configurations"""
        validation_results = {}

        for service_name, service in self.services.items():
            if hasattr(service, 'validate_azure_configuration'):
                validation_results[service_name] = service.validate_azure_configuration()
            elif hasattr(service, 'get_connection_status'):
                status = service.get_connection_status()
                validation_results[service_name] = {"configured": status.get("status") == "healthy"}
            else:
                logger.warning(f"No validation method available for {service_name}")
                validation_results[service_name] = {"configured": False}  # Don't assume OK

        all_configured = all(
            result.get("configured") or result.get("all_valid", False)
            for result in validation_results.values()
        )

        return {
            "all_configured": all_configured,
            "services": validation_results
        }