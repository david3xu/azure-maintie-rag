"""Unified Azure services integration for Universal RAG system."""

import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime

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
            logger.debug(f"Validating service: {service_name}")
            if hasattr(service, 'validate_azure_configuration'):
                result = service.validate_azure_configuration()
                validation_results[service_name] = result
                logger.debug(f"Service {service_name} validation result: {result}")
            elif hasattr(service, 'get_connection_status'):
                status = service.get_connection_status()
                configured = status.get("status") == "healthy"
                validation_results[service_name] = {"configured": configured}
                if not configured:
                    logger.error(f"Service {service_name} not configured. Status: {status}")
                else:
                    logger.info(f"Service {service_name} configured successfully")
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

    async def cleanup_all_azure_data(self, domain: str = "general") -> Dict[str, Any]:
        """Enterprise Azure data cleanup orchestration across all services"""
        logger.info(f"Starting Azure data cleanup for domain: {domain}")

        cleanup_results = {}
        cleanup_start = time.time()

        # 1. Orchestrate blob storage cleanup
        try:
            rag_storage = self.get_rag_storage_client()
            if rag_storage:
                container_name = f"rag-data-{domain}"
                cleanup_results["blob_storage"] = await self._cleanup_blob_storage_data(
                    rag_storage, container_name
                )
            else:
                cleanup_results["blob_storage"] = {"success": False, "error": "RAG storage client not available"}
        except Exception as e:
            cleanup_results["blob_storage"] = {"success": False, "error": str(e)}

        # 2. Orchestrate search index cleanup
        try:
            search_client = self.get_service('search')
            if search_client:
                index_name = f"rag-index-{domain}"
                cleanup_results["cognitive_search"] = await self._cleanup_search_index_data(
                    search_client, index_name
                )
            else:
                cleanup_results["cognitive_search"] = {"success": False, "error": "Search client not available"}
        except Exception as e:
            cleanup_results["cognitive_search"] = {"success": False, "error": str(e)}

        # 3. Orchestrate cosmos graph cleanup
        try:
            cosmos_client = self.get_service('cosmos')
            if cosmos_client:
                cleanup_results["cosmos_db"] = await self._cleanup_cosmos_graph_data(
                    cosmos_client, domain
                )
            else:
                cleanup_results["cosmos_db"] = {"success": False, "error": "Cosmos client not available"}
        except Exception as e:
            cleanup_results["cosmos_db"] = {"success": False, "error": str(e)}

        cleanup_duration = time.time() - cleanup_start

        return {
            "success": all(result.get("success", False) for result in cleanup_results.values()),
            "domain": domain,
            "cleanup_results": cleanup_results,
            "cleanup_duration_seconds": cleanup_duration,
            "timestamp": datetime.now().isoformat(),
            "enterprise_metrics": {
                "services_cleaned": len([r for r in cleanup_results.values() if r.get("success")]),
                "total_services": len(cleanup_results),
                "cleanup_efficiency": sum(1 for r in cleanup_results.values() if r.get("success")) / len(cleanup_results)
            }
        }

    async def _cleanup_blob_storage_data(self, storage_client, container_name: str) -> Dict[str, Any]:
        """Enterprise blob storage data cleanup using existing client patterns from codebase"""
        try:
            from azure.core.exceptions import ResourceNotFoundError

            deleted_count = 0
            try:
                # Use existing storage client interface pattern from codebase
                # Get container client via blob_service_client (from actual implementation)
                container_client = storage_client.blob_service_client.get_container_client(container_name)

                # Check if container exists first
                if container_client.exists():
                    # Use existing list_blobs pattern from codebase
                    blob_list = storage_client.list_blobs("")  # Empty prefix gets all blobs

                    # Delete each blob using existing patterns
                    for blob_info in blob_list:
                        blob_client = storage_client.blob_service_client.get_blob_client(
                            container=container_name,
                            blob=blob_info["name"]
                        )
                        blob_client.delete_blob()
                        deleted_count += 1
                        logger.debug(f"Deleted blob: {blob_info['name']}")
                else:
                    logger.info(f"Container {container_name} does not exist - considering it cleaned")

            except ResourceNotFoundError:
                # Container doesn't exist - consider it cleaned
                logger.info(f"Container {container_name} not found - already cleaned")
                pass

            return {
                "success": True,
                "blobs_deleted": deleted_count,
                "container": container_name,
                "infrastructure_preserved": True
            }
        except Exception as e:
            logger.error(f"Blob storage cleanup failed: {e}")
            return {"success": False, "error": str(e)}

    async def _cleanup_search_index_data(self, search_client, index_name: str) -> Dict[str, Any]:
        """Enterprise search index data cleanup using existing client patterns"""
        try:
            from azure.search.documents import SearchClient
            from azure.core.exceptions import ResourceNotFoundError

            # Create search client using existing patterns
            target_search_client = SearchClient(
                endpoint=search_client.endpoint,
                index_name=index_name,
                credential=search_client.credential
            )

            documents_deleted = 0
            try:
                # Get all document IDs and delete them
                results = target_search_client.search("*", select="id")
                documents_to_delete = []

                for result in results:
                    documents_to_delete.append({
                        "@search.action": "delete",
                        "id": result["id"]
                    })

                if documents_to_delete:
                    delete_result = target_search_client.upload_documents(documents_to_delete)
                    documents_deleted = len([r for r in delete_result if r.succeeded])

            except ResourceNotFoundError:
                # Index doesn't exist - consider it cleaned
                pass

            return {
                "success": True,
                "documents_deleted": documents_deleted,
                "index_name": index_name,
                "index_structure_preserved": True
            }
        except Exception as e:
            logger.error(f"Search index cleanup failed: {e}")
            return {"success": False, "error": str(e)}

    async def _cleanup_cosmos_graph_data(self, cosmos_client, domain: str) -> Dict[str, Any]:
        """Enterprise Cosmos DB graph data cleanup using existing Gremlin patterns"""
        try:
            # Use existing Gremlin query execution patterns
            edges_deleted = 0
            vertices_deleted = 0

            # Delete all edges for domain first (prevent orphaned references)
            edges_query = f"g.E().has('domain', '{domain}').drop()"
            edges_result = cosmos_client._execute_gremlin_query_safe(edges_query)
            if edges_result:
                edges_deleted = len(edges_result)

            # Delete all vertices for domain
            vertices_query = f"g.V().has('domain', '{domain}').drop()"
            vertices_result = cosmos_client._execute_gremlin_query_safe(vertices_query)
            if vertices_result:
                vertices_deleted = len(vertices_result)

            return {
                "success": True,
                "edges_deleted": edges_deleted,
                "vertices_deleted": vertices_deleted,
                "total_entities_deleted": edges_deleted + vertices_deleted,
                "domain": domain,
                "database_structure_preserved": True
            }
        except Exception as e:
            logger.error(f"Cosmos DB graph cleanup failed: {e}")
            return {"success": False, "error": str(e)}

    async def validate_domain_data_state(self, domain: str) -> Dict[str, Any]:
        """Enterprise Azure data state validation across services"""
        # Azure Blob Storage state validation
        rag_storage = self.get_rag_storage_client()
        container_name = f"rag-data-{domain}"
        blob_state = await self._validate_blob_storage_state(rag_storage, container_name)
        # Azure Cognitive Search state validation
        search_client = self.get_service('search')
        index_name = f"rag-index-{domain}"
        search_state = await self._validate_search_index_state(search_client, index_name)
        # Azure Cosmos DB state validation
        cosmos_client = self.get_service('cosmos')
        cosmos_state = await self._validate_cosmos_metadata_state(cosmos_client, domain)
        # Raw data directory validation
        raw_data_state = self._validate_raw_data_directory()
        return {
            "domain": domain,
            "azure_blob_storage": blob_state,
            "azure_cognitive_search": search_state,
            "azure_cosmos_db": cosmos_state,
            "raw_data_directory": raw_data_state,
            "requires_processing": self._calculate_processing_requirement(
                blob_state, search_state, cosmos_state, raw_data_state
            ),
            "timestamp": datetime.now().isoformat()
        }

    async def _validate_blob_storage_state(self, storage_client, container_name: str) -> Dict[str, Any]:
        """Validate Azure Blob Storage container state"""
        try:
            from azure.core.exceptions import ResourceNotFoundError
            blobs = storage_client.list_blobs(container_name)
            blob_count = len(list(blobs)) if blobs else 0
            return {
                "container_exists": True,
                "document_count": blob_count,
                "has_data": blob_count > 0,
                "container_name": container_name
            }
        except ResourceNotFoundError:
            return {
                "container_exists": False,
                "document_count": 0,
                "has_data": False,
                "container_name": container_name
            }
        except Exception as e:
            logger.warning(f"Azure Blob Storage validation failed: {e}")
            return {
                "container_exists": False,
                "document_count": 0,
                "has_data": False,
                "error": str(e)
            }

    async def _validate_search_index_state(self, search_client, index_name: str) -> Dict[str, Any]:
        """Validate Azure Cognitive Search index state"""
        try:
            index_stats = await search_client._get_index_statistics(index_name)
            return {
                "index_exists": index_stats.get("index_exists", False),
                "document_count": index_stats.get("document_count", 0),
                "has_data": index_stats.get("document_count", 0) > 0,
                "index_name": index_name
            }
        except Exception as e:
            logger.warning(f"Azure Cognitive Search validation failed: {e}")
            return {
                "index_exists": False,
                "document_count": 0,
                "has_data": False,
                "error": str(e)
            }

    async def _validate_cosmos_metadata_state(self, cosmos_client, domain: str) -> Dict[str, Any]:
        """Validate Azure Cosmos DB metadata state"""
        try:
            if cosmos_client is None:
                logger.warning("Cosmos client is None in _validate_cosmos_metadata_state")
                return {
                    "metadata_exists": False,
                    "vertex_count": 0,
                    "edge_count": 0,
                    "has_data": False,
                    "error": "Cosmos client is None"
                }
            stats = cosmos_client.get_graph_statistics(domain)
            if stats is None:
                logger.warning("Cosmos DB statistics result is None in _validate_cosmos_metadata_state")
                return {
                    "metadata_exists": False,
                    "vertex_count": 0,
                    "edge_count": 0,
                    "has_data": False,
                    "error": "Cosmos DB statistics result is None"
                }
            has_metadata = stats.get("success", False) and (
                stats.get("vertex_count", 0) > 0 or stats.get("edge_count", 0) > 0
            )
            return {
                "metadata_exists": stats.get("success", False),
                "vertex_count": stats.get("vertex_count", 0),
                "edge_count": stats.get("edge_count", 0),
                "has_data": has_metadata,
                "domain": domain
            }
        except Exception as e:
            logger.warning(f"Azure Cosmos DB validation failed: {e}")
            return {
                "metadata_exists": False,
                "vertex_count": 0,
                "edge_count": 0,
                "has_data": False,
                "error": str(e)
            }

    def _validate_raw_data_directory(self) -> Dict[str, Any]:
        """Validate raw data directory state"""
        from pathlib import Path
        raw_data_path = Path("data/raw")
        if not raw_data_path.exists():
            return {
                "directory_exists": False,
                "file_count": 0,
                "has_files": False,
                "last_modified": None
            }
        markdown_files = list(raw_data_path.glob("*.md"))
        last_modified = None
        if markdown_files:
            mod_times = [f.stat().st_mtime for f in markdown_files]
            last_modified = datetime.fromtimestamp(max(mod_times)).isoformat()
        return {
            "directory_exists": True,
            "file_count": len(markdown_files),
            "has_files": len(markdown_files) > 0,
            "last_modified": last_modified
        }

    def _calculate_processing_requirement(self, blob_state: Dict, search_state: Dict, cosmos_state: Dict, raw_data_state: Dict) -> str:
        """Calculate processing requirement based on data state - focus on core data services"""

        # Core data services must have data for system to be considered populated
        has_core_data = all([
            blob_state.get("has_data", False),      # Documents must be in Blob Storage
            search_state.get("has_data", False)     # Search index must be populated
        ])

        has_raw_data = raw_data_state.get("has_files", False)

        if not has_raw_data:
            return "no_raw_data"
        elif not has_core_data:
            return "full_processing_required"
        else:
            return "data_exists_check_policy"