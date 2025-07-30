"""
Cleanup Service
Handles Azure data cleanup, maintenance, and resource management
Clean services architecture implementation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from config.settings import azure_settings

logger = logging.getLogger(__name__)


class CleanupService:
    """Service for Azure resource cleanup and maintenance"""
    
    def __init__(self, infrastructure_service):
        """Initialize cleanup service with infrastructure dependencies"""
        self.infrastructure = infrastructure_service
    
    async def cleanup_all_azure_data(self, domain: str = "general") -> Dict[str, Any]:
        """
        Comprehensive cleanup of all Azure data for a domain
        Removes data from Storage, Search, and Cosmos DB
        """
        start_time = datetime.now()
        cleanup_results = {
            "domain": domain,
            "start_time": start_time.isoformat(),
            "status": "in_progress",
            "cleanup_operations": {},
            "summary": {
                "total_operations": 3,
                "successful_operations": 0,
                "failed_operations": 0
            }
        }
        
        try:
            logger.info(f"🧹 Starting comprehensive cleanup for domain: {domain}")
            
            # Cleanup Azure Blob Storage
            storage_result = await self._cleanup_blob_storage_data(domain)
            cleanup_results["cleanup_operations"]["storage"] = storage_result
            if storage_result.get("success"):
                cleanup_results["summary"]["successful_operations"] += 1
            else:
                cleanup_results["summary"]["failed_operations"] += 1
            
            # Cleanup Azure Cognitive Search
            search_result = await self._cleanup_search_data(domain)
            cleanup_results["cleanup_operations"]["search"] = search_result
            if search_result.get("success"):
                cleanup_results["summary"]["successful_operations"] += 1
            else:
                cleanup_results["summary"]["failed_operations"] += 1
            
            # Cleanup Azure Cosmos DB
            cosmos_result = await self._cleanup_cosmos_data(domain)
            cleanup_results["cleanup_operations"]["cosmos"] = cosmos_result
            if cosmos_result.get("success"):
                cleanup_results["summary"]["successful_operations"] += 1
            else:
                cleanup_results["summary"]["failed_operations"] += 1
            
            # Determine overall status
            if cleanup_results["summary"]["successful_operations"] == 3:
                cleanup_results["status"] = "completed"
                logger.info("✅ All cleanup operations completed successfully")
            elif cleanup_results["summary"]["successful_operations"] > 0:
                cleanup_results["status"] = "partial_success"
                logger.warning("⚠️ Some cleanup operations failed")
            else:
                cleanup_results["status"] = "failed"
                logger.error("❌ All cleanup operations failed")
            
            cleanup_results["end_time"] = datetime.now().isoformat()
            cleanup_results["duration"] = str(datetime.now() - start_time)
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Critical error during cleanup: {e}")
            cleanup_results["status"] = "error"
            cleanup_results["error"] = str(e)
            cleanup_results["end_time"] = datetime.now().isoformat()
            return cleanup_results
    
    async def _cleanup_blob_storage_data(self, domain: str) -> Dict[str, Any]:
        """Cleanup Azure Blob Storage data for domain"""
        try:
            storage_client = self.infrastructure.get_rag_storage_client()
            if not storage_client:
                return {"success": False, "error": "Storage client not available"}
            
            container_name = f"{azure_settings.azure_blob_container}-{domain}"
            
            # List all blobs in the domain container
            try:
                blobs = await storage_client.list_blobs(container_name)
                deleted_blobs = []
                failed_deletions = []
                
                for blob in blobs:
                    try:
                        delete_result = await storage_client.delete_blob(container_name, blob.name)
                        if delete_result:
                            deleted_blobs.append(blob.name)
                        else:
                            failed_deletions.append(blob.name)
                    except Exception as e:
                        logger.error(f"Failed to delete blob {blob.name}: {e}")
                        failed_deletions.append(blob.name)
                
                # Optionally delete the container if empty
                if len(deleted_blobs) > 0 and len(failed_deletions) == 0:
                    try:
                        await storage_client.delete_container(container_name)
                        logger.info(f"Container {container_name} deleted")
                    except Exception as e:
                        logger.warning(f"Failed to delete container {container_name}: {e}")
                
                return {
                    "success": len(failed_deletions) == 0,
                    "container": container_name,
                    "deleted_blobs": len(deleted_blobs),
                    "failed_deletions": len(failed_deletions),
                    "failed_files": failed_deletions
                }
                
            except Exception as e:
                if "ContainerNotFound" in str(e):
                    return {
                        "success": True,
                        "message": f"Container {container_name} does not exist (already clean)",
                        "container": container_name
                    }
                else:
                    raise e
                    
        except Exception as e:
            logger.error(f"Storage cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cleanup_search_data(self, domain: str) -> Dict[str, Any]:
        """Cleanup Azure Cognitive Search data for domain"""
        try:
            search_service = self.infrastructure.search_service
            if not search_service:
                return {"success": False, "error": "Search service not available"}
            
            index_name = f"{azure_settings.azure_search_index}-{domain}"
            
            # Delete all documents in the domain index
            try:
                delete_result = await search_service.delete_documents_by_query(
                    index_name, f"domain eq '{domain}'"
                )
                
                if delete_result:
                    # Optionally delete the index if no documents remain
                    document_count = await search_service.get_document_count(index_name)
                    if document_count == 0:
                        await search_service.delete_index(index_name)
                        logger.info(f"Search index {index_name} deleted")
                    
                    return {
                        "success": True,
                        "index": index_name,
                        "documents_deleted": delete_result.get("deleted_count", 0),
                        "index_deleted": document_count == 0
                    }
                else:
                    return {"success": False, "error": "Failed to delete search documents"}
                    
            except Exception as e:
                if "IndexNotFound" in str(e):
                    return {
                        "success": True,
                        "message": f"Search index {index_name} does not exist (already clean)",
                        "index": index_name
                    }
                else:
                    raise e
                    
        except Exception as e:
            logger.error(f"Search cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cleanup_cosmos_data(self, domain: str) -> Dict[str, Any]:
        """Cleanup Azure Cosmos DB data for domain"""
        try:
            cosmos_client = self.infrastructure.cosmos_client
            if not cosmos_client:
                return {"success": False, "error": "Cosmos client not available"}
            
            database_name = f"rag-{domain}"
            graph_name = f"knowledge-graph-{domain}"
            
            # Delete all vertices and edges for the domain
            try:
                # Delete all vertices with domain property
                vertices_deleted = await cosmos_client.delete_vertices_by_property(
                    database_name, graph_name, "domain", domain
                )
                
                # Delete all edges related to domain
                edges_deleted = await cosmos_client.delete_edges_by_property(
                    database_name, graph_name, "domain", domain
                )
                
                # Check if graph is empty and delete if so
                vertex_count = await cosmos_client.get_vertex_count(database_name, graph_name)
                if vertex_count == 0:
                    await cosmos_client.delete_graph(database_name, graph_name)
                    logger.info(f"Cosmos graph {graph_name} deleted")
                    
                    # Check if database is empty and delete if so
                    graph_count = await cosmos_client.get_graph_count(database_name)
                    if graph_count == 0:
                        await cosmos_client.delete_database(database_name)
                        logger.info(f"Cosmos database {database_name} deleted")
                
                return {
                    "success": True,
                    "database": database_name,
                    "graph": graph_name,
                    "vertices_deleted": vertices_deleted,
                    "edges_deleted": edges_deleted,
                    "graph_deleted": vertex_count == 0
                }
                
            except Exception as e:
                if "DatabaseNotFound" in str(e) or "GraphNotFound" in str(e):
                    return {
                        "success": True,
                        "message": f"Cosmos structures for {domain} do not exist (already clean)",
                        "database": database_name,
                        "graph": graph_name
                    }
                else:
                    raise e
                    
        except Exception as e:
            logger.error(f"Cosmos cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup_expired_data(self, retention_days: int = 30) -> Dict[str, Any]:
        """Cleanup data older than specified retention period"""
        try:
            cutoff_date = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
            
            cleanup_results = {
                "retention_days": retention_days,
                "cutoff_date": datetime.fromtimestamp(cutoff_date).isoformat(),
                "operations": {}
            }
            
            # Cleanup expired blobs
            storage_client = self.infrastructure.get_rag_storage_client()
            if storage_client:
                expired_blobs = await storage_client.list_blobs_older_than(cutoff_date)
                deleted_count = 0
                for blob in expired_blobs:
                    if await storage_client.delete_blob(blob.container, blob.name):
                        deleted_count += 1
                
                cleanup_results["operations"]["storage"] = {
                    "expired_blobs_found": len(expired_blobs),
                    "expired_blobs_deleted": deleted_count
                }
            
            # Additional cleanup operations can be added here
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Expired data cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def maintenance_health_check(self) -> Dict[str, Any]:
        """Perform maintenance health checks"""
        try:
            health_results = self.infrastructure.check_all_services_health()
            
            # Add maintenance-specific checks
            maintenance_checks = {
                "disk_space": await self._check_disk_space(),
                "memory_usage": await self._check_memory_usage(),
                "connection_pools": await self._check_connection_pools()
            }
            
            health_results["maintenance"] = maintenance_checks
            return health_results
            
        except Exception as e:
            logger.error(f"Maintenance health check failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            
            return {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round((used / total) * 100, 2),
                "status": "healthy" if (free / total) > 0.1 else "warning"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
                "status": "healthy" if memory.percent < 80 else "warning"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_connection_pools(self) -> Dict[str, Any]:
        """Check connection pool status"""
        try:
            pool_status = {}
            
            # Check each service's connection pool
            for service_name in ["openai", "search", "cosmos", "ml"]:
                service = self.infrastructure.get_service(service_name)
                if service and hasattr(service, 'get_connection_pool_status'):
                    pool_status[service_name] = service.get_connection_pool_status()
                else:
                    pool_status[service_name] = {"status": "not_available"}
            
            return pool_status
            
        except Exception as e:
            return {"status": "error", "error": str(e)}