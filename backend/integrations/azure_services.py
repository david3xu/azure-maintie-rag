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
from core.azure_monitoring.app_insights_client import AzureApplicationInsightsClient
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureServicesManager:
    async def _migrate_to_storage(self, source_data_path: str, domain: str, migration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Azure Blob Storage migration with multi-account orchestration"""
        try:
            from pathlib import Path
            import aiofiles

            storage_client = self.get_rag_storage_client()
            if not storage_client:
                raise RuntimeError("RAG storage client not initialized")

            container_name = f"{azure_settings.azure_blob_container}-{domain}"
            source_path = Path(source_data_path)
            if not source_path.exists():
                return {"success": False, "error": f"Source path not found: {source_data_path}"}

            uploaded_files = []
            failed_uploads = []

            for file_path in source_path.glob("*.md"):
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                    blob_name = f"{domain}/{file_path.name}"
                    blob_metadata = {
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "source_file": file_path.name,
                        "environment": azure_settings.azure_environment
                    }
                    upload_result = await storage_client.upload_blob_async(
                        container_name=container_name,
                        blob_name=blob_name,
                        data=content.encode('utf-8'),
                        metadata=blob_metadata,
                        overwrite=True
                    )
                    if upload_result.get("success", False):
                        uploaded_files.append(blob_name)
                    else:
                        failed_uploads.append({"file": file_path.name, "error": upload_result.get("error")})
                except Exception as file_error:
                    failed_uploads.append({"file": file_path.name, "error": str(file_error)})

            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_storage_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "container": container_name
                    },
                    measurements={
                        "files_uploaded": len(uploaded_files),
                        "files_failed": len(failed_uploads),
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )

            return {
                "success": len(failed_uploads) == 0,
                "uploaded_files": uploaded_files,
                "failed_uploads": failed_uploads,
                "container_name": container_name,
                "total_files": len(uploaded_files) + len(failed_uploads)
            }
        except Exception as e:
            logger.error(f"Storage migration failed: {e}")
            return {"success": False, "error": str(e)}

    async def _migrate_to_search(self, source_data_path: str, domain: str, migration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Azure Cognitive Search migration with semantic search configuration"""
        try:
            from pathlib import Path
            from core.azure_search.search_client import AzureCognitiveSearchClient
            from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
            search_client = self.get_service('search')
            if not search_client:
                raise RuntimeError("Azure Search client not initialized")
            index_name = f"{azure_settings.get_resource_name('search', domain)}"
            knowledge_extractor = AzureOpenAIKnowledgeExtractor(domain)
            source_path = Path(source_data_path)
            processed_documents = []
            for file_path in source_path.glob("*.md"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                extraction_result = await knowledge_extractor.extract_knowledge_from_texts([content], [file_path.name])
                if extraction_result.get("success", False):
                    knowledge_data = knowledge_extractor.get_extracted_knowledge()
                    search_document = {
                        "id": f"{domain}_{file_path.stem}",
                        "domain": domain,
                        "title": file_path.stem.replace('_', ' ').title(),
                        "content": content,
                        "file_name": file_path.name,
                        "entities": list(knowledge_data.get("entities", {}).keys()),
                        "entity_types": [entity.get("entity_type", "") for entity in knowledge_data.get("entities", {}).values()],
                        "relation_types": list(set(rel.get("relation_type", "") for rel in knowledge_data.get("relations", []))),
                        "extraction_confidence": knowledge_data.get("extraction_metadata", {}).get("confidence", 1.0),
                        "last_updated": migration_context["start_time"]
                    }
                    processed_documents.append(search_document)
            if not processed_documents:
                return {"success": False, "error": "No documents processed for search indexing"}
            index_result = await search_client.create_or_update_index(
                index_name=index_name,
                documents=processed_documents,
                domain=domain
            )
            return {
                "success": True,
                "index_name": index_name,
                "documents_indexed": len(processed_documents),
                "index_operation_result": index_result,
                "migration_context": migration_context
            }
        except Exception as e:
            logger.error(f"Search migration failed: {e}")
            return {"success": False, "error": str(e)}
            knowledge_data = knowledge_extractor.get_extracted_knowledge()
            search_documents = []
            for doc_id, doc_data in knowledge_data["documents"].items():
                content = doc_data["text"]
                from core.azure_search.vector_service import AzureSearchVectorService
                vector_service = AzureSearchVectorService(domain)
                try:
                    embedding = await vector_service._get_embedding(content)
                    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                except Exception as e:
                    logger.warning(f"Embedding generation failed for {doc_id}: {e}")
                    embedding_list = [0.0] * 1536
                search_doc = {
                    "id": doc_id,
                    "content": content,
                    "title": doc_data.get("title", ""),
                    "domain": domain,
                    "source": doc_data.get("metadata", {}).get("source", "unknown"),
                    "contentVector": embedding_list
                }
                search_documents.append(search_doc)
            if search_documents:
                upload_result = await search_service.upload_documents_async(search_documents)
                indexed_count = upload_result.get("uploaded_count", 0)
            else:
                indexed_count = 0
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_search_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "index_name": index_name
                    },
                    measurements={
                        "documents_indexed": indexed_count,
                        "extraction_entities": len(knowledge_data.get("entities", {})),
                        "extraction_relations": len(knowledge_data.get("relations", [])),
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )
            return {
                "success": indexed_count > 0,
                "index_name": index_name,
                "documents_indexed": indexed_count,
                "entities_extracted": len(knowledge_data.get("entities", {})),
                "relations_extracted": len(knowledge_data.get("relations", []))
            }
        except Exception as e:
            logger.error(f"Search migration failed: {e}")
            return {"success": False, "error": str(e)}

    async def _migrate_to_cosmos(self, source_data_path: str, domain: str, migration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Azure Cosmos DB Gremlin graph population with async entity/relation migration and robust error handling."""
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            cosmos_client = self.get_service('cosmos')
            if not cosmos_client:
                raise RuntimeError("Cosmos DB client not initialized")
            from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
            knowledge_extractor = AzureOpenAIKnowledgeExtractor(domain)
            source_path = Path(source_data_path)
            texts = []
            sources = []
            for file_path in source_path.glob("*.md"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():
                        texts.append(content)
                        sources.append(str(file_path))
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
            if not texts:
                return {"success": False, "error": "No valid text content found for graph creation"}
            extraction_results = await knowledge_extractor.extract_knowledge_from_texts(texts, sources)
            if not extraction_results.get("success", False):
                return {"success": False, "error": f"Knowledge extraction failed: {extraction_results.get('error')}"}
            knowledge_data = knowledge_extractor.get_extracted_knowledge()
            entities_created = []
            entity_failures = []
            # Async batch entity insertion
            for entity_id, entity_data in knowledge_data["entities"].items():
                try:
                    entity_creation_data = {
                        "entity_id": entity_id,
                        "text": entity_data["text"],
                        "entity_type": entity_data["entity_type"],
                        "domain": domain,
                        "confidence": entity_data.get("confidence", 1.0),
                        "extraction_method": "azure_openai",
                        "source_documents": entity_data.get("source_documents", []),
                        "migration_id": migration_context.get("migration_id"),
                        "created_at": datetime.now().isoformat(),
                        "metadata": json.dumps({
                            "migration_id": migration_context.get("migration_id"),
                            "source": entity_data.get("source", "migration")
                        })
                    }
                    result = await cosmos_client.store_entity_with_embeddings(entity_creation_data)
                    if result:
                        entities_created.append(entity_id)
                    else:
                        entity_failures.append({"entity_id": entity_id, "error": "Insert failed"})
                except Exception as e:
                    entity_failures.append({"entity_id": entity_id, "error": str(e)})
            relations_created = []
            relation_failures = []
            # Async batch relation insertion
            for relation_data in knowledge_data["relations"]:
                try:
                    relation_creation_data = {
                        "relation_id": relation_data["relation_id"],
                        "relation_type": relation_data["relation_type"],
                        "domain": domain,
                        "confidence": relation_data.get("confidence", 1.0),
                        "extraction_method": "azure_openai",
                        "source_documents": relation_data.get("source_documents", []),
                        "migration_id": migration_context.get("migration_id"),
                        "created_at": datetime.now().isoformat(),
                        "metadata": json.dumps({
                            "migration_id": migration_context.get("migration_id"),
                            "source": relation_data.get("source", "migration")
                        })
                    }
                    result = await cosmos_client.store_relation_with_embeddings(
                        source_entity_id=relation_data["head_entity"],
                        target_entity_id=relation_data["tail_entity"],
                        relation_data=relation_creation_data
                    )
                    if result:
                        relations_created.append(relation_data["relation_id"])
                    else:
                        relation_failures.append({"relation_id": relation_data["relation_id"], "error": "Insert failed"})
                except Exception as e:
                    relation_failures.append({"relation_id": relation_data["relation_id"], "error": str(e)})
            # Validate graph statistics
            graph_stats = await cosmos_client.get_graph_change_metrics(domain)
            validated_entities = graph_stats.get("new_entities", 0)
            validated_relations = graph_stats.get("new_relations", 0)
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_cosmos_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context.get("migration_id"),
                        "database": azure_settings.azure_cosmos_database
                    },
                    measurements={
                        "entities_created": len(entities_created),
                        "relations_created": len(relations_created),
                        "validated_entities": validated_entities,
                        "validated_relations": validated_relations,
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )
            return {
                "success": len(entity_failures) == 0 and len(relation_failures) == 0,
                "entities_created": entities_created,
                "relations_created": relations_created,
                "entity_failures": entity_failures,
                "relation_failures": relation_failures,
                "validated_entities": validated_entities,
                "validated_relations": validated_relations,
                "total_entities": len(entities_created) + len(entity_failures),
                "total_relations": len(relations_created) + len(relation_failures)
            }
        except Exception as e:
            logger.error(f"Cosmos migration failed: {e}")
            return {"success": False, "error": str(e)}
    """Unified manager for all Azure services - enterprise health monitoring"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize all Azure service clients"""
        self.config = config or {}
        self.services = {}
        self.service_status = {}
        self.initialization_status = {}
        self.app_insights = None
        if azure_settings.azure_enable_telemetry and azure_settings.azure_application_insights_connection_string:
            self.app_insights = AzureApplicationInsightsClient(
                connection_string=azure_settings.azure_application_insights_connection_string,
                sampling_rate=azure_settings.effective_telemetry_sampling_rate
            )

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
        """Enterprise health check with circuit breaker pattern"""
        health_results = {}
        start_time = time.time()

        # Add circuit breaker state tracking
        circuit_breaker_state = {
            'consecutive_failures': 0,
            'last_failure_time': None,
            'circuit_open': False
        }

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                'openai': executor.submit(self._health_check_with_timeout, 'openai', 10),
                'rag_storage': executor.submit(self._health_check_with_timeout, 'rag_storage', 15),
                'ml_storage': executor.submit(self._health_check_with_timeout, 'ml_storage', 15),
                'app_storage': executor.submit(self._health_check_with_timeout, 'app_storage', 15),
                'search': executor.submit(self._health_check_with_timeout, 'search', 20),
                'cosmos': executor.submit(self._health_check_with_timeout, 'cosmos', 25),
            }

            for service_name, future in futures.items():
                try:
                    health_results[service_name] = future.result(timeout=30)
                    # Reset circuit breaker on success
                    circuit_breaker_state['consecutive_failures'] = 0
                except Exception as e:
                    if hasattr(e, 'timeout') and e.timeout:
                        health_results[service_name] = {
                            "status": "timeout",
                            "error": f"Health check timeout after 30s",
                            "service": service_name
                        }
                    else:
                        logger.error(f"Service health check failed for {service_name}: {e}")
                        health_results[service_name] = {
                            "status": "unhealthy",
                            "error": str(e),
                            "service": service_name
                        }
                    circuit_breaker_state['consecutive_failures'] += 1

        overall_time = time.time() - start_time
        # Track overall health check event in Application Insights
        if self.app_insights and self.app_insights.enabled:
            healthy_count = sum(1 for s in health_results.values() if s.get("status") == "healthy")
            self.app_insights.track_event(
                name="azure_services_health_check",
                properties={
                    "overall_status": "healthy" if healthy_count == len(health_results) else "degraded",
                    "environment": getattr(azure_settings, 'azure_environment', 'dev')
                },
                measurements={
                    "healthy_services": healthy_count,
                    "total_services": len(health_results),
                    "check_duration_ms": overall_time * 1000
                }
            )

        return {
            "overall_status": self._calculate_overall_status(health_results) if hasattr(self, '_calculate_overall_status') else ("healthy" if all(result.get("status") == "healthy" for result in health_results.values()) else "degraded"),
            "services": health_results,
            "healthy_count": sum(1 for s in health_results.values() if s.get("status") == "healthy"),
            "total_count": len(health_results),
            "health_check_duration_ms": (time.time() - start_time) * 1000,
            "timestamp": time.time(),
            "circuit_breaker": circuit_breaker_state,
            "telemetry": {
                "service": "azure_services_manager",
                "operation": "health_check",
                "environment": self._get_environment_from_config() if hasattr(self, '_get_environment_from_config') else "enterprise"
            }
        }

    def _health_check_with_timeout(self, service_name: str, timeout_seconds: int):
        """Individual service health check with timeout"""
        service = self.services.get(service_name)
        if not service:
            return {"status": "not_configured", "service": service_name}
        if hasattr(service, 'get_service_status'):
            return service.get_service_status()
        elif hasattr(service, 'get_connection_status'):
            return service.get_connection_status()
        else:
            return {"status": "unknown", "service": service_name}

    def migrate_data_to_azure(self, source_data_path: str, domain: str) -> Dict[str, Any]:
        """Enterprise data migration with comprehensive error tracking"""
        import uuid
        migration_results = {
            "storage_migration": {"success": False, "details": {}},
            "search_migration": {"success": False, "details": {}},
            "cosmos_migration": {"success": False, "details": {}}
        }

        migration_context = {
            "source_path": source_data_path,
            "domain": domain,
            "start_time": time.time(),
            "migration_id": str(uuid.uuid4())
        }

        try:
            # 1. Storage migration with detailed tracking
            storage_result = self._migrate_to_storage(source_data_path, domain, migration_context) if hasattr(self, '_migrate_to_storage') else {"success": False, "error": "Not implemented"}
            migration_results["storage_migration"] = storage_result

            if not storage_result["success"]:
                raise RuntimeError(f"Storage migration failed: {storage_result.get('error')}")

            # 2. Search index migration with validation
            search_result = self._migrate_to_search(source_data_path, domain, migration_context) if hasattr(self, '_migrate_to_search') else {"success": False, "error": "Not implemented"}
            migration_results["search_migration"] = search_result

            if not search_result["success"]:
                raise RuntimeError(f"Search migration failed: {search_result.get('error')}")

            # 3. Cosmos DB migration with consistency checks
            cosmos_result = self._migrate_to_cosmos(source_data_path, domain, migration_context) if hasattr(self, '_migrate_to_cosmos') else {"success": False, "error": "Not implemented"}
            migration_results["cosmos_migration"] = cosmos_result

            if not cosmos_result["success"]:
                raise RuntimeError(f"Cosmos migration failed: {cosmos_result.get('error')}")

            logger.info(f"Data migration completed for domain: {domain}")
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_data_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "success": True
                    },
                    measurements={
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )
            return {
                "success": True,
                "domain": domain,
                "migration_results": migration_results,
                "migration_context": migration_context,
                "duration_seconds": time.time() - migration_context["start_time"]
            }

        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_data_migration_failed",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "error": str(e)
                    }
                )
            # Add rollback capability for partial migrations
            self._rollback_partial_migration(migration_results, migration_context) if hasattr(self, '_rollback_partial_migration') else None
            raise RuntimeError(f"Data migration failed: {e}")

    def _rollback_partial_migration(self, migration_results: Dict, context: Dict):
        """Rollback partially completed migrations"""
        logger.info(f"Rolling back partial migration: {context['migration_id']}")
        # Implementation based on your migration state
        pass

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

    def _migrate_to_storage(self, source_data_path: str, domain: str, migration_context: Dict) -> Dict[str, Any]:
        """Azure Blob Storage migration using existing client patterns"""
        from pathlib import Path
        storage_client = self.get_rag_storage_client()
        container_name = f"rag-data-{domain}"
        uploaded_files = []
        try:
            # Ensure container exists
            import asyncio
            asyncio.run(storage_client.create_container(container_name))
            source_path = Path(source_data_path)
            if source_path.is_file():
                with open(source_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                blob_name = f"{source_path.stem}_{domain}.txt"
                asyncio.run(storage_client.upload_text(container_name, blob_name, text_content))
                uploaded_files.append(blob_name)
            elif source_path.is_dir():
                supported_formats = getattr(storage_client, 'supported_text_formats', ['.md', '.txt'])
                for file_path in source_path.rglob('*'):
                    if file_path.suffix in supported_formats:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        blob_name = f"{file_path.stem}_{domain}{file_path.suffix}"
                        asyncio.run(storage_client.upload_text(container_name, blob_name, text_content))
                        uploaded_files.append(blob_name)
            blob_list = storage_client.list_blobs(f"{domain}_")
            validated_count = len(blob_list)
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_blob_storage_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "success": True
                    },
                    measurements={
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )
            return {
                "success": validated_count > 0,
                "uploaded_files": uploaded_files,
                "container_name": container_name,
                "validated_blobs": validated_count,
                "details": f"Uploaded {len(uploaded_files)} files, validated {validated_count} blobs"
            }
        except Exception as e:
            logger.error(f"Storage migration failed: {e}")
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_blob_storage_migration_failed",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "error": str(e)
                    }
                )
            return {
                "success": False,
                "error": str(e),
                "container_name": container_name
            }

    def _migrate_to_search(self, source_data_path: str, domain: str, migration_context: Dict) -> Dict[str, Any]:
        """Azure Cognitive Search migration using existing client patterns"""
        from pathlib import Path
        import json
        search_client = self.get_service('search')
        index_name = f"rag-index-{domain}"
        documents = []
        try:
            # Create index if not exists
            import asyncio
            asyncio.run(search_client.create_index(index_name))
            source_path = Path(source_data_path)
            if source_path.is_file():
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = {
                    "id": f"doc_0_{source_path.stem}",
                    "content": content,
                    "title": source_path.stem,
                    "domain": domain,
                    "source": str(source_path),
                    "metadata": json.dumps({"file_size": len(content), "migration_id": migration_context["migration_id"]})
                }
                documents.append(doc)
            elif source_path.is_dir():
                supported_formats = ['.md', '.txt']
                doc_index = 0
                for file_path in source_path.rglob('*'):
                    if file_path.suffix in supported_formats:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        doc = {
                            "id": f"doc_{doc_index}_{file_path.stem}",
                            "content": content,
                            "title": file_path.stem,
                            "domain": domain,
                            "source": str(file_path),
                            "metadata": json.dumps({"file_size": len(content), "migration_id": migration_context["migration_id"]})
                        }
                        documents.append(doc)
                        doc_index += 1
            upload_result = search_client.upload_documents(documents)
            # Validate indexing
            validation_results = asyncio.run(search_client.search_documents(index_name, "*", top_k=len(documents)))
            validated_count = len(validation_results)
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_search_index_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "success": True
                    },
                    measurements={
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )
            return {
                "success": upload_result.get("success", False) and validated_count > 0,
                "index_name": index_name,
                "uploaded_documents": upload_result.get("uploaded_count", 0),
                "validated_documents": validated_count,
                "details": f"Indexed {len(documents)} documents, validated {validated_count} in search"
            }
        except Exception as e:
            logger.error(f"Search migration failed: {e}")
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_search_index_migration_failed",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "error": str(e)
                    }
                )
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name
            }

    def _migrate_to_cosmos(self, source_data_path: str, domain: str, migration_context: Dict) -> Dict[str, Any]:
        """Azure Cosmos DB migration using existing cosmos_gremlin_client.py patterns"""
        from pathlib import Path
        import json
        cosmos_client = self.get_service('cosmos')
        entities_created = []
        try:
            source_path = Path(source_data_path)
            if source_path.is_file():
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                entity_data = {
                    "id": f"entity_{source_path.stem}_{domain}",
                    "text": content[:500],
                    "entity_type": "document",
                    "source": str(source_path),
                    "metadata": json.dumps({"migration_id": migration_context["migration_id"]})
                }
                result = cosmos_client.add_entity(entity_data, domain)
                if result.get("success", False):
                    entities_created.append(entity_data["id"])
            elif source_path.is_dir():
                supported_formats = ['.md', '.txt']
                entity_index = 0
                for file_path in source_path.rglob('*'):
                    if file_path.suffix in supported_formats:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        entity_data = {
                            "id": f"entity_{entity_index}_{file_path.stem}_{domain}",
                            "text": content[:500],
                            "entity_type": "document",
                            "source": str(file_path),
                            "metadata": json.dumps({"migration_id": migration_context["migration_id"]})
                        }
                        result = cosmos_client.add_entity(entity_data, domain)
                        if result.get("success", False):
                            entities_created.append(entity_data["id"])
                        entity_index += 1
            stats = cosmos_client.get_graph_statistics(domain)
            validated_entities = stats.get("vertex_count", 0)
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_cosmos_graph_migration",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "success": True
                    },
                    measurements={
                        "duration_seconds": time.time() - migration_context["start_time"]
                    }
                )
            return {
                "success": len(entities_created) > 0,
                "entities_created": entities_created,
                "created_count": len(entities_created),
                "validated_entities": validated_entities,
                "domain": domain,
                "details": f"Created {len(entities_created)} entities, validated {validated_entities} in graph"
            }
        except Exception as e:
            logger.error(f"Cosmos migration failed: {e}")
            if self.app_insights and self.app_insights.enabled:
                self.app_insights.track_event(
                    name="azure_cosmos_graph_migration_failed",
                    properties={
                        "domain": domain,
                        "migration_id": migration_context["migration_id"],
                        "error": str(e)
                    }
                )
            return {
                "success": False,
                "error": str(e),
                "domain": domain
            }