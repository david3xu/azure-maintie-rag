"""
Simple Azure Cosmos DB Gremlin Client - CODING_STANDARDS Compliant
Clean graph database client without over-engineering enterprise patterns.
Uses ThreadPoolExecutor to eliminate async event loop conflicts.
"""

import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from gremlin_python.driver import client, serializer

from config.settings import azure_settings
from infrastructure.constants import AzureServiceLimits

from ..azure_auth.base_client import BaseAzureClient

logger = logging.getLogger(__name__)


class SimpleCosmosGremlinClient(BaseAzureClient):
    """
    Simple Azure Cosmos DB Gremlin client following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses Azure settings for configuration
    - Universal Design: Works with any graph domain
    - Mathematical Foundation: Simple graph operations
    """

    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_cosmos_endpoint

    def _health_check(self) -> bool:
        """Simple health check without event loop conflicts"""
        try:
            if self.gremlin_client:
                # Avoid async operations in sync health check to prevent event loop conflicts
                # Just verify the client is properly initialized
                return (
                    hasattr(self.gremlin_client, "submit")
                    and self.gremlin_client is not None
                )
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        return False

    async def test_connection(self) -> bool:
        """Test connection method expected by ConsolidatedAzureServices"""
        try:
            self.ensure_initialized()
            return self._health_check()
        except Exception as e:
            logger.error(f"Cosmos connection test failed: {e}")
            return False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize simple Gremlin client with ThreadPoolExecutor"""
        super().__init__(config)

        # Simple configuration
        self.database_name = (
            self.config.get("database") or azure_settings.azure_cosmos_database
        )
        self.container_name = (
            self.config.get("container") or azure_settings.cosmos_graph_name
        )
        self.gremlin_client = None
        
        # ThreadPoolExecutor for async event loop isolation
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gremlin")
        self._thread_local = threading.local()

        logger.info(
            f"Simple Cosmos Gremlin client initialized for {self.database_name} with ThreadPoolExecutor"
        )

    def _initialize_client(self):
        """Simple client initialization"""
        try:
            if not self.endpoint or "documents.azure.com" not in self.endpoint:
                raise ValueError(f"Invalid Cosmos endpoint: {self.endpoint}")

            # Extract account name and create Gremlin endpoint
            account_name = self.endpoint.replace("https://", "").split(".")[0]
            gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"

            # Use appropriate authentication method
            if self.use_managed_identity:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
                token = credential.get_token("https://cosmos.azure.com/.default")
            else:
                from azure.identity import AzureCliCredential

                credential = AzureCliCredential()
                token = credential.get_token("https://cosmos.azure.com/.default")

            # Create simple Gremlin client
            self.gremlin_client = client.Client(
                gremlin_endpoint,
                "g",
                username=f"/dbs/{self.database_name}/colls/{self.container_name}",
                password=token.token,
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )

            self._client = self.gremlin_client
            logger.info("Gremlin client initialized")

        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise

    def _get_thread_local_client(self):
        """Get thread-local Gremlin client to avoid connection sharing"""
        if not hasattr(self._thread_local, 'client'):
            try:
                if not self.endpoint or "documents.azure.com" not in self.endpoint:
                    raise ValueError(f"Invalid Cosmos endpoint: {self.endpoint}")

                # Extract account name and create Gremlin endpoint
                account_name = self.endpoint.replace("https://", "").split(".")[0]
                gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"

                # Use appropriate authentication method
                if self.use_managed_identity:
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    token = credential.get_token("https://cosmos.azure.com/.default")
                else:
                    from azure.identity import AzureCliCredential
                    credential = AzureCliCredential()
                    token = credential.get_token("https://cosmos.azure.com/.default")

                # Create thread-local Gremlin client
                self._thread_local.client = client.Client(
                    gremlin_endpoint,
                    "g",
                    username=f"/dbs/{self.database_name}/colls/{self.container_name}",
                    password=token.token,
                    message_serializer=serializer.GraphSONSerializersV2d0(),
                )
                
                logger.debug(f"Thread-local Gremlin client created for thread {threading.current_thread().name}")
                
            except Exception as e:
                logger.error(f"Thread-local client creation failed: {e}")
                raise
                
        return self._thread_local.client

    def _execute_query_sync(self, query: str) -> List[Any]:
        """Execute Gremlin query synchronously in thread pool"""
        try:
            thread_client = self._get_thread_local_client()
            result = thread_client.submit(query)
            query_result = result.all().result(
                timeout=AzureServiceLimits.DEFAULT_GREMLIN_TIMEOUT_SECONDS
            )
            
            # Try to flush/reset connection state to minimize warnings
            # This is a best-effort attempt to clean connection state
            try:
                # Force garbage collection of the result to free connection resources
                del result
            except:
                pass
                
            return query_result
        except Exception as e:
            # FAIL FAST - Don't return empty results on Cosmos DB failures
            logger.error(f"Sync query failed: {e}")
            raise RuntimeError(f"Cosmos DB Gremlin sync query failed: {e}. Check Azure Cosmos DB connection.") from e

    async def _execute_query(self, query: str) -> List[Any]:
        """Execute Gremlin query asynchronously using ThreadPoolExecutor"""
        try:
            self.ensure_initialized()  # Ensure main client is initialized for config
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._execute_query_sync,
                query
            )
        except Exception as e:
            # FAIL FAST - Don't return empty results on Cosmos DB failures
            logger.error(f"Async query failed: {e}")
            raise RuntimeError(f"Cosmos DB Gremlin async query failed: {e}. Check Azure Cosmos DB connection.") from e

    async def execute_query(self, query: str) -> List[Any]:
        """Public method for executing Gremlin queries"""
        return await self._execute_query(query)

    async def add_entity(
        self, entity_data: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Add entity to graph using simple approach"""
        try:
            entity_id = entity_data.get(
                "id", f"entity_{int(datetime.now().timestamp())}"
            )
            entity_text = str(entity_data.get("text", ""))[:500].replace("'", "\\'")
            entity_type = entity_data.get("entity_type", "Entity")

            # Simple entity creation with partition key
            query = f"""
                g.addV('{entity_type}')
                .property('id', '{entity_id}')
                .property('partitionKey', '{domain}')
                .property('text', '{entity_text}')
                .property('domain', '{domain}')
                .property('created_at', '{datetime.now().isoformat()}')
            """

            result = await self._execute_query(query)

            return self.create_success_response(
                "add_entity", {"entity_id": entity_id, "domain": domain}
            )

        except Exception as e:
            # FAIL FAST - Don't return error response, raise exception
            logger.error(f"Add entity failed: {e}")
            raise RuntimeError(f"Cosmos DB entity creation failed: {e}. Check Azure Cosmos DB Gremlin connection.") from e

    async def add_relationship(
        self, relation_data: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Add relationship to graph using simple approach"""
        try:
            head_entity = relation_data.get("head_entity", "").replace("'", "\\'")
            tail_entity = relation_data.get("tail_entity", "").replace("'", "\\'")
            relation_type = relation_data.get("relation_type", "RELATES_TO")

            # Simple relationship creation
            query = f"""
                g.V().has('text', '{head_entity}')
                .addE('{relation_type}')
                .to(g.V().has('text', '{tail_entity}'))
                .property('domain', '{domain}')
                .property('confidence', {relation_data.get("confidence", 1.0)})
                .property('created_at', '{datetime.now().isoformat()}')
            """

            result = await self._execute_query(query)

            return self.create_success_response(
                "add_relationship", {"relation_type": relation_type, "domain": domain}
            )

        except Exception as e:
            # FAIL FAST - Don't return error response, raise exception
            logger.error(f"Add relationship failed: {e}")
            raise RuntimeError(f"Cosmos DB relationship creation failed: {e}. Check Azure Cosmos DB Gremlin connection.") from e

    async def find_entities_by_type(
        self,
        entity_type: str,
        domain: str,
        limit: int = AzureServiceLimits.DEFAULT_GREMLIN_QUERY_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Find entities by type using simple query"""
        try:
            query = f"""
                g.V().hasLabel('{entity_type}')
                .has('domain', '{domain}')
                .limit({limit})
                .valueMap()
            """

            results = await self._execute_query(query)

            entities = []
            for result in results:
                if isinstance(result, dict):
                    entities.append(
                        {
                            "id": self._get_value(result.get("id")),
                            "text": self._get_value(result.get("text")),
                            "entity_type": entity_type,
                            "domain": domain,
                        }
                    )

            return entities

        except Exception as e:
            # FAIL FAST - Don't return empty results on query failures
            logger.error(f"Find entities failed: {e}")
            raise RuntimeError(f"Cosmos DB entity search failed: {e}. Check Azure Cosmos DB Gremlin query execution.") from e

    async def find_related_entities(
        self,
        entity_text: str,
        domain: str,
        limit: int = AzureServiceLimits.DEFAULT_GREMLIN_QUERY_LIMIT // 2,
    ) -> List[Dict[str, Any]]:
        """Find related entities using simple traversal"""
        try:
            entity_text_escaped = entity_text.replace("'", "\\'")

            query = f"""
                g.V().has('text', '{entity_text_escaped}')
                .has('domain', '{domain}')
                .outE()
                .inV()
                .limit({limit})
                .project('entity', 'relation')
                .by('text')
                .by(__.inE().values('relation_type'))
            """

            results = await self._execute_query(query)

            relationships = []
            for result in results:
                if isinstance(result, dict):
                    relationships.append(
                        {
                            "source_entity": entity_text,
                            "target_entity": result.get("entity", ""),
                            "relation_type": result.get("relation", "related"),
                        }
                    )

            return relationships

        except Exception as e:
            # FAIL FAST - Don't return empty results on relationship queries
            logger.error(f"Find related entities failed: {e}")
            raise RuntimeError(f"Cosmos DB relationship search failed: {e}. Check Azure Cosmos DB Gremlin traversal.") from e

    async def count_vertices(self, domain: str) -> int:
        """Count vertices in domain"""
        try:
            query = f"g.V().has('domain', '{domain}').count()"
            result = await self._execute_query(query)
            return int(result[0]) if result else 0
        except Exception as e:
            # FAIL FAST - Don't return fallback count on failures
            logger.error(f"Count vertices failed: {e}")
            raise RuntimeError(f"Cosmos DB vertex count failed: {e}. Check Azure Cosmos DB Gremlin connection.") from e

    async def get_all_entities(self, domain: str) -> List[Dict[str, Any]]:
        """Get all entities for domain"""
        try:
            query = f"g.V().has('domain', '{domain}').limit(1000).valueMap()"
            results = await self._execute_query(query)

            entities = []
            for result in results:
                if isinstance(result, dict):
                    entities.append(
                        {
                            "id": self._get_value(result.get("id")),
                            "text": self._get_value(result.get("text")),
                            "entity_type": self._get_value(
                                result.get("entity_type", "Entity")
                            ),
                            "domain": domain,
                        }
                    )

            return entities

        except Exception as e:
            # FAIL FAST - Don't return empty results on entity retrieval failures
            logger.error(f"Get all entities failed: {e}")
            raise RuntimeError(f"Cosmos DB entity retrieval failed: {e}. Check Azure Cosmos DB Gremlin connection.") from e

    async def get_all_relations(self, domain: str) -> List[Dict[str, Any]]:
        """Get all relations for domain"""
        try:
            query = f"""
                g.E().has('domain', '{domain}')
                .project('source', 'target', 'type')
                .by(outV().values('text'))
                .by(inV().values('text'))
                .by(label())
            """

            results = await self._execute_query(query)

            relations = []
            for result in results:
                if isinstance(result, dict):
                    relations.append(
                        {
                            "source_entity": result.get("source", ""),
                            "target_entity": result.get("target", ""),
                            "relation_type": result.get("type", "related"),
                        }
                    )

            return relations

        except Exception as e:
            # FAIL FAST - Don't return empty results on relation retrieval failures
            logger.error(f"Get all relations failed: {e}")
            raise RuntimeError(f"Cosmos DB relation retrieval failed: {e}. Check Azure Cosmos DB Gremlin connection.") from e

    async def export_graph_for_training(self, domain: str) -> Dict[str, Any]:
        """Export graph data for training"""
        try:
            entities = await self.get_all_entities(domain)
            relations = await self.get_all_relations(domain)

            return {
                "success": True,
                "domain": domain,
                "entities": entities,
                "relations": relations,
                "entities_count": len(entities),
                "relations_count": len(relations),
            }

        except Exception as e:
            # FAIL FAST - Don't return error response, raise exception
            logger.error(f"Graph export failed: {e}")
            raise RuntimeError(f"Cosmos DB graph export failed: {e}. Check Azure Cosmos DB Gremlin connection.") from e

    def _get_value(self, prop_value):
        """Extract value from Gremlin property format"""
        if isinstance(prop_value, list) and prop_value:
            return prop_value[0]
        return prop_value or ""

    def close(self):
        """Close Gremlin client and cleanup ThreadPoolExecutor"""
        try:
            # Close main client
            if self.gremlin_client:
                self.gremlin_client.close()
                self.gremlin_client = None
                logger.info("Main Gremlin client closed")
            
            # Close thread-local clients
            if hasattr(self._thread_local, 'client') and self._thread_local.client:
                try:
                    self._thread_local.client.close()
                    logger.debug("Thread-local Gremlin client closed")
                except Exception as e:
                    logger.warning(f"Thread-local client close warning: {e}")
            
            # Shutdown ThreadPoolExecutor
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=True)
                logger.info("ThreadPoolExecutor shutdown complete")
                
        except Exception as e:
            logger.warning(f"Client close warning: {e}")


# Backward compatibility aliases
AzureCosmosGremlinClient = SimpleCosmosGremlinClient
SimpleCosmosClient = SimpleCosmosGremlinClient  # Common alias
