"""Azure Cosmos DB Gremlin client for Universal RAG knowledge graphs."""

import logging
from typing import Dict, List, Any, Optional
import json
import time
from gremlin_python.driver import client, serializer
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T, P

from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureCosmosGremlinClient:
    """Universal Azure Cosmos DB Gremlin client for knowledge graphs - native graph operations"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure Cosmos DB Gremlin client"""
        self.config = config or {}

        # Load from environment (matches azure_openai.py pattern)
        self.endpoint = self.config.get('endpoint') or azure_settings.azure_cosmos_endpoint
        self.key = self.config.get('key') or azure_settings.azure_cosmos_key
        self.database_name = self.config.get('database') or azure_settings.azure_cosmos_database
        self.container_name = self.config.get('container') or azure_settings.azure_cosmos_container

        if not self.endpoint or not self.key:
            raise ValueError("Azure Cosmos DB endpoint and key are required")

        # Initialize Gremlin client lazily to avoid async event loop issues
        self.gremlin_client = None
        self._client_initialized = False

        logger.info(f"AzureCosmosGremlinClient initialized for database: {self.database_name}")

    def _initialize_client(self):
        """Lazy initialize the Gremlin client to avoid async event loop issues"""
        if self._client_initialized:
            return

        try:
            # Convert endpoint to Gremlin endpoint
            gremlin_endpoint = self.endpoint.replace('https://', 'wss://').replace('http://', 'ws://')
            gremlin_endpoint = gremlin_endpoint.replace(':443/', ':443/gremlin/')

            self.gremlin_client = client.Client(
                gremlin_endpoint,
                'g',
                username=f"/dbs/{self.database_name}/colls/{self.container_name}",
                password=self.key,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            self._client_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Azure Cosmos Gremlin client: {e}")
            raise

    def _test_connection(self):
        """Test Gremlin connection"""
        try:
            # Simple query to test connection
            result = self.gremlin_client.submit("g.V().limit(1)")
            result.all().result()  # This will raise an exception if connection fails
            logger.info("Gremlin connection test successful")
        except Exception as e:
            logger.error(f"Gremlin connection test failed: {e}")
            raise

    def add_entity(self, entity_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Add entity vertex to graph using Gremlin"""
        try:
            # Lazy initialize client if needed
            if not self._client_initialized:
                self._initialize_client()

            # Suppress async warnings during Gremlin operations
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                # Gremlin query to add vertex
                query = f"""
                    g.addV('Entity')
                        .property('text', '{entity_data.get("text", "")}')
                        .property('entity_type', '{entity_data.get("entity_type", "unknown")}')
                        .property('domain', '{domain}')
                        .property('confidence', {entity_data.get("confidence", 1.0)})
                        .property('created_at', '{entity_data.get("created_at", "")}')
                """

                result = self.gremlin_client.submit(query)
                vertex = result.all().result()[0]

            return {
                "success": True,
                "id": str(vertex.id),
                "entity_type": entity_data.get("entity_type", "unknown")
            }

        except Exception as e:
            logger.error(f"Entity addition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "entity_id": entity_data.get("id", "unknown")
            }

    def add_relationship(self, relation_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Add relationship edge to graph using Gremlin"""
        try:
            # Gremlin query to add edge
            query = f"""
                g.V().has('text', '{relation_data.get("head_entity", "")}')
                    .addE('RELATES_TO')
                    .to(g.V().has('text', '{relation_data.get("tail_entity", "")}'))
                    .property('relation_type', '{relation_data.get("relation_type", "unknown")}')
                    .property('domain', '{domain}')
                    .property('confidence', {relation_data.get("confidence", 1.0)})
                    .property('created_at', '{relation_data.get("created_at", "")}')
            """

            result = self.gremlin_client.submit(query)
            edge = result.all().result()[0]

            return {
                "success": True,
                "id": str(edge.id),
                "relation_type": relation_data.get("relation_type", "unknown")
            }

        except Exception as e:
            logger.error(f"Relationship addition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "relation_id": relation_data.get("id", "unknown")
            }

    def find_entities_by_type(self, entity_type: str, domain: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Find entities by type using Gremlin traversal"""
        try:
            query = f"""
                g.V().has('type', 'Entity')
                    .has('entity_type', '{entity_type}')
                    .has('domain', '{domain}')
                    .limit({limit})
                    .valueMap()
            """

            result = self.gremlin_client.submit(query)
            entities = result.all().result()

            return [
                {
                    "id": str(entity.id),
                    "text": entity.get("text", [""])[0],
                    "entity_type": entity.get("entity_type", [""])[0],
                    "domain": entity.get("domain", [""])[0],
                    "confidence": entity.get("confidence", [1.0])[0]
                }
                for entity in entities
            ]

        except Exception as e:
            logger.error(f"Entity query failed: {e}")
            return []

    def find_related_entities(self, entity_text: str, domain: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find related entities using Gremlin path traversal"""
        try:
            query = f"""
                g.V().has('text', '{entity_text}')
                    .has('domain', '{domain}')
                    .outE()
                    .inV()
                    .limit({limit})
                    .project('entity', 'relation')
                    .by('text')
                    .by(__.inE().values('relation_type'))
            """

            result = self.gremlin_client.submit(query)
            relationships = result.all().result()

            return [
                {
                    "source_entity": entity_text,
                    "target_entity": rel["entity"],
                    "relation_type": rel["relation"]
                }
                for rel in relationships
            ]

        except Exception as e:
            logger.error(f"Relationship query failed: {e}")
            return []

    def find_entity_paths(self, start_entity: str, end_entity: str, domain: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Find paths between entities using Gremlin path finding"""
        try:
            query = f"""
                g.V().has('text', '{start_entity}')
                    .has('domain', '{domain}')
                    .repeat(outE().inV().simplePath())
                    .times({max_hops})
                    .until(has('text', '{end_entity}'))
                    .path()
                    .by('text')
                    .by('relation_type')
            """

            result = self.gremlin_client.submit(query)
            paths = result.all().result()

            return [
                {
                    "start_entity": start_entity,
                    "end_entity": end_entity,
                    "path": path,
                    "hops": len(path) // 2  # Alternating vertices and edges
                }
                for path in paths
            ]

        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return []

    def _execute_gremlin_query_safe(self, query: str, timeout_seconds: int = 30):
        """Enterprise thread-isolated Gremlin query execution"""
        import concurrent.futures
        import threading
        import warnings
        def _run_gremlin_query():
            """Execute Gremlin query in isolated thread context"""
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", DeprecationWarning)
                    result = self.gremlin_client.submit(query)
                    return result.all().result()
            except Exception as e:
                logger.warning(f"Gremlin query execution failed: {e}")
                return []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_gremlin_query)
                return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Gremlin query timed out after {timeout_seconds}s: {query}")
            return []
        except Exception as e:
            logger.warning(f"Thread execution failed for Gremlin query: {e}")
            return []

    def get_graph_statistics(self, domain: str) -> Dict[str, Any]:
        """Get knowledge graph statistics using enterprise thread-safe pattern"""
        try:
            if not self._client_initialized:
                self._initialize_client()
            vertex_query = f"g.V().has('domain', '{domain}').count()"
            vertex_result = self._execute_gremlin_query_safe(vertex_query)
            vertex_count = vertex_result[0] if vertex_result else 0
            edge_query = f"g.E().has('domain', '{domain}').count()"
            edge_result = self._execute_gremlin_query_safe(edge_query)
            edge_count = edge_result[0] if edge_result else 0
            return {
                "success": True,
                "domain": domain,
                "vertex_count": vertex_count,
                "edge_count": edge_count,
                "total_elements": vertex_count + edge_count
            }
        except Exception as e:
            logger.warning(f"Statistics query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain,
                "vertex_count": 0,
                "edge_count": 0,
                "total_elements": 0
            }

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status for service validation"""
        try:
            # Check basic configuration first
            if not self.endpoint:
                return {
                    "status": "unhealthy",
                    "error": "Cosmos DB endpoint not configured",
                    "service": "cosmos"
                }

            if not self.key:
                return {
                    "status": "unhealthy",
                    "error": "Cosmos DB key not configured",
                    "service": "cosmos"
                }

            # Test client initialization
            if not self._client_initialized:
                try:
                    self._initialize_client()
                except Exception as init_error:
                    return {
                        "status": "unhealthy",
                        "error": f"Client initialization failed: {init_error}",
                        "service": "cosmos"
                    }

            return {
                "status": "healthy",
                "service": "cosmos",
                "endpoint": self.endpoint,
                "database": self.database_name,
                "container": self.container_name
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "cosmos"
            }

    def close(self):
        """Enterprise-safe Gremlin client cleanup"""
        try:
            if self.gremlin_client and self._client_initialized:
                import concurrent.futures
                import warnings
                def _safe_close():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        warnings.simplefilter("ignore", DeprecationWarning)
                        self.gremlin_client.close()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_safe_close)
                    future.result(timeout=10)
            logger.info("Azure Cosmos Gremlin client closed successfully")
        except Exception as e:
            logger.warning(f"Gremlin client cleanup warning: {e}")
        finally:
            self._client_initialized = False
            self.gremlin_client = None

    def get_all_entities(self, domain: str) -> List[Dict[str, Any]]:
        """Get all entities using enterprise thread-safe pattern"""
        try:
            if not self._client_initialized:
                self._initialize_client()
            query = f"""
                g.V().has('domain', '{domain}')
                    .valueMap()
            """
            entities = self._execute_gremlin_query_safe(query)
            return [
                {
                    "id": str(entity.id),
                    "text": entity.get("text", [""])[0],
                    "entity_type": entity.get("entity_type", [""])[0],
                    "domain": entity.get("domain", [""])[0],
                    "confidence": entity.get("confidence", [1.0])[0]
                }
                for entity in entities
            ]
        except Exception as e:
            logger.warning(f"Get all entities failed: {e}")
            return []

    def get_all_relations(self, domain: str) -> List[Dict[str, Any]]:
        """Get all relations using enterprise thread-safe pattern"""
        try:
            if not self._client_initialized:
                self._initialize_client()
            query = f"""
                g.E().has('domain', '{domain}')
                    .project('source', 'target', 'relation_type')
                    .by(__.outV().values('text'))
                    .by(__.inV().values('text'))
                    .by('relation_type')
            """
            relations = self._execute_gremlin_query_safe(query)
            return [
                {
                    "source_entity": rel["source"],
                    "target_entity": rel["target"],
                    "relation_type": rel["relation_type"]
                }
                for rel in relations
            ]
        except Exception as e:
            logger.warning(f"Get all relations failed: {e}")
            return []