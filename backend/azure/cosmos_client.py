"""Azure Cosmos DB client for Universal RAG knowledge graphs."""

import logging
from typing import Dict, List, Any, Optional
import json
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError

from backend.config.azure_settings import azure_settings

logger = logging.getLogger(__name__)


class AzureCosmosClient:
    """Universal Azure Cosmos DB client for knowledge graphs - follows azure_openai.py pattern"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure Cosmos DB client"""
        self.config = config or {}

        # Load from environment (matches azure_openai.py pattern)
        self.endpoint = self.config.get('endpoint') or azure_settings.azure_cosmos_endpoint
        self.key = self.config.get('key') or azure_settings.azure_cosmos_key
        self.database_name = self.config.get('database') or azure_settings.azure_cosmos_database
        self.container_name = self.config.get('container') or azure_settings.azure_cosmos_container

        if not self.endpoint or not self.key:
            raise ValueError("Azure Cosmos DB endpoint and key are required")

        # Initialize client (follows azure_openai.py error handling pattern)
        try:
            self.cosmos_client = CosmosClient(self.endpoint, self.key)
            self._ensure_database_exists()
            self._ensure_container_exists()
        except Exception as e:
            logger.error(f"Failed to initialize Azure Cosmos client: {e}")
            raise

        logger.info(f"AzureCosmosClient initialized for database: {self.database_name}")

    def _ensure_database_exists(self):
        """Ensure database exists - create if needed"""
        try:
            self.database = self.cosmos_client.create_database_if_not_exists(
                id=self.database_name
            )
        except CosmosHttpResponseError as e:
            logger.error(f"Database operation failed: {e}")
            raise

    def _ensure_container_exists(self):
        """Ensure container exists - create if needed"""
        try:
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key="/domain",  # Partition by domain for universal support
                offer_throughput=400
            )
        except CosmosHttpResponseError as e:
            logger.error(f"Container operation failed: {e}")
            raise

    def store_entity(self, entity_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Store entity in Cosmos DB - universal format"""
        try:
            # Universal entity format
            item = {
                "id": entity_data.get("id", f"entity_{entity_data.get('text', 'unknown')}"),
                "type": "entity",
                "text": entity_data.get("text", ""),
                "entity_type": entity_data.get("entity_type", "unknown"),
                "domain": domain,
                "confidence": entity_data.get("confidence", 1.0),
                "metadata": entity_data.get("metadata", {}),
                "created_at": entity_data.get("created_at", "")
            }

            result = self.container.create_item(body=item)

            return {
                "success": True,
                "id": result["id"],
                "entity_type": result["entity_type"]
            }

        except Exception as e:
            logger.error(f"Entity storage failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "entity_id": entity_data.get("id", "unknown")
            }

    def store_relation(self, relation_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Store relation in Cosmos DB - universal format"""
        try:
            # Universal relation format
            item = {
                "id": relation_data.get("id", f"rel_{relation_data.get('head')}_{relation_data.get('tail')}"),
                "type": "relation",
                "head_entity": relation_data.get("head_entity", ""),
                "tail_entity": relation_data.get("tail_entity", ""),
                "relation_type": relation_data.get("relation_type", "unknown"),
                "domain": domain,
                "confidence": relation_data.get("confidence", 1.0),
                "metadata": relation_data.get("metadata", {}),
                "created_at": relation_data.get("created_at", "")
            }

            result = self.container.create_item(body=item)

            return {
                "success": True,
                "id": result["id"],
                "relation_type": result["relation_type"]
            }

        except Exception as e:
            logger.error(f"Relation storage failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "relation_id": relation_data.get("id", "unknown")
            }

    def query_entities_by_type(self, entity_type: str, domain: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query entities by type and domain"""
        try:
            query = f"""
                SELECT * FROM c
                WHERE c.type = 'entity'
                AND c.entity_type = '{entity_type}'
                AND c.domain = '{domain}'
                ORDER BY c.confidence DESC
                OFFSET 0 LIMIT {limit}
            """

            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))

            return items

        except Exception as e:
            logger.error(f"Entity query failed: {e}")
            return []

    def query_relations_by_entity(self, entity_text: str, domain: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Query relations involving specific entity"""
        try:
            query = f"""
                SELECT * FROM c
                WHERE c.type = 'relation'
                AND (c.head_entity = '{entity_text}' OR c.tail_entity = '{entity_text}')
                AND c.domain = '{domain}'
                ORDER BY c.confidence DESC
                OFFSET 0 LIMIT {limit}
            """

            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))

            return items

        except Exception as e:
            logger.error(f"Relation query failed: {e}")
            return []

    def get_graph_statistics(self, domain: str) -> Dict[str, Any]:
        """Get knowledge graph statistics for domain"""
        try:
            # Count entities
            entity_query = f"SELECT VALUE COUNT(1) FROM c WHERE c.type = 'entity' AND c.domain = '{domain}'"
            entity_count = list(self.container.query_items(
                query=entity_query,
                enable_cross_partition_query=True
            ))[0]

            # Count relations
            relation_query = f"SELECT VALUE COUNT(1) FROM c WHERE c.type = 'relation' AND c.domain = '{domain}'"
            relation_count = list(self.container.query_items(
                query=relation_query,
                enable_cross_partition_query=True
            ))[0]

            return {
                "success": True,
                "domain": domain,
                "entity_count": entity_count,
                "relation_count": relation_count,
                "total_items": entity_count + relation_count
            }

        except Exception as e:
            logger.error(f"Statistics query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain
            }

    def get_connection_status(self) -> Dict[str, Any]:
        """Get Cosmos DB connection status - follows azure_openai.py pattern"""
        try:
            # Test connection by getting database properties
            database_props = self.database.read()

            return {
                "status": "healthy",
                "endpoint": self.endpoint,
                "database_name": self.database_name,
                "container_name": self.container_name,
                "database_id": database_props.get("id")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "endpoint": self.endpoint
            }