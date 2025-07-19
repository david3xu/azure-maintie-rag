"""
Enhanced Azure Cosmos DB Gremlin Client for Enterprise GNN Pipeline
Supports GNN embeddings, real-time updates, and enterprise graph management
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from gremlin_python.driver import client
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T, P

logger = logging.getLogger(__name__)


class EnterpriseGremlinGraphManager:
    """Enterprise-grade Cosmos DB Gremlin integration with GNN embeddings"""

    def __init__(self, cosmos_endpoint: str, cosmos_key: str, database_name: str = "universal-rag-db"):
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.database_name = database_name
        self.container_name = "knowledge-graph"
        self._client = None
        self._graph = Graph()

    async def _get_client(self):
        """Get or create Gremlin client"""
        if self._client is None:
            self._client = client.Client(
                f'wss://{self.cosmos_endpoint}:443/gremlin',
                'g',
                username=f'/dbs/{self.database_name}/colls/{self.container_name}',
                password=self.cosmos_key
            )
        return self._client

    async def store_entity_with_embeddings(
        self,
        entity_data: Dict[str, Any],
        gnn_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Store entity with optional GNN embeddings"""

        try:
            client = await self._get_client()

            # Build Gremlin query
            query = f"""
                g.addV('Entity')
                    .property('entity_id', '{entity_data.get("entity_id")}')
                    .property('text', '{entity_data.get("text", "")}')
                    .property('entity_type', '{entity_data.get("entity_type", "unknown")}')
                    .property('domain', '{entity_data.get("domain", "general")}')
                    .property('confidence', {entity_data.get("confidence", 1.0)})
                    .property('extraction_method', '{entity_data.get("extraction_method", "azure_openai")}')
                    .property('last_updated', '{datetime.now().isoformat()}')
            """

            # Add GNN embeddings if available
            if gnn_embeddings is not None:
                embedding_str = ','.join(map(str, gnn_embeddings.tolist()))
                query += f".property('gnn_embeddings', '{embedding_str}')"
                query += f".property('embedding_dimension', {len(gnn_embeddings)})"
                query += f".property('embedding_updated_at', '{datetime.now().isoformat()}')"

            # Execute query
            result = await self._execute_gremlin_query(query)

            logger.info(f"Stored entity: {entity_data.get('entity_id')} with embeddings: {gnn_embeddings is not None}")
            return result

        except Exception as e:
            logger.error(f"Failed to store entity with embeddings: {e}")
            raise

    async def store_relation_with_embeddings(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_data: Dict[str, Any],
        gnn_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Store relation with optional GNN embeddings"""

        try:
            client = await self._get_client()

            query = f"""
                g.V().has('entity_id', '{source_entity_id}')
                    .as('source')
                    .V().has('entity_id', '{target_entity_id}')
                    .as('target')
                    .addE('{relation_data.get("relation_type", "RELATED_TO")}')
                    .from('source')
                    .to('target')
                    .property('relation_id', '{relation_data.get("relation_id")}')
                    .property('confidence', {relation_data.get("confidence", 1.0)})
                    .property('domain', '{relation_data.get("domain", "general")}')
                    .property('last_updated', '{datetime.now().isoformat()}')
            """

            # Add GNN embeddings if available
            if gnn_embeddings is not None:
                embedding_str = ','.join(map(str, gnn_embeddings.tolist()))
                query += f".property('gnn_embeddings', '{embedding_str}')"
                query += f".property('embedding_dimension', {len(gnn_embeddings)})"
                query += f".property('embedding_updated_at', '{datetime.now().isoformat()}')"

            result = await self._execute_gremlin_query(query)

            logger.info(f"Stored relation: {relation_data.get('relation_id')} with embeddings: {gnn_embeddings is not None}")
            return result

        except Exception as e:
            logger.error(f"Failed to store relation with embeddings: {e}")
            raise

    async def update_graph_embeddings(
        self,
        trained_model: torch.nn.Module,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """Update all entities/relations with fresh GNN embeddings"""

        try:
            # Load current graph data
            graph_data = await self.export_graph_for_training(domain)

            # Generate embeddings using trained model
            with torch.no_grad():
                embeddings = await self._generate_embeddings(trained_model, graph_data)

            # Update Cosmos DB with new embeddings
            update_stats = {"entities_updated": 0, "relations_updated": 0}

            # Update entity embeddings
            for entity_id, embedding in embeddings.get("entities", {}).items():
                update_query = f"""
                    g.V().has('entity_id', '{entity_id}')
                        .property('gnn_embeddings', '{embedding.tolist()}')
                        .property('embedding_updated_at', '{datetime.now().isoformat()}')
                """
                await self._execute_gremlin_query(update_query)
                update_stats["entities_updated"] += 1

            # Update relation embeddings
            for relation_id, embedding in embeddings.get("relations", {}).items():
                update_query = f"""
                    g.E().has('relation_id', '{relation_id}')
                        .property('gnn_embeddings', '{embedding.tolist()}')
                        .property('embedding_updated_at', '{datetime.now().isoformat()}')
                """
                await self._execute_gremlin_query(update_query)
                update_stats["relations_updated"] += 1

            logger.info(f"Updated embeddings: {update_stats}")
            return update_stats

        except Exception as e:
            logger.error(f"Failed to update graph embeddings: {e}")
            raise

    async def export_graph_for_training(self, domain: str = "general") -> Dict[str, Any]:
        """Export graph data for GNN training"""

        try:
            client = await self._get_client()

            # Get all entities in domain
            entities_query = f"""
                g.V().has('domain', '{domain}')
                    .project('entity_id', 'text', 'entity_type', 'gnn_embeddings')
                    .by('entity_id')
                    .by('text')
                    .by('entity_type')
                    .by('gnn_embeddings')
            """

            # Get all relations in domain
            relations_query = f"""
                g.E().has('domain', '{domain}')
                    .project('relation_id', 'source', 'target', 'relation_type', 'gnn_embeddings')
                    .by('relation_id')
                    .by(__.outV().values('entity_id'))
                    .by(__.inV().values('entity_id'))
                    .by(__.label())
                    .by('gnn_embeddings')
            """

            entities_result = await self._execute_gremlin_query(entities_query)
            relations_result = await self._execute_gremlin_query(relations_query)

            return {
                "entities": entities_result,
                "relations": relations_result,
                "domain": domain,
                "export_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to export graph for training: {e}")
            raise

    async def get_graph_change_metrics(self, domain: str = "general") -> Dict[str, int]:
        """Get metrics about graph changes for triggering retraining"""

        try:
            client = await self._get_client()

            # Count new entities in last 24 hours
            new_entities_query = f"""
                g.V().has('domain', '{domain}')
                    .has('last_updated', P.gte('{datetime.now().replace(hour=0, minute=0, second=0).isoformat()}'))
                    .count()
            """

            # Count new relations in last 24 hours
            new_relations_query = f"""
                g.E().has('domain', '{domain}')
                    .has('last_updated', P.gte('{datetime.now().replace(hour=0, minute=0, second=0).isoformat()}'))
                    .count()
            """

            new_entities = await self._execute_gremlin_query(new_entities_query)
            new_relations = await self._execute_gremlin_query(new_relations_query)

            return {
                "new_entities": new_entities[0] if new_entities else 0,
                "new_relations": new_relations[0] if new_relations else 0,
                "total_entities": await self._get_total_entities(domain),
                "total_relations": await self._get_total_relations(domain)
            }

        except Exception as e:
            logger.error(f"Failed to get graph change metrics: {e}")
            return {"new_entities": 0, "new_relations": 0, "total_entities": 0, "total_relations": 0}

    async def _execute_gremlin_query(self, query: str) -> List[Any]:
        """Execute Gremlin query and return results"""
        try:
            client = await self._get_client()
            result = client.submit(query)
            return result.all().result()
        except Exception as e:
            logger.error(f"Gremlin query failed: {query}, error: {e}")
            raise

    async def _generate_embeddings(
        self,
        trained_model: torch.nn.Module,
        graph_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings using trained GNN model"""

        # Convert graph data to PyTorch format
        # This is a simplified version - in practice, you'd use proper graph conversion
        embeddings = {
            "entities": {},
            "relations": {}
        }

        # Generate entity embeddings
        for entity in graph_data.get("entities", []):
            entity_id = entity.get("entity_id")
            if entity_id:
                # Use model to generate embedding
                # This is placeholder - actual implementation depends on your model
                embedding = np.random.rand(128)  # Placeholder
                embeddings["entities"][entity_id] = embedding

        # Generate relation embeddings
        for relation in graph_data.get("relations", []):
            relation_id = relation.get("relation_id")
            if relation_id:
                # Use model to generate embedding
                embedding = np.random.rand(128)  # Placeholder
                embeddings["relations"][relation_id] = embedding

        return embeddings

    async def _get_total_entities(self, domain: str) -> int:
        """Get total number of entities in domain"""
        query = f"g.V().has('domain', '{domain}').count()"
        result = await self._execute_gremlin_query(query)
        return result[0] if result else 0

    async def _get_total_relations(self, domain: str) -> int:
        """Get total number of relations in domain"""
        query = f"g.E().has('domain', '{domain}').count()"
        result = await self._execute_gremlin_query(query)
        return result[0] if result else 0