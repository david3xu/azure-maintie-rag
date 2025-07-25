"""
Enhanced Azure Cosmos DB Gremlin Client for Enterprise GNN Pipeline
Supports GNN embeddings, real-time updates, and enterprise graph management
"""

import logging
import concurrent.futures
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from gremlin_python.driver import client, serializer
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T, P

logger = logging.getLogger(__name__)

class EnterpriseGremlinGraphManager:
    """Enterprise-grade Cosmos DB Gremlin integration with GNN embeddings"""

    def __init__(self, cosmos_endpoint: str, cosmos_key: str, database_name: str = "universal-rag-db", container_name: str = "knowledge-graph"):
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.database_name = database_name
        self.container_name = container_name
        self.gremlin_client = None
        self._client_initialized = False

    def _initialize_client(self):
        """Enterprise Gremlin client initialization with Azure service endpoint validation"""
        if self._client_initialized:
            return
        try:
            # Extract account name from endpoint (following cosmos_gremlin_client.py pattern)
            if 'documents.azure.com' not in self.cosmos_endpoint:
                raise ValueError(f"Invalid Azure Cosmos DB endpoint: {self.cosmos_endpoint}")
            account_name = self.cosmos_endpoint.replace('https://', '').replace('.documents.azure.com:443/', '')
            gremlin_endpoint = f"wss://{account_name}.gremlin.cosmosdb.azure.com:443/"
            logger.info(f"Initializing Gremlin client with endpoint: {gremlin_endpoint}")
            self.gremlin_client = client.Client(
                gremlin_endpoint,
                'g',
                username=f"/dbs/{self.database_name}/colls/{self.container_name}",
                password=self.cosmos_key,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            self._client_initialized = True
            logger.info("Azure Cosmos DB Gremlin client initialized successfully")
        except Exception as e:
            logger.error(f"Azure Cosmos DB Gremlin client initialization failed: {e}")
            self._client_initialized = False
            raise

    def _execute_gremlin_query_safe(self, query: str, timeout_seconds: int = 30):
        """Enterprise thread-isolated Gremlin query execution with proper error propagation"""
        def _run_gremlin_query():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", DeprecationWarning)
                    result = self.gremlin_client.submit(query)
                    return result.all().result()
            except Exception as e:
                logger.error(f"Gremlin query execution failed: {e}", exc_info=True)
                raise RuntimeError(f"Azure Cosmos DB Gremlin query failed: {e}") from e  # ✅ Raise error

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_gremlin_query)
                return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as e:
            error_msg = f"Gremlin query timed out after {timeout_seconds}s: {query}"
            logger.error(error_msg)
            raise TimeoutError(error_msg) from e  # ✅ Raise timeout error
        except Exception as e:
            logger.error(f"Thread execution failed for Gremlin query: {e}", exc_info=True)
            raise RuntimeError(f"Azure Cosmos DB thread execution failed: {e}") from e  # ✅ Raise error

    def _execute_gremlin_query(self, query: str, timeout_seconds: int = 30):
        """Wrapper for backward compatibility - calls synchronous _execute_gremlin_query_safe"""
        return self._execute_gremlin_query_safe(query, timeout_seconds)

    def get_graph_change_metrics(self, domain: str) -> Dict[str, Any]:
        """Get graph change metrics for domain"""
        try:
            if not self._client_initialized:
                self._initialize_client()
            new_entities_query = f"g.V().has('domain', '{domain}').count()"
            new_entities_result = self._execute_gremlin_query_safe(new_entities_query)
            new_entities = new_entities_result[0] if new_entities_result else 0
            new_relations_query = f"g.E().has('domain', '{domain}').count()"
            new_relations_result = self._execute_gremlin_query_safe(new_relations_query)
            new_relations = new_relations_result[0] if new_relations_result else 0
            total_changes = new_entities + new_relations
            return {
                'domain': domain,
                'analysis_timestamp': datetime.now().isoformat(),
                'trigger_threshold': 100,
                'new_entities': new_entities,
                'new_relations': new_relations,
                'updated_entities': 0,
                'total_changes': total_changes,
                'requires_training': total_changes >= 100
            }
        except Exception as e:
            logger.error(f"Failed to get graph change metrics for domain {domain}: {e}")
            return {
                'domain': domain,
                'analysis_timestamp': datetime.now().isoformat(),
                'trigger_threshold': 100,
                'new_entities': 0,
                'new_relations': 0,
                'updated_entities': 0,
                'total_changes': 0,
                'requires_training': False
            }

    def get_entities_without_embeddings(self, domain: str) -> List[str]:
        """Get entities without GNN embeddings"""
        try:
            if not self._client_initialized:
                self._initialize_client()
            query = f"""
                g.V().has('domain', '{domain}')
                    .not(__.has('gnn_embeddings'))
                    .values('entity_id')
            """
            results = self._execute_gremlin_query_safe(query)
            return results or []
        except Exception as e:
            logger.error(f"Failed to get entities without embeddings for domain {domain}: {e}")
            return []

    def store_entity_with_embeddings(
        self,
        entity_data: Dict[str, Any],
        gnn_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Store entity with optional GNN embeddings"""
        try:
            if not self._client_initialized:
                self._initialize_client()
            entity_id = entity_data.get("entity_id", f"entity_{int(datetime.now().timestamp())}")
            entity_text = str(entity_data.get("text", "")).replace("'", "\\'")[:500]
            query = f"""
                g.addV('Entity')
                    .property('entity_id', '{entity_id}')
                    .property('text', '{entity_text}')
                    .property('entity_type', '{entity_data.get("entity_type", "unknown")}')
                    .property('domain', '{entity_data.get("domain", "general")}')
                    .property('confidence', {entity_data.get("confidence", 1.0)})
                    .property('extraction_method', '{entity_data.get("extraction_method", "azure_openai")}')
                    .property('last_updated', '{datetime.now().isoformat()}')
            """
            if gnn_embeddings is not None:
                embedding_str = ','.join(map(str, gnn_embeddings.tolist()))
                query += f".property('gnn_embeddings', '{embedding_str}')"
                query += f".property('embedding_dimension', {len(gnn_embeddings)})"
                query += f".property('embedding_updated_at', '{datetime.now().isoformat()}')"
            result = self._execute_gremlin_query_safe(query)
            logger.info(f"Stored entity: {entity_id} with embeddings: {gnn_embeddings is not None}")
            return {"success": True, "entity_id": entity_id}
        except Exception as e:
            logger.error(f"Failed to store entity with embeddings: {e}")
            return {"success": False, "error": str(e)}

    def close(self):
        """Enterprise-safe Gremlin client cleanup with connection leak prevention"""
        try:
            if self.gremlin_client and self._client_initialized:
                import concurrent.futures
                import warnings
                def _safe_close_with_timeout():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        # Add explicit connection pool cleanup
                        if hasattr(self.gremlin_client, '_transport'):
                            self.gremlin_client._transport.close()
                        self.gremlin_client.close()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_safe_close_with_timeout)
                    future.result(timeout=5)
            logger.info("Azure Cosmos Gremlin client closed successfully")
        except Exception as e:
            logger.warning(f"Gremlin client cleanup warning: {e}")
        finally:
            self._client_initialized = False
            self.gremlin_client = None