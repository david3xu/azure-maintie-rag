"""
Unified Azure Cosmos DB Client  
Consolidates all Cosmos DB functionality: graph operations, bulk loading, queries
Replaces: cosmos_gremlin_client.py, enhanced_gremlin_client.py
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from gremlin_python.driver import client, serializer
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T, P

from .base_client import BaseAzureClient
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class UnifiedCosmosClient(BaseAzureClient):
    """Unified client for all Azure Cosmos DB operations"""
    
    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_cosmos_endpoint
        
    def _get_default_key(self) -> str:
        return azure_settings.azure_cosmos_key
        
    def _initialize_client(self):
        """Initialize Gremlin client"""
        # Extract account name for Gremlin WebSocket endpoint
        account_name = self.endpoint.replace('https://', '').replace('.documents.azure.com:443/', '')
        gremlin_endpoint = f"wss://{account_name}.gremlin.cosmos.azure.com:443/gremlin"
        
        self._client = client.Client(
            gremlin_endpoint,
            'g',
            username=f"/dbs/{azure_settings.azure_cosmos_database}/colls/{azure_settings.azure_cosmos_container}",
            password=self.key,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
    
    # === BULK OPERATIONS ===
    
    async def bulk_load_entities(self, entities: List[Dict], batch_size: int = 50) -> Dict[str, Any]:
        """Bulk load entities to Cosmos DB"""
        self.ensure_initialized()
        
        try:
            total_success = 0
            total_errors = 0
            
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                success, errors = await self._load_entity_batch(batch, i // batch_size + 1)
                total_success += success
                total_errors += errors
                
                # Progress logging
                progress = min((i + batch_size) / len(entities) * 100, 100)
                logger.info(f"Loaded batch {i // batch_size + 1}: {progress:.1f}% complete")
            
            return self.create_success_response('bulk_load_entities', {
                'entities_loaded': total_success,
                'entities_failed': total_errors,
                'success_rate': total_success / len(entities) if entities else 0
            })
            
        except Exception as e:
            return self.handle_azure_error('bulk_load_entities', e)
    
    async def _load_entity_batch(self, entities: List[Dict], batch_num: int) -> Tuple[int, int]:
        """Load a batch of entities"""
        success_count = 0
        error_count = 0
        
        for entity in entities:
            try:
                # Create vertex with properties
                query = f"""
                g.addV('{entity.get('entity_type', 'unknown')}')
                 .property('id', '{entity.get('entity_id', 'unknown')}')
                 .property('text', '{self._escape_string(entity.get('text', '')[:200])}')
                 .property('entity_type', '{entity.get('entity_type', 'unknown')}')
                 .property('domain', 'maintenance')
                """
                
                result = self._client.submit(query).all().result()
                success_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # Log first few errors
                    logger.warning(f"Entity load error: {str(e)[:100]}")
        
        return success_count, error_count
    
    async def bulk_load_relationships(self, relationships: List[Dict]) -> Dict[str, Any]:
        """Bulk load relationships to Cosmos DB"""
        self.ensure_initialized()
        
        try:
            # First check which entities exist
            existing_entities = await self._get_existing_entity_ids()
            
            # Filter valid relationships  
            valid_relationships = [
                rel for rel in relationships
                if rel.get('source_entity_id') in existing_entities 
                and rel.get('target_entity_id') in existing_entities
            ]
            
            total_success = 0
            for rel in valid_relationships:
                try:
                    query = f"""
                    g.V('{rel['source_entity_id']}')
                     .addE('{rel.get('relation_type', 'related')}')
                     .to(g.V('{rel['target_entity_id']}'))
                     .property('relation_type', '{rel.get('relation_type', 'related')}')
                     .property('confidence', {rel.get('confidence', 1.0)})
                    """
                    
                    result = self._client.submit(query).all().result()
                    total_success += 1
                    
                except Exception as e:
                    logger.warning(f"Relationship load error: {str(e)[:100]}")
            
            return self.create_success_response('bulk_load_relationships', {
                'relationships_loaded': total_success,
                'valid_relationships': len(valid_relationships),
                'total_relationships': len(relationships)
            })
            
        except Exception as e:
            return self.handle_azure_error('bulk_load_relationships', e)
    
    # === QUERY OPERATIONS ===
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        self.ensure_initialized()
        
        try:
            # Count vertices
            vertex_result = self._client.submit("g.V().count()").all().result()
            vertex_count = vertex_result[0] if vertex_result else 0
            
            # Count edges  
            edge_result = self._client.submit("g.E().count()").all().result()
            edge_count = edge_result[0] if edge_result else 0
            
            # Get entity types
            type_result = self._client.submit("g.V().groupCount().by('entity_type')").all().result()
            entity_types = len(type_result[0]) if type_result else 0
            
            connectivity = edge_count / vertex_count if vertex_count > 0 else 0
            
            return self.create_success_response('get_graph_stats', {
                'vertices': vertex_count,
                'edges': edge_count,
                'entity_types': entity_types,
                'connectivity_ratio': connectivity
            })
            
        except Exception as e:
            return self.handle_azure_error('get_graph_stats', e)
    
    async def find_multi_hop_paths(self, start_id: str, end_id: str, max_hops: int = 3) -> List[List[str]]:
        """Find paths between entities"""
        self.ensure_initialized()
        
        try:
            query = f"""
            g.V('{start_id}')
             .repeat(both().simplePath())
             .until(hasId('{end_id}').or().loops().is(gte({max_hops})))
             .hasId('{end_id}')
             .path()
             .limit(10)
            """
            
            result = self._client.submit(query).all().result()
            
            paths = []
            for path in result:
                path_ids = [vertex.id for vertex in path]
                paths.append(path_ids)
            
            return paths
            
        except Exception as e:
            logger.error(f"Multi-hop path finding failed: {e}")
            return []
    
    # === UTILITY METHODS ===
    
    async def clear_graph(self) -> Dict[str, Any]:
        """Clear all data from graph"""
        self.ensure_initialized()
        
        try:
            # Drop all edges first
            self._client.submit("g.E().drop()").all().result()
            
            # Drop all vertices
            self._client.submit("g.V().drop()").all().result()
            
            return self.create_success_response('clear_graph', {
                'message': 'Graph cleared successfully'
            })
            
        except Exception as e:
            return self.handle_azure_error('clear_graph', e)
    
    async def _get_existing_entity_ids(self) -> set:
        """Get all existing entity IDs"""
        try:
            result = self._client.submit("g.V().id()").all().result()
            return set(result)
        except:
            return set()
    
    def _escape_string(self, text: str) -> str:
        """Escape string for Gremlin queries"""
        return text.replace("'", "\\\\'").replace('"', '\\\\"').replace('\\n', ' ')