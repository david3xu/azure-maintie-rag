"""
Graph Service
Handles all knowledge graph operations and management
Consolidates: graph loading, querying, analysis, multi-hop reasoning
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime

from core.azure_unified import UnifiedCosmosClient, UnifiedStorageClient

logger = logging.getLogger(__name__)


class GraphService:
    """High-level service for knowledge graph operations"""
    
    def __init__(self):
        self.cosmos_client = UnifiedCosmosClient()
        self.storage_client = UnifiedStorageClient()
        
    # === GRAPH LOADING ===
    
    async def load_knowledge_to_graph(self, entities: List[Dict], relationships: List[Dict], 
                                    batch_size: int = 50, clear_existing: bool = False) -> Dict[str, Any]:
        """Load entities and relationships to knowledge graph"""
        try:
            # Clear existing data if requested
            if clear_existing:
                clear_result = await self.cosmos_client.clear_graph()
                if not clear_result['success']:
                    return clear_result
            
            # Load entities
            entity_result = await self.cosmos_client.bulk_load_entities(entities, batch_size)
            if not entity_result['success']:
                return entity_result
            
            # Load relationships
            rel_result = await self.cosmos_client.bulk_load_relationships(relationships)
            
            # Get final statistics
            stats_result = await self.cosmos_client.get_graph_stats()
            
            # Save loading results
            loading_data = {
                'timestamp': datetime.now().isoformat(),
                'entities_loaded': entity_result['data']['entities_loaded'],
                'relationships_loaded': rel_result['data']['relationships_loaded'] if rel_result['success'] else 0,
                'final_stats': stats_result['data'] if stats_result['success'] else {},
                'batch_size': batch_size,
                'clear_existing': clear_existing
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            await self.storage_client.save_json(
                loading_data,
                f"graph_loading_{timestamp}.json",
                container="graph-results"
            )
            
            return {
                'success': True,
                'operation': 'load_knowledge_to_graph',
                'data': loading_data
            }
            
        except Exception as e:
            logger.error(f"Graph loading failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'load_knowledge_to_graph'
            }
    
    async def load_from_extraction_file(self, file_path: str, max_entities: int = None, 
                                      batch_size: int = 50) -> Dict[str, Any]:
        """Load graph from extraction results file"""
        try:
            # Load extraction data
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            entities = data.get('entities', [])
            relationships = data.get('relationships', [])
            
            # Limit entities if specified
            if max_entities and len(entities) > max_entities:
                entities = entities[:max_entities]
                # Filter relationships to match limited entities
                entity_ids = {e['entity_id'] for e in entities}
                relationships = [
                    r for r in relationships 
                    if r.get('source_entity_id') in entity_ids and r.get('target_entity_id') in entity_ids
                ]
            
            # Load to graph
            return await self.load_knowledge_to_graph(entities, relationships, batch_size)
            
        except Exception as e:
            logger.error(f"Loading from extraction file failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'load_from_extraction_file'
            }
    
    # === GRAPH ANALYSIS ===
    
    async def analyze_graph(self) -> Dict[str, Any]:
        """Perform comprehensive graph analysis"""
        try:
            # Get basic statistics
            stats_result = await self.cosmos_client.get_graph_stats()
            if not stats_result['success']:
                return stats_result
            
            stats = stats_result['data']
            
            # Perform analysis
            analysis = {
                'basic_stats': stats,
                'connectivity_analysis': self._analyze_connectivity(stats),
                'scale_assessment': self._assess_scale(stats),
                'recommendations': self._generate_recommendations(stats)
            }
            
            return {
                'success': True,
                'operation': 'analyze_graph',
                'data': analysis
            }
            
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'analyze_graph'
            }
    
    # === MULTI-HOP REASONING ===
    
    async def find_reasoning_paths(self, start_entities: List[str], end_entities: List[str], 
                                 max_hops: int = 3, max_paths: int = 10) -> Dict[str, Any]:
        """Find multi-hop reasoning paths between entity sets"""
        try:
            all_paths = []
            
            # Find paths between all combinations
            for start_entity in start_entities:
                for end_entity in end_entities:
                    paths = await self.cosmos_client.find_multi_hop_paths(
                        start_entity, end_entity, max_hops
                    )
                    
                    for path in paths[:max_paths // len(start_entities) // len(end_entities)]:
                        all_paths.append({
                            'start_entity': start_entity,
                            'end_entity': end_entity,
                            'path': path,
                            'hops': len(path) - 1
                        })
            
            # Analyze reasoning patterns
            reasoning_analysis = self._analyze_reasoning_patterns(all_paths)
            
            return {
                'success': True,
                'operation': 'find_reasoning_paths',
                'data': {
                    'paths': all_paths,
                    'path_count': len(all_paths),
                    'analysis': reasoning_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'find_reasoning_paths'
            }
    
    async def query_graph(self, query_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute various graph queries"""
        try:
            if query_type == "neighbors":
                return await self._query_neighbors(parameters)
            elif query_type == "shortest_path":
                return await self._query_shortest_path(parameters)
            elif query_type == "subgraph":
                return await self._query_subgraph(parameters)
            else:
                return {
                    'success': False,
                    'error': f"Unknown query type: {query_type}",
                    'operation': 'query_graph'
                }
                
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'query_graph'
            }
    
    # === GRAPH MAINTENANCE ===
    
    async def cleanup_graph(self, keep_backup: bool = True) -> Dict[str, Any]:
        """Clean up graph data"""
        try:
            # Create backup if requested
            backup_result = None
            if keep_backup:
                backup_result = await self._create_graph_backup()
            
            # Clear graph
            clear_result = await self.cosmos_client.clear_graph()
            
            return {
                'success': clear_result['success'],
                'operation': 'cleanup_graph',
                'data': {
                    'backup_created': backup_result['success'] if backup_result else False,
                    'graph_cleared': clear_result['success']
                }
            }
            
        except Exception as e:
            logger.error(f"Graph cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'cleanup_graph'
            }
    
    # === ANALYSIS UTILITIES ===
    
    def _analyze_connectivity(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        vertices = stats.get('vertices', 0)
        edges = stats.get('edges', 0)
        connectivity = stats.get('connectivity_ratio', 0)
        
        if connectivity < 1:
            level = "sparse"
        elif connectivity < 10:
            level = "moderate"  
        elif connectivity < 100:
            level = "dense"
        else:
            level = "highly_connected"
        
        return {
            'connectivity_ratio': connectivity,
            'connectivity_level': level,
            'average_degree': (edges * 2) / vertices if vertices > 0 else 0
        }
    
    def _assess_scale(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess graph scale and performance characteristics"""
        vertices = stats.get('vertices', 0)
        edges = stats.get('edges', 0)
        
        if vertices < 100:
            scale = "small"
            performance = "excellent"
        elif vertices < 1000:
            scale = "medium"
            performance = "good"
        elif vertices < 10000:
            scale = "large"
            performance = "moderate"
        else:
            scale = "very_large"
            performance = "requires_optimization"
        
        return {
            'scale': scale,
            'expected_performance': performance,
            'vertices': vertices,
            'edges': edges
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on graph characteristics"""
        recommendations = []
        
        connectivity = stats.get('connectivity_ratio', 0)
        vertices = stats.get('vertices', 0)
        
        if connectivity < 1:
            recommendations.append("Consider adding more relationships to improve graph connectivity")
        
        if vertices > 10000:
            recommendations.append("Consider implementing graph partitioning for better performance")
        
        if connectivity > 1000:
            recommendations.append("Very high connectivity detected - ensure relationship quality")
        
        return recommendations
    
    def _analyze_reasoning_patterns(self, paths: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in multi-hop reasoning paths"""
        if not paths:
            return {'pattern_count': 0}
        
        hop_distribution = {}
        for path in paths:
            hops = path['hops']
            hop_distribution[hops] = hop_distribution.get(hops, 0) + 1
        
        return {
            'pattern_count': len(paths),
            'hop_distribution': hop_distribution,
            'average_hops': sum(p['hops'] for p in paths) / len(paths),
            'max_hops': max(p['hops'] for p in paths),
            'min_hops': min(p['hops'] for p in paths)
        }
    
    # === QUERY IMPLEMENTATIONS ===
    
    async def _query_neighbors(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query entity neighbors"""
        # Simplified implementation - would use Cosmos client
        return {
            'success': True,
            'operation': 'query_neighbors',
            'data': {'neighbors': []}
        }
    
    async def _query_shortest_path(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Find shortest path between entities"""
        start_id = parameters.get('start_id')
        end_id = parameters.get('end_id')
        
        if start_id and end_id:
            paths = await self.cosmos_client.find_multi_hop_paths(start_id, end_id, max_hops=5)
            shortest = min(paths, key=len) if paths else []
            
            return {
                'success': True,
                'operation': 'query_shortest_path',
                'data': {'shortest_path': shortest}
            }
        
        return {
            'success': False,
            'error': 'Missing start_id or end_id',
            'operation': 'query_shortest_path'
        }
    
    async def _query_subgraph(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract subgraph around entities"""
        # Simplified implementation
        return {
            'success': True,
            'operation': 'query_subgraph',
            'data': {'subgraph': {}}
        }
    
    async def _create_graph_backup(self) -> Dict[str, Any]:
        """Create backup of current graph state"""
        try:
            stats = await self.cosmos_client.get_graph_stats()
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': stats['data'] if stats['success'] else {},
                'backup_type': 'statistics_only'
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return await self.storage_client.save_json(
                backup_data,
                f"graph_backup_{timestamp}.json",
                container="backups"
            )
            
        except Exception as e:
            return {'success': False, 'error': str(e)}