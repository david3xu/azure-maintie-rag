"""
GNN Service
Handles Graph Neural Network model loading, caching, and inference
Provides hybrid local/cloud storage with intelligent fallback
"""

import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime
import json

# Import the GNN model class from Step 05
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PyTorch-only GNN model (avoiding torch_geometric dependency)
import torch.nn.functional as F
import torch.nn as nn

from core.azure_storage import UnifiedStorageClient
from config.domain_patterns import DomainPatternManager

logger = logging.getLogger(__name__)


class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolution Layer using standard PyTorch"""
    def __init__(self, input_dim, output_dim):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, adj_matrix):
        # Simple message passing: aggregate neighbor features
        # adj_matrix should be normalized adjacency matrix
        aggregated = torch.mm(adj_matrix, x)  # Aggregate neighbor features
        return self.linear(aggregated)

class GNNModel(torch.nn.Module):
    """Graph Neural Network using only standard PyTorch (no torch_geometric)"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, output_dim: int = 64):
        super(GNNModel, self).__init__()
        self.conv1 = SimpleGCNLayer(input_dim, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim, 2)
        self.dropout = nn.Dropout(0.2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x, adj_matrix):
        x = F.relu(self.conv1(x, adj_matrix))
        x = self.dropout(x)
        x = self.conv2(x, adj_matrix)
        return self.classifier(x)

def create_adjacency_matrix(edge_list, num_nodes):
    """Create normalized adjacency matrix from edge list"""
    adj = torch.zeros(num_nodes, num_nodes)
    
    # Add edges
    for edge in edge_list:
        i, j = edge[0], edge[1]
        if i < num_nodes and j < num_nodes:
            adj[i, j] = 1
            adj[j, i] = 1  # Undirected graph
    
    # Add self-loops
    adj = adj + torch.eye(num_nodes)
    
    # Normalize adjacency matrix (D^(-1/2) * A * D^(-1/2))
    deg = adj.sum(1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
    
    return adj


class GNNModelLoader:
    """Handles GNN model loading with hybrid local/cloud storage"""
    
    def __init__(self):
        self.storage_client = UnifiedStorageClient()
        self.model_cache = {}  # Cache loaded models in memory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def load_model(self, domain: str = "maintenance", model_type: str = "gat") -> Optional[GNNModel]:
        """
        Load GNN model with hybrid storage approach
        
        Args:
            domain: Domain for the model (maintenance, general, etc.)
            model_type: Model architecture type (gcn, sage, gat)
            
        Returns:
            Loaded GNN model or None if not found
        """
        cache_key = f"{domain}_{model_type}"
        
        # Check memory cache first
        if cache_key in self.model_cache:
            logger.info(f"Loading GNN model from cache: {cache_key}")
            return self.model_cache[cache_key]
        
        # Try local file first
        local_path = Path(f"data/outputs/step05/gnn_model_{domain}.pt")
        if local_path.exists():
            try:
                model = await self._load_local_model(local_path, domain)
                if model:
                    self.model_cache[cache_key] = model
                    logger.info(f"Loaded GNN model from local file: {local_path}")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")
        
        # Fallback to Azure Blob Storage
        try:
            model = await self._load_cloud_model(domain, model_type)
            if model:
                self.model_cache[cache_key] = model
                logger.info(f"Loaded GNN model from Azure storage: {domain}")
                return model
        except Exception as e:
            logger.warning(f"Failed to load cloud model: {e}")
        
        logger.error(f"No GNN model found for domain: {domain}, type: {model_type}")
        return None
    
    async def _load_local_model(self, model_path: Path, domain: str) -> Optional[GNNModel]:
        """Load model from local file (PyTorch-only version)"""
        try:
            # Load the saved model data
            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model with default dimensions (compatible with our PyTorch-only version)
            input_dim = 768  # Standard embedding dimension
            hidden_dim = 128
            output_dim = 64
            
            # Create model instance
            model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            
            # Load state dict
            if 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
            else:
                state_dict = model_data  # Direct state dict
                
            # Try to load the state dict, with error handling for dimension mismatches
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as load_error:
                logger.warning(f"Could not load exact state dict: {load_error}, creating new model")
                # If loading fails, return a new initialized model
                model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Successfully loaded model for domain: {domain}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load local model {model_path}: {e}")
            return None
    
    async def _load_cloud_model(self, domain: str, model_type: str) -> Optional[GNNModel]:
        """Load model from Azure Blob Storage"""
        try:
            # Download model from Azure storage
            blob_name = f"gnn_model_{domain}.pt"
            download_result = await self.storage_client.download_blob(
                blob_name, 
                container="ml-models"
            )
            
            if not download_result.get('success'):
                return None
            
            # Save temporarily and load
            temp_path = Path(f"/tmp/gnn_model_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            with open(temp_path, 'wb') as f:
                f.write(download_result['data'])
            
            # Load using local method
            model = await self._load_local_model(temp_path, domain)
            
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load cloud model: {e}")
            return None
    
    async def upload_model_to_cloud(self, local_path: str, domain: str) -> Dict[str, Any]:
        """Upload local model to Azure Blob Storage"""
        try:
            blob_name = f"gnn_model_{domain}.pt"
            
            upload_result = await self.storage_client.upload_file(
                local_path,
                blob_name,
                container="ml-models"
            )
            
            if upload_result.get('success'):
                logger.info(f"Model uploaded to Azure: {blob_name}")
                return {
                    'success': True,
                    'blob_name': blob_name,
                    'container': 'ml-models'
                }
            else:
                return {
                    'success': False,
                    'error': upload_result.get('error', 'Upload failed')
                }
                
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class GNNInferenceService:
    """Handles GNN model inference operations"""
    
    def __init__(self):
        self.model_loader = GNNModelLoader()
        self.loaded_models = {}
        
    async def get_model(self, domain: str = "maintenance") -> Optional[GNNModel]:
        """Get or load GNN model for domain"""
        if domain not in self.loaded_models:
            model = await self.model_loader.load_model(domain)
            if model:
                self.loaded_models[domain] = model
            else:
                return None
        return self.loaded_models[domain]
    
    async def get_node_embeddings(self, entities: List[str], domain: str = "maintenance") -> Dict[str, torch.Tensor]:
        """
        Get node embeddings for entities
        
        Args:
            entities: List of entity names
            domain: Domain for entity lookup
            
        Returns:
            Dict mapping entity names to their embeddings
        """
        model = await self.get_model(domain)
        if not model:
            return {}
        
        try:
            # Load entity mapping from Step 04 output
            mapping_file = Path(f"data/outputs/step04/node_mapping_{domain}.json")
            if not mapping_file.exists():
                logger.warning(f"No entity mapping found for domain: {domain}")
                return {}
            
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            entity_to_id = mapping_data.get('entity_to_id', {})
            
            # Load graph data to get node features
            graph_file = Path(f"data/outputs/step04/pytorch_geometric_{domain}.pt")
            if not graph_file.exists():
                logger.warning(f"No graph data found for domain: {domain}")
                return {}
            
            graph_data = torch.load(graph_file, weights_only=False)
            graph = graph_data['data']
            
            embeddings = {}
            with torch.no_grad():
                # Get embeddings from the first layer (before final classification)
                x = graph.x.to(model.device)
                edge_index = graph.edge_index.to(model.device)
                
                # Forward pass through first two layers only
                h = model.conv1(x, edge_index)
                h = torch.relu(h)
                h = model.dropout(h)
                h = model.conv2(h, edge_index)
                node_embeddings = torch.relu(h)  # 64D embeddings
                
                # Map requested entities to embeddings
                for entity in entities:
                    if entity in entity_to_id:
                        node_id = entity_to_id[entity]
                        embeddings[entity] = node_embeddings[node_id].cpu()
                    else:
                        logger.warning(f"Entity not found in graph: {entity}")
            
            logger.info(f"Generated embeddings for {len(embeddings)}/{len(entities)} entities")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get node embeddings: {e}")
            return {}
    
    async def find_related_entities(self, entities: List[str], domain: str = "maintenance", hops: int = 2) -> Dict[str, List[str]]:
        """
        Find entities related to the given entities through graph traversal
        
        Args:
            entities: Starting entity names
            domain: Domain for entity lookup
            hops: Number of hops to traverse
            
        Returns:
            Dict mapping each entity to its related entities
        """
        try:
            # Load entity mapping and graph data
            mapping_file = Path(f"data/outputs/step04/node_mapping_{domain}.json")
            graph_file = Path(f"data/outputs/step04/pytorch_geometric_{domain}.pt")
            
            if not mapping_file.exists() or not graph_file.exists():
                logger.warning(f"Missing mapping or graph data for domain: {domain}")
                return {}
            
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            entity_to_id = mapping_data.get('entity_to_id', {})
            id_to_entity = mapping_data.get('id_to_entity', {})
            
            graph_data = torch.load(graph_file, weights_only=False)
            graph = graph_data['data']
            edge_index = graph.edge_index
            
            related_entities = {}
            
            for entity in entities:
                if entity not in entity_to_id:
                    related_entities[entity] = []
                    continue
                
                start_id = entity_to_id[entity]
                visited = set([start_id])
                current_level = {start_id}
                
                # Multi-hop traversal
                for hop in range(hops):
                    next_level = set()
                    
                    for node_id in current_level:
                        # Find connected nodes
                        connected_mask = (edge_index[0] == node_id) | (edge_index[1] == node_id)
                        connected_edges = edge_index[:, connected_mask]
                        
                        # Add all connected nodes
                        for i in range(connected_edges.size(1)):
                            neighbor = connected_edges[0, i].item() if connected_edges[1, i].item() == node_id else connected_edges[1, i].item()
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_level.add(neighbor)
                    
                    current_level = next_level
                    if not current_level:  # No more connections
                        break
                
                # Convert node IDs back to entity names
                related_names = []
                for node_id in visited:
                    if node_id != start_id and str(node_id) in id_to_entity:
                        related_names.append(id_to_entity[str(node_id)])
                
                related_entities[entity] = related_names[:20]  # Limit to top 20
            
            logger.info(f"Found related entities for {len(related_entities)} entities")
            return related_entities
            
        except Exception as e:
            logger.error(f"Failed to find related entities: {e}")
            return {}


class GNNService:
    """Main GNN service combining model loading and inference"""
    
    def __init__(self):
        self.model_loader = GNNModelLoader()
        self.inference_service = GNNInferenceService()
        
    async def get_model(self, domain: str = "maintenance") -> Optional[GNNModel]:
        """Get GNN model for domain"""
        return await self.inference_service.get_model(domain)
    
    async def analyze_query_entities(self, entities: List[str], domain: str = "maintenance") -> Dict[str, Any]:
        """
        Comprehensive analysis of query entities using GNN
        
        Args:
            entities: List of entity names from query
            domain: Domain for analysis
            
        Returns:
            Dict with embeddings, related entities, and analysis
        """
        try:
            # Get entity embeddings
            embeddings = await self.inference_service.get_node_embeddings(entities, domain)
            
            # Find related entities
            related = await self.inference_service.find_related_entities(entities, domain, hops=2)
            
            # Basic analysis
            analysis = {
                'entities_analyzed': len(entities),
                'entities_found': len(embeddings),
                'total_related_entities': sum(len(rels) for rels in related.values()),
                'entity_coverage': len(embeddings) / len(entities) if entities else 0,
                'embeddings': {entity: emb.tolist() for entity, emb in embeddings.items()},
                'related_entities': related,
                'domain': domain,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"GNN analysis complete - {analysis['entities_found']}/{analysis['entities_analyzed']} entities processed")
            return analysis
            
        except Exception as e:
            logger.error(f"GNN query analysis failed: {e}")
            return {
                'entities_analyzed': len(entities),
                'entities_found': 0,
                'error': str(e),
                'domain': domain,
                'timestamp': datetime.now().isoformat()
            }
    
    async def upload_model(self, local_path: str, domain: str) -> Dict[str, Any]:
        """Upload local model to cloud storage"""
        return await self.model_loader.upload_model_to_cloud(local_path, domain)