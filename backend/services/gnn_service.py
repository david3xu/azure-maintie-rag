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

# Import PyTorch Geometric for model creation
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F

from core.azure_storage import UnifiedStorageClient
from config.domain_patterns import DomainPatternManager

logger = logging.getLogger(__name__)


class GNNModel(torch.nn.Module):
    """Graph Neural Network for maintenance domain node classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, model_type: str = "gcn"):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, output_dim)
        elif model_type == "sage":
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, output_dim)
        elif model_type == "gat":
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.dropout = torch.nn.Dropout(0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x, edge_index, edge_attr=None):
        # First convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final convolution
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


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
        """Load model from local file"""
        try:
            # Load the saved model data
            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            model_type = model_data.get('model_type', 'gat')
            training_results = model_data.get('training_results', {})
            
            # Get domain configuration for model dimensions
            pytorch_config = DomainPatternManager.get_pytorch_geometric(domain)
            
            # Determine model dimensions from saved state
            state_dict = model_data['model_state_dict']
            
            # Infer dimensions from the first layer
            input_dim = None
            output_dim = None
            hidden_dim = pytorch_config.node_feature_dim if hasattr(pytorch_config, 'node_feature_dim') else 64
            
            for key, tensor in state_dict.items():
                if 'conv1' in key and 'weight' in key:
                    if input_dim is None:
                        input_dim = tensor.size(1) if len(tensor.shape) > 1 else tensor.size(0)
                elif 'conv3' in key and 'weight' in key:
                    if output_dim is None:
                        output_dim = tensor.size(0)
            
            # Default dimensions if not found
            if input_dim is None:
                input_dim = 64
            if output_dim is None:
                output_dim = 10
            
            # Create model instance
            model = GNNModel(input_dim, hidden_dim, output_dim, model_type)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Local model loaded - Type: {model_type}, Input: {input_dim}D, Output: {output_dim}D")
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