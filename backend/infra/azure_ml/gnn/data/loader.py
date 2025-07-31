"""
Consolidated GNN Data Loader
Combines functionality from gnn_processor.py and data_loader.py
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from datetime import datetime

from config.settings import azure_settings
from config.domain_patterns import DomainPatternManager

logger = logging.getLogger(__name__)


class UnifiedGNNDataLoader:
    """
    Unified data loader for GNN training
    Consolidates graph data loading and processing functionality
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.gnn_data_dir = azure_settings.processed_data_dir / "gnn" / domain
        self.gnn_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Universal entity and relation mappings
        self.entity_to_idx: Dict[str, int] = {}
        self.relation_to_idx: Dict[str, int] = {} 
        self.idx_to_entity: Dict[int, str] = {}
        self.idx_to_relation: Dict[int, str] = {}

    async def load_graph_data_from_cosmos(
        self, 
        cosmos_client,
        domain: str,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load graph data from Cosmos DB and create PyTorch Geometric data loaders
        
        Returns:
            Tuple of (train_loader, validation_loader)
        """
        try:
            logger.info(f"Loading graph data for domain: {domain}")
            
            # Export graph data from Cosmos DB
            graph_export = cosmos_client.export_graph_for_training(domain)
            
            if not graph_export.get("success", False):
                raise ValueError(f"Failed to export graph data: {graph_export.get('error', 'Unknown error')}")
            
            entities = graph_export["entities"]
            relations = graph_export["relations"]
            
            logger.info(f"Loaded {len(entities)} entities and {len(relations)} relations")
            
            # Build mappings
            self._build_entity_mappings(entities)
            self._build_relation_mappings(relations)
            
            # Convert to PyTorch Geometric format
            data_list = self._create_pytorch_geometric_data(entities, relations)
            
            # Split into train/validation
            train_data, val_data = self._split_data(data_list, validation_split)
            
            # Create data loaders
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            logger.info(f"Created train loader with {len(train_data)} samples, val loader with {len(val_data)} samples")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            raise

    def _build_entity_mappings(self, entities: List[Dict[str, Any]]) -> None:
        """Build entity to index mappings"""
        for idx, entity in enumerate(entities):
            entity_text = entity.get("text", f"entity_{idx}")
            self.entity_to_idx[entity_text] = idx
            self.idx_to_entity[idx] = entity_text

    def _build_relation_mappings(self, relations: List[Dict[str, Any]]) -> None:
        """Build relation to index mappings"""
        unique_relations = set()
        for relation in relations:
            rel_type = relation.get("relation_type", "RELATES_TO")
            unique_relations.add(rel_type)
        
        for idx, rel_type in enumerate(sorted(unique_relations)):
            self.relation_to_idx[rel_type] = idx
            self.idx_to_relation[idx] = rel_type

    def _create_pytorch_geometric_data(
        self, 
        entities: List[Dict[str, Any]], 
        relations: List[Dict[str, Any]]
    ) -> List[Data]:
        """Convert graph data to PyTorch Geometric format"""
        
        # Create node features (simplified)
        node_features = []
        for entity in entities:
            # Create feature vector from entity properties
            feature_vector = [
                len(entity.get("text", "")),  # Text length
                hash(entity.get("entity_type", "unknown")) % 1000,  # Entity type hash
                entity.get("confidence", 0.5),  # Confidence score
                hash(self.domain) % 100  # Domain hash
            ]
            node_features.append(feature_vector)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge indices and features
        edge_indices = []
        edge_features = []
        
        for relation in relations:
            source_entity = relation.get("source_entity", "")
            target_entity = relation.get("target_entity", "")
            
            if source_entity in self.entity_to_idx and target_entity in self.entity_to_idx:
                source_idx = self.entity_to_idx[source_entity]
                target_idx = self.entity_to_idx[target_entity]
                
                edge_indices.append([source_idx, target_idx])
                
                # Create edge feature vector
                rel_type = relation.get("relation_type", "RELATES_TO")
                edge_feature = [
                    self.relation_to_idx.get(rel_type, 0),  # Relation type index
                    relation.get("confidence", 0.5),  # Confidence
                    1.0  # Weight
                ]
                edge_features.append(edge_feature)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Handle case with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        
        # Create labels (for now, use entity types as labels)
        y = torch.zeros(len(entities), dtype=torch.long)
        for idx, entity in enumerate(entities):
            entity_type = entity.get("entity_type", "unknown")
            # Simple hash-based labeling (would be more sophisticated in practice)
            y[idx] = hash(entity_type) % 10  # Assume 10 classes
        
        # Create single graph data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        return [data]  # Return as list for compatibility

    def _split_data(self, data_list: List[Data], validation_split: float) -> Tuple[List[Data], List[Data]]:
        """Split data into training and validation sets"""
        if not data_list:
            return [], []
        
        split_idx = int(len(data_list) * (1 - validation_split))
        
        train_data = data_list[:split_idx] if split_idx > 0 else data_list
        val_data = data_list[split_idx:] if split_idx < len(data_list) else [data_list[-1]]  # Ensure at least one validation sample
        
        return train_data, val_data

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        return {
            "num_entities": len(self.entity_to_idx),
            "num_relations": len(self.relation_to_idx),
            "domain": self.domain,
            "data_dir": str(self.gnn_data_dir),
            "entity_types": list(self.relation_to_idx.keys())
        }


# Legacy function for backward compatibility
async def load_graph_data_from_cosmos(
    cosmos_client, 
    domain: str = "general",
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Legacy function for backward compatibility
    Use UnifiedGNNDataLoader directly for new code
    """
    loader = UnifiedGNNDataLoader(domain)
    return await loader.load_graph_data_from_cosmos(cosmos_client, domain, batch_size)