"""
GNN Data Preparation from MaintIE Annotations
Professional implementation that builds on existing data_transformer.py
"""

import json
import logging
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

try:
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.transforms import ToUndirected
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Define fallback classes for when PyTorch Geometric is not available
    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class HeteroData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

from core.knowledge.data_transformer import MaintIEDataTransformer
from core.models.maintenance_models import MaintenanceEntity, MaintenanceRelation
from config.settings import settings

logger = logging.getLogger(__name__)

class MaintIEGNNDataProcessor:
    """Convert MaintIE annotations to GNN-ready format"""

    def __init__(self, data_transformer: MaintIEDataTransformer):
        """Initialize with existing data transformer"""
        self.data_transformer = data_transformer
        self.gnn_data_dir = settings.processed_data_dir / "gnn"
        self.gnn_data_dir.mkdir(parents=True, exist_ok=True)

        # Entity and relation mappings
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}

        # Node features
        self.entity_features: Optional[torch.Tensor] = None
        self.entity_types: Optional[torch.Tensor] = None

        # Edge data
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr: Optional[torch.Tensor] = None
        self.edge_types: Optional[torch.Tensor] = None

        logger.info("MaintIEGNNDataProcessor initialized")

    def prepare_gnn_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """Prepare complete GNN dataset from MaintIE data"""

        cache_path = self.gnn_data_dir / "gnn_dataset.pkl"

        if use_cache and cache_path.exists():
            logger.info("Loading GNN data from cache...")
            return self._load_cached_data(cache_path)

        logger.info("Preparing GNN data from MaintIE annotations...")

        # Step 1: Build entity and relation mappings
        self._build_mappings()

        # Step 2: Create node features from MaintIE entity types
        self._create_node_features()

        # Step 3: Create edge indices and features from MaintIE relations
        self._create_edge_data()

        # Step 4: Create PyTorch Geometric data object
        gnn_data = self._create_torch_geometric_data()

        # Step 5: Create training/validation splits
        train_data, val_data, test_data = self._create_data_splits(gnn_data)

        # Step 6: Prepare task-specific labels
        node_labels, edge_labels = self._create_task_labels()

        dataset = {
            'full_data': gnn_data,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'node_labels': node_labels,
            'edge_labels': edge_labels,
            'entity_to_idx': self.entity_to_idx,
            'idx_to_entity': self.idx_to_entity,
            'relation_to_idx': self.relation_to_idx,
            'idx_to_relation': self.idx_to_relation,
            'stats': self._get_dataset_stats()
        }

        # Cache the results
        self._save_cached_data(dataset, cache_path)

        logger.info(f"GNN dataset prepared: {dataset['stats']}")
        return dataset

    def _build_mappings(self):
        """Build entity and relation index mappings from your existing data"""

        # Build entity mappings from data_transformer.entities
        entity_idx = 0
        if hasattr(self.data_transformer, 'entities'):
            for entity_id, entity in self.data_transformer.entities.items():
                if entity_id not in self.entity_to_idx:
                    self.entity_to_idx[entity_id] = entity_idx
                    self.idx_to_entity[entity_idx] = entity_id
                    entity_idx += 1

        # Build relation mappings from data_transformer.relations
        relation_types = set()
        if hasattr(self.data_transformer, 'relations'):
            for relation in self.data_transformer.relations:
                relation_types.add(relation.relation_type.value)

        for idx, rel_type in enumerate(sorted(relation_types)):
            self.relation_to_idx[rel_type] = idx
            self.idx_to_relation[idx] = rel_type

        logger.info(f"Created mappings: {len(self.entity_to_idx)} entities, {len(self.relation_to_idx)} relation types")

    def _create_node_features(self):
        """Create node features from MaintIE entity types and text"""

        num_entities = len(self.entity_to_idx)

        # Create entity type features (one-hot encoding)
        entity_type_to_idx = self._get_entity_type_mapping()
        num_entity_types = len(entity_type_to_idx)

        # Initialize feature matrix
        feature_dim = num_entity_types + 10  # type features + additional features
        self.entity_features = torch.zeros(num_entities, feature_dim)
        self.entity_types = torch.zeros(num_entities, dtype=torch.long)

        # Fill features for each entity
        for entity_id, entity_idx in self.entity_to_idx.items():
            if hasattr(self.data_transformer, 'entities') and entity_id in self.data_transformer.entities:
                entity = self.data_transformer.entities[entity_id]

                # Entity type one-hot encoding
                entity_type_str = entity.entity_type.value
                if entity_type_str in entity_type_to_idx:
                    type_idx = entity_type_to_idx[entity_type_str]
                    self.entity_features[entity_idx, type_idx] = 1.0
                    self.entity_types[entity_idx] = type_idx

                # Additional features
                # Feature: Text length (normalized)
                text_length = len(entity.text) if entity.text else 0
                self.entity_features[entity_idx, num_entity_types] = min(text_length / 50.0, 1.0)

                # Feature: Confidence score
                self.entity_features[entity_idx, num_entity_types + 1] = entity.confidence

                # Feature: Is equipment (from entity type)
                if 'PhysicalObject' in entity_type_str:
                    self.entity_features[entity_idx, num_entity_types + 2] = 1.0

                # Feature: Is activity (from entity type)
                if 'Activity' in entity_type_str:
                    self.entity_features[entity_idx, num_entity_types + 3] = 1.0

                # Feature: Is failure/problem (from entity type)
                if 'UndesirableState' in entity_type_str or 'UndesirableProcess' in entity_type_str:
                    self.entity_features[entity_idx, num_entity_types + 4] = 1.0

        logger.info(f"Created node features: {self.entity_features.shape}")

    def _create_edge_data(self):
        """Create edge indices and features from MaintIE relations"""

        # Collect all edges from relations
        edge_list = []
        edge_types_list = []
        edge_features_list = []

        if hasattr(self.data_transformer, 'relations'):
            for relation in self.data_transformer.relations:
                source_id = relation.source_entity
                target_id = relation.target_entity

                # Check if both entities exist in our mapping
                if source_id in self.entity_to_idx and target_id in self.entity_to_idx:
                    source_idx = self.entity_to_idx[source_id]
                    target_idx = self.entity_to_idx[target_id]
                    relation_type_idx = self.relation_to_idx[relation.relation_type.value]

                    # Add edge
                    edge_list.append([source_idx, target_idx])
                    edge_types_list.append(relation_type_idx)

                    # Edge features: [confidence, relation_type_one_hot]
                    edge_feature = [relation.confidence]
                    # Add one-hot encoding for relation type
                    relation_one_hot = [0.0] * len(self.relation_to_idx)
                    relation_one_hot[relation_type_idx] = 1.0
                    edge_feature.extend(relation_one_hot)

                    edge_features_list.append(edge_feature)

        # Convert to tensors
        if edge_list:
            self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            self.edge_types = torch.tensor(edge_types_list, dtype=torch.long)
            self.edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
        else:
            # Empty graph fallback
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_types = torch.empty(0, dtype=torch.long)
            self.edge_attr = torch.empty((0, 1 + len(self.relation_to_idx)), dtype=torch.float)

        logger.info(f"Created edge data: {self.edge_index.shape[1]} edges")

    def _create_torch_geometric_data(self) -> Optional[Data]:
        """Create PyTorch Geometric Data object"""

        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available, returning None")
            return None

        # Create basic graph data
        data = Data(
            x=self.entity_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=len(self.entity_to_idx)
        )

        # Add additional attributes
        data.entity_types = self.entity_types
        data.edge_types = self.edge_types
        data.entity_to_idx = self.entity_to_idx
        data.relation_to_idx = self.relation_to_idx

        # Make undirected (optional, for some GNN models)
        transform = ToUndirected()
        data = transform(data)

        return data

    def _create_data_splits(self, gnn_data) -> Tuple[Any, Any, Any]:
        """Create train/validation/test splits"""

        if gnn_data is None:
            return None, None, None

        num_nodes = gnn_data.num_nodes

        # Simple random split (80/10/10)
        indices = torch.randperm(num_nodes)
        train_size = int(0.8 * num_nodes)
        val_size = int(0.1 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        # Create data copies with masks
        train_data = gnn_data.clone()
        train_data.train_mask = train_mask

        val_data = gnn_data.clone()
        val_data.val_mask = val_mask

        test_data = gnn_data.clone()
        test_data.test_mask = test_mask

        return train_data, val_data, test_data

    def _create_task_labels(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create labels for GNN tasks (node classification, link prediction)"""

        # Node classification labels: predict entity type
        node_labels = self.entity_types.clone() if self.entity_types is not None else None

        # Edge prediction labels: 1 for existing edges, 0 for non-existing
        num_nodes = len(self.entity_to_idx)
        edge_labels = torch.ones(self.edge_index.shape[1]) if self.edge_index is not None else None

        return node_labels, edge_labels

    def _get_entity_type_mapping(self) -> Dict[str, int]:
        """Get mapping from entity type strings to indices"""
        entity_types = set()

        if hasattr(self.data_transformer, 'entities'):
            for entity in self.data_transformer.entities.values():
                entity_types.add(entity.entity_type.value)

        return {etype: idx for idx, etype in enumerate(sorted(entity_types))}

    def _get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'num_entities': len(self.entity_to_idx),
            'num_relations': len(self.relation_to_idx),
            'num_edges': self.edge_index.shape[1] if self.edge_index is not None else 0,
            'feature_dim': self.entity_features.shape[1] if self.entity_features is not None else 0,
            'entity_types': len(set(self.entity_types.tolist())) if self.entity_types is not None else 0
        }

    def _save_cached_data(self, dataset: Dict[str, Any], cache_path: Path):
        """Save dataset to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            logger.info(f"GNN dataset cached to {cache_path}")
        except Exception as e:
            logger.error(f"Error caching dataset: {e}")

    def _load_cached_data(self, cache_path: Path) -> Dict[str, Any]:
        """Load dataset from cache"""
        try:
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f"GNN dataset loaded from cache")
            return dataset
        except Exception as e:
            logger.error(f"Error loading cached dataset: {e}")
            return self.prepare_gnn_data(use_cache=False)