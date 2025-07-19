"""
Universal GNN Data Processor
Replaces MaintIEGNNDataProcessor with domain-agnostic GNN data preparation
Works with AzureOpenAITextProcessor and universal models for any domain
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    torch = None
    Data = None

from ..azure_openai.text_processor import AzureOpenAITextProcessor
from ..models.azure_rag_data_models import UniversalEntity, UniversalRelation
from ...config.settings import settings

logger = logging.getLogger(__name__)


class AzureMLGNNProcessor:
    """Convert universal text knowledge to GNN-ready format for any domain"""

    def __init__(self, text_processor: AzureOpenAITextProcessor, domain: str = "general"):
        """Initialize with universal text processor"""
        self.text_processor = text_processor
        self.domain = domain
        self.gnn_data_dir = settings.processed_data_dir / "gnn" / domain
        self.gnn_data_dir.mkdir(parents=True, exist_ok=True)

        # Universal entity and relation mappings (no domain assumptions)
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}
        self.entity_type_to_idx: Dict[str, int] = {}
        self.relation_type_to_idx: Dict[str, int] = {}

        # Universal node features (dynamic based on discovered types)
        self.entity_features: Optional[torch.Tensor] = None
        self.entity_types: Optional[torch.Tensor] = None

        # Universal edge data (dynamic based on discovered relations)
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr: Optional[torch.Tensor] = None
        self.edge_types: Optional[torch.Tensor] = None

        # Check PyTorch Geometric availability
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available, GNN processing will be limited")

        logger.info(f"AzureMLGNNProcessor initialized for domain: {domain}")

    def prepare_universal_gnn_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """Prepare complete GNN dataset from universal text knowledge"""

        cache_path = self.gnn_data_dir / "universal_gnn_dataset.pkl"

        if use_cache and cache_path.exists():
            logger.info("Loading universal GNN data from cache...")
            return self._load_cached_data(cache_path)

        logger.info(f"Preparing universal GNN data for domain: {self.domain}")

        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("PyTorch Geometric not available for GNN processing")
            return {"error": "PyTorch Geometric not installed"}

        try:
            # Step 1: Build universal entity and relation mappings
            self._build_universal_mappings()

            # Step 2: Create universal node features from discovered entity types
            self._create_universal_node_features()

            # Step 3: Create universal edge data from discovered relations
            self._create_universal_edge_data()

            # Step 4: Create PyTorch Geometric data object
            gnn_data = self._create_universal_torch_geometric_data()

            # Step 5: Create training/validation splits
            train_data, val_data, test_data = self._create_universal_data_splits(gnn_data)

            # Step 6: Prepare universal task-specific labels
            node_labels, edge_labels = self._create_universal_task_labels()

            dataset = {
                'domain': self.domain,
                'processing_method': 'universal_text_based',
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
                'entity_types': list(self.entity_type_to_idx.keys()),
                'relation_types': list(self.relation_type_to_idx.keys()),
                'stats': self._get_universal_dataset_stats()
            }

            # Cache the results
            self._save_cached_data(dataset, cache_path)

            logger.info(f"Universal GNN dataset prepared for {self.domain}: {dataset['stats']}")
            return dataset

        except Exception as e:
            logger.error(f"Universal GNN data preparation failed: {e}")
            return {"error": str(e), "domain": self.domain}

    def _build_universal_mappings(self):
        """Build entity and relation mappings from universal text processor"""

        # Build entity mappings from universal entities
        entity_idx = 0
        if hasattr(self.text_processor, 'entities') and self.text_processor.entities:
            for entity_id, entity in self.text_processor.entities.items():
                self.entity_to_idx[entity_id] = entity_idx
                self.idx_to_entity[entity_idx] = entity_id

                # Map entity types
                entity_type = entity.entity_type
                if entity_type not in self.entity_type_to_idx:
                    self.entity_type_to_idx[entity_type] = len(self.entity_type_to_idx)

                entity_idx += 1

        # Build relation mappings from universal relations
        relation_idx = 0
        if hasattr(self.text_processor, 'relations') and self.text_processor.relations:
            for relation in self.text_processor.relations:
                relation_id = relation.relation_id
                self.relation_to_idx[relation_id] = relation_idx
                self.idx_to_relation[relation_idx] = relation_id

                # Map relation types
                relation_type = relation.relation_type
                if relation_type not in self.relation_type_to_idx:
                    self.relation_type_to_idx[relation_type] = len(self.relation_type_to_idx)

                relation_idx += 1

        logger.info(f"Universal mappings built: {len(self.entity_to_idx)} entities, {len(self.relation_to_idx)} relations")

    def _create_universal_node_features(self):
        """Create universal node features from discovered entity types"""

        if not self.entity_to_idx:
            logger.warning("No entities found for node feature creation")
            return

        try:
            num_entities = len(self.entity_to_idx)
            num_entity_types = len(self.entity_type_to_idx)

            # Create simple one-hot encoding for entity types
            # In production, this could use more sophisticated embeddings
            feature_dim = max(num_entity_types, 8)  # Minimum feature dimension

            # Initialize features
            self.entity_features = torch.zeros(num_entities, feature_dim)
            self.entity_types = torch.zeros(num_entities, dtype=torch.long)

            # Fill features based on entity types
            for entity_id, entity_idx in self.entity_to_idx.items():
                if entity_id in self.text_processor.entities:
                    entity = self.text_processor.entities[entity_id]
                    entity_type = entity.entity_type

                    if entity_type in self.entity_type_to_idx:
                        type_idx = self.entity_type_to_idx[entity_type]
                        self.entity_types[entity_idx] = type_idx

                        # One-hot encoding for entity type
                        if type_idx < feature_dim:
                            self.entity_features[entity_idx, type_idx] = 1.0

                        # Add confidence as additional feature
                        if feature_dim > num_entity_types:
                            confidence_idx = min(num_entity_types, feature_dim - 1)
                            self.entity_features[entity_idx, confidence_idx] = entity.confidence

            logger.info(f"Universal node features created: {self.entity_features.shape}")

        except Exception as e:
            logger.error(f"Universal node feature creation failed: {e}")
            # Fallback to simple features
            num_entities = len(self.entity_to_idx)
            self.entity_features = torch.randn(num_entities, 8)
            self.entity_types = torch.zeros(num_entities, dtype=torch.long)

    def _create_universal_edge_data(self):
        """Create universal edge data from discovered relations"""

        if not self.relation_to_idx:
            logger.warning("No relations found for edge data creation")
            return

        try:
            # Build edge index and attributes
            edge_list = []
            edge_attrs = []
            edge_types = []

            for relation in self.text_processor.relations:
                source_id = relation.source_entity_id
                target_id = relation.target_entity_id

                # Check if both entities exist in our mapping
                if source_id in self.entity_to_idx and target_id in self.entity_to_idx:
                    source_idx = self.entity_to_idx[source_id]
                    target_idx = self.entity_to_idx[target_id]

                    # Add edge
                    edge_list.append([source_idx, target_idx])

                    # Add edge attributes (confidence, relation type)
                    edge_attrs.append([relation.confidence])

                    # Add edge type
                    relation_type = relation.relation_type
                    if relation_type in self.relation_type_to_idx:
                        edge_types.append(self.relation_type_to_idx[relation_type])
                    else:
                        edge_types.append(0)  # Default type

            if edge_list:
                # Convert to tensors
                self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                self.edge_types = torch.tensor(edge_types, dtype=torch.long)

                logger.info(f"Universal edge data created: {self.edge_index.shape[1]} edges")
            else:
                logger.warning("No valid edges found")
                # Create empty tensors
                self.edge_index = torch.zeros((2, 0), dtype=torch.long)
                self.edge_attr = torch.zeros((0, 1), dtype=torch.float)
                self.edge_types = torch.zeros(0, dtype=torch.long)

        except Exception as e:
            logger.error(f"Universal edge data creation failed: {e}")
            # Fallback to empty edges
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_attr = torch.zeros((0, 1), dtype=torch.float)
            self.edge_types = torch.zeros(0, dtype=torch.long)

    def _create_universal_torch_geometric_data(self) -> Optional[Data]:
        """Create PyTorch Geometric data object from universal features"""

        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("PyTorch Geometric not available")
            return None

        try:
            data = Data(
                x=self.entity_features,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                edge_type=self.edge_types,
                node_type=self.entity_types
            )

            logger.info(f"Universal PyTorch Geometric data created: {data}")
            return data

        except Exception as e:
            logger.error(f"Universal PyTorch Geometric data creation failed: {e}")
            return None

    def _create_universal_data_splits(self, gnn_data) -> Tuple[Any, Any, Any]:
        """Create universal train/validation/test splits"""

        if gnn_data is None:
            return None, None, None

        try:
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

        except Exception as e:
            logger.error(f"Universal data split creation failed: {e}")
            return None, None, None

    def _create_universal_task_labels(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create universal labels for GNN tasks"""

        try:
            # Node classification labels: predict entity type
            node_labels = self.entity_types.clone() if self.entity_types is not None else None

            # Edge prediction labels: 1 for existing edges, 0 for non-existing
            edge_labels = torch.ones(self.edge_index.shape[1]) if self.edge_index is not None else None

            return node_labels, edge_labels

        except Exception as e:
            logger.error(f"Universal task label creation failed: {e}")
            return None, None

    def _get_universal_dataset_stats(self) -> Dict[str, Any]:
        """Get universal dataset statistics"""
        return {
            "domain": self.domain,
            "num_entities": len(self.entity_to_idx),
            "num_relations": len(self.relation_to_idx),
            "num_entity_types": len(self.entity_type_to_idx),
            "num_relation_types": len(self.relation_type_to_idx),
            "num_edges": self.edge_index.shape[1] if self.edge_index is not None else 0,
            "feature_dimension": self.entity_features.shape[1] if self.entity_features is not None else 0,
            "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE
        }

    def _save_cached_data(self, dataset: Dict[str, Any], cache_path: Path):
        """Save universal dataset to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            logger.info(f"Universal GNN dataset cached at {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache universal GNN dataset: {e}")

    def _load_cached_data(self, cache_path: Path) -> Dict[str, Any]:
        """Load universal dataset from cache"""
        try:
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f"Universal GNN dataset loaded from cache: {cache_path}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load cached universal GNN dataset: {e}")
            return {"error": f"Cache loading failed: {e}"}


# Legacy compatibility alias
MaintIEGNNDataProcessor = AzureMLGNNProcessor


def create_universal_gnn_processor(text_processor: AzureOpenAITextProcessor,
                                 domain: str = "general") -> AzureMLGNNProcessor:
    """Factory function to create universal GNN data processor"""
    return AzureMLGNNProcessor(text_processor, domain)


if __name__ == "__main__":
    # Example usage
    from ..azure_openai.text_processor import AzureOpenAITextProcessor

    processor = AzureOpenAITextProcessor("maintenance")
    gnn_processor = AzureMLGNNProcessor(processor, "maintenance")

    # Process GNN data
    dataset = gnn_processor.prepare_universal_gnn_data()
    print(f"Universal GNN dataset: {dataset.get('stats', {})}")