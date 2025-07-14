"""
GNN-Based Query Expansion that integrates with your existing query_analyzer.py
Professional implementation with fallbacks
"""

import torch
import logging
from typing import List, Dict, Optional, Any, Set
from pathlib import Path

from src.gnn.gnn_models import MaintenanceGNNModel
from src.gnn.data_preparation import MaintIEGNNDataProcessor
from src.knowledge.data_transformer import MaintIEDataTransformer

logger = logging.getLogger(__name__)

class GNNQueryExpander:
    """
    GNN-based query expansion for maintenance domain
    Integrates with your existing MaintenanceQueryAnalyzer
    """

    def __init__(self, data_transformer: MaintIEDataTransformer,
                 model_path: Optional[Path] = None):
        """Initialize GNN query expander"""

        self.data_transformer = data_transformer
        self.model = None
        self.gnn_data = None
        self.entity_to_idx = {}
        self.idx_to_entity = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.enabled = False

        # Try to initialize GNN components
        self._initialize_gnn(model_path)

    def _initialize_gnn(self, model_path: Optional[Path]):
        """Initialize GNN model and data"""
        try:
            # Load GNN data
            gnn_processor = MaintIEGNNDataProcessor(self.data_transformer)
            self.gnn_data = gnn_processor.prepare_gnn_data(use_cache=True)

            if self.gnn_data['full_data'] is None:
                logger.warning("GNN data not available, falling back to rule-based expansion")
                return

            self.entity_to_idx = self.gnn_data['entity_to_idx']
            self.idx_to_entity = self.gnn_data['idx_to_entity']

            # Initialize or load model
            if model_path and model_path.exists():
                self._load_trained_model(model_path)
            else:
                self._create_and_train_model()

            self.enabled = True
            logger.info("GNN query expander initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize GNN query expander: {e}")
            self.enabled = False

    def _create_and_train_model(self):
        """Create and train GNN model"""

        if self.gnn_data['full_data'] is None:
            return

        # Model configuration
        config = {
            'input_dim': self.gnn_data['full_data'].x.shape[1],
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 3,
            'num_entity_types': len(set(self.gnn_data['node_labels'].tolist())),
            'gnn_type': 'GraphSAGE',
            'dropout': 0.2
        }

        # Create model
        self.model = MaintenanceGNNModel(config)

        # Quick training (for demo - in production, use proper training)
        from src.gnn.gnn_models import GNNTrainer
        trainer = GNNTrainer(self.model, self.device)

        # Move data to device
        train_data = self.gnn_data['train_data']
        val_data = self.gnn_data['val_data']
        train_labels = self.gnn_data['node_labels']
        val_labels = self.gnn_data['node_labels']

        if train_data is not None:
            # Short training for demonstration
            trainer.train(train_data, val_data, train_labels, val_labels, num_epochs=50)

            # Save model
            model_dir = Path("data/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(model_dir / "maintenance_gnn.pt")

        logger.info("GNN model training completed")

    def _load_trained_model(self, model_path: Path):
        """Load pre-trained GNN model"""

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']

        self.model = MaintenanceGNNModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded pre-trained GNN model from {model_path}")

    def expand_query_entities(self, entities: List[str], max_expansions: int = 10) -> List[str]:
        """
        Expand query entities using GNN

        This method integrates with your existing query_analyzer.py
        """

        if not self.enabled or not self.model or self.gnn_data['full_data'] is None:
            logger.debug("GNN not available, falling back to rule-based expansion")
            return self._fallback_expansion(entities)

        try:
            # Convert entity texts to indices
            query_indices = []
            for entity in entities:
                # Find entity in our mapping
                entity_id = self._find_entity_id(entity)
                if entity_id and entity_id in self.entity_to_idx:
                    query_indices.append(self.entity_to_idx[entity_id])

            if not query_indices:
                logger.debug("No entities found in GNN graph, using fallback")
                return self._fallback_expansion(entities)

            # Get GNN-based expansion
            self.model.eval()
            with torch.no_grad():
                data = self.gnn_data['full_data']
                expanded_indices = self.model.get_query_expansion(
                    query_indices, data.x, data.edge_index, top_k=max_expansions
                )

            # Convert indices back to entity texts
            expanded_entities = []
            for idx in expanded_indices:
                if idx in self.idx_to_entity:
                    entity_id = self.idx_to_entity[idx]
                    if entity_id in self.data_transformer.entities:
                        entity_text = self.data_transformer.entities[entity_id].text
                        expanded_entities.append(entity_text)

            logger.debug(f"GNN expansion: {len(entities)} → {len(expanded_entities)} entities")
            return expanded_entities

        except Exception as e:
            logger.warning(f"GNN expansion failed: {e}, using fallback")
            return self._fallback_expansion(entities)

    def _find_entity_id(self, entity_text: str) -> Optional[str]:
        """Find entity ID for given text"""
        if hasattr(self.data_transformer, 'entities'):
            for entity_id, entity in self.data_transformer.entities.items():
                if entity.text.lower() == entity_text.lower():
                    return entity_id
        return None

    def _fallback_expansion(self, entities: List[str]) -> List[str]:
        """Fallback rule-based expansion when GNN not available"""

        # Simple fallback using existing knowledge graph
        expanded = set(entities)

        if (hasattr(self.data_transformer, 'knowledge_graph') and
            self.data_transformer.knowledge_graph):

            for entity in entities:
                entity_id = self._find_entity_id(entity)
                if entity_id:
                    # Get neighbors from NetworkX graph
                    try:
                        neighbors = list(self.data_transformer.knowledge_graph.neighbors(entity_id))
                        for neighbor_id in neighbors[:5]:  # Limit to 5 neighbors
                            if neighbor_id in self.data_transformer.entities:
                                neighbor_text = self.data_transformer.entities[neighbor_id].text
                                expanded.add(neighbor_text)
                    except:
                        continue

        return list(expanded)

    def get_domain_context(self, entities: List[str]) -> Dict[str, Any]:
        """Get domain context for entities using GNN"""

        if not self.enabled or not self.model:
            return {}

        try:
            # Convert entities to indices
            entity_indices = []
            for entity in entities:
                entity_id = self._find_entity_id(entity)
                if entity_id and entity_id in self.entity_to_idx:
                    entity_indices.append(self.entity_to_idx[entity_id])

            if not entity_indices:
                return {}

            # Get domain context from GNN
            self.model.eval()
            with torch.no_grad():
                data = self.gnn_data['full_data']
                outputs = self.model(data.x, data.edge_index)

                # Get context for query entities
                entity_contexts = outputs['domain_context'][entity_indices]
                context_mean = entity_contexts.mean(dim=0)

                # Convert to interpretable format
                context = {
                    'equipment_focus': float(context_mean[0]),
                    'maintenance_focus': float(context_mean[1]),
                    'safety_focus': float(context_mean[2]),
                    'complexity_level': float(context_mean[3])
                }

                return context

        except Exception as e:
            logger.warning(f"Failed to get GNN domain context: {e}")
            return {}