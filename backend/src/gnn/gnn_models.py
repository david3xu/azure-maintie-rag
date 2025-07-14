"""
GNN Models for MaintIE Query Understanding
Professional implementation with fallbacks and integration points
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class MaintenanceGNNModel(nn.Module):
    """
    GNN Model for Maintenance Domain Understanding

    Tasks:
    1. Entity classification (better than rule-based)
    2. Query expansion (find related concepts)
    3. Domain understanding (maintenance context)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.input_dim = config.get('input_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.output_dim = config.get('output_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.num_entity_types = config.get('num_entity_types', 50)
        self.dropout = config.get('dropout', 0.2)
        self.gnn_type = config.get('gnn_type', 'GraphSAGE')  # GraphSAGE, GCN, GAT

        # Build GNN layers
        self._build_gnn_layers()

        # Task-specific heads
        self.entity_classifier = nn.Linear(self.hidden_dim, self.num_entity_types)
        self.query_expander = nn.Linear(self.hidden_dim, self.output_dim)
        self.domain_context = nn.Linear(self.hidden_dim, 32)  # Domain embedding

        logger.info(f"MaintenanceGNNModel initialized: {self.gnn_type}, {self.num_layers} layers")

    def _build_gnn_layers(self):
        """Build GNN layers based on configuration"""

        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available, using linear layers as fallback")
            self.gnn_layers = nn.ModuleList([
                nn.Linear(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim)
                for i in range(self.num_layers)
            ])
            return

        self.gnn_layers = nn.ModuleList()

        for i in range(self.num_layers):
            input_size = self.input_dim if i == 0 else self.hidden_dim

            if self.gnn_type == 'GraphSAGE':
                layer = SAGEConv(input_size, self.hidden_dim)
            elif self.gnn_type == 'GCN':
                layer = GCNConv(input_size, self.hidden_dim)
            elif self.gnn_type == 'GAT':
                layer = GATConv(input_size, self.hidden_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")

            self.gnn_layers.append(layer)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through GNN"""

        # GNN feature extraction
        h = x
        for i, layer in enumerate(self.gnn_layers):
            if TORCH_GEOMETRIC_AVAILABLE and hasattr(layer, 'forward'):
                h = layer(h, edge_index)
            else:
                # Fallback: simple linear transformation
                h = layer(h)

            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Task-specific outputs
        outputs = {
            'node_embeddings': h,
            'entity_logits': self.entity_classifier(h),
            'query_embeddings': self.query_expander(h),
            'domain_context': self.domain_context(h)
        }

        # Graph-level aggregation if batch is provided
        if batch is not None and TORCH_GEOMETRIC_AVAILABLE:
            outputs['graph_embedding'] = global_mean_pool(h, batch)

        return outputs

    def predict_entity_types(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict entity types for given nodes"""
        outputs = self.forward(x, edge_index)
        return F.softmax(outputs['entity_logits'], dim=-1)

    def get_query_expansion(self, query_entities: List[int], x: torch.Tensor,
                           edge_index: torch.Tensor, top_k: int = 10) -> List[int]:
        """Get expanded entities for query entities using GNN"""

        outputs = self.forward(x, edge_index)
        node_embeddings = outputs['node_embeddings']

        # Get embeddings for query entities
        query_embeddings = node_embeddings[query_entities]  # [num_query_entities, hidden_dim]

        # Compute similarity to all other entities
        all_embeddings = node_embeddings  # [num_nodes, hidden_dim]

        # Cosine similarity
        query_mean = query_embeddings.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        similarities = F.cosine_similarity(query_mean, all_embeddings, dim=-1)  # [num_nodes]

        # Get top-k most similar entities (excluding query entities)
        _, top_indices = torch.topk(similarities, k=top_k + len(query_entities))

        # Filter out original query entities
        expanded_entities = []
        for idx in top_indices.tolist():
            if idx not in query_entities:
                expanded_entities.append(idx)
            if len(expanded_entities) >= top_k:
                break

        return expanded_entities

class GNNTrainer:
    """Professional GNN training with your existing architecture patterns"""

    def __init__(self, model: MaintenanceGNNModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics
        self.train_losses = []
        self.val_accuracies = []

        logger.info(f"GNNTrainer initialized on device: {device}")

    def train_epoch(self, train_data, train_labels: torch.Tensor) -> float:
        """Train one epoch"""
        self.model.train()

        if train_data is None:
            logger.warning("No training data available")
            return 0.0

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(train_data.x, train_data.edge_index)
        entity_logits = outputs['entity_logits']

        # Compute loss (only on training nodes if mask exists)
        if hasattr(train_data, 'train_mask'):
            loss = self.criterion(entity_logits[train_data.train_mask],
                                train_labels[train_data.train_mask])
        else:
            loss = self.criterion(entity_logits, train_labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, val_data, val_labels: torch.Tensor) -> float:
        """Evaluate model"""
        self.model.eval()

        if val_data is None:
            return 0.0

        with torch.no_grad():
            outputs = self.model(val_data.x, val_data.edge_index)
            entity_logits = outputs['entity_logits']

            # Compute accuracy
            if hasattr(val_data, 'val_mask'):
                predictions = entity_logits[val_data.val_mask].argmax(dim=-1)
                accuracy = (predictions == val_labels[val_data.val_mask]).float().mean()
            else:
                predictions = entity_logits.argmax(dim=-1)
                accuracy = (predictions == val_labels).float().mean()

        return accuracy.item()

    def train(self, train_data, val_data, train_labels: torch.Tensor,
              val_labels: torch.Tensor, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Full training loop"""

        logger.info(f"Starting GNN training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_data, train_labels)
            self.train_losses.append(train_loss)

            # Validate
            if epoch % 10 == 0:
                val_accuracy = self.evaluate(val_data, val_labels)
                self.val_accuracies.append(val_accuracy)

                logger.info(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Val Acc={val_accuracy:.4f}")

        logger.info("GNN training completed")

        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }

    def save_model(self, path: Path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")