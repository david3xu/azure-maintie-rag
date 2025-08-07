"""Universal GNN model architecture for Azure Universal RAG system."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from typing import Dict, Any, Optional, List
import logging

from infrastructure.constants import MLModelConstants

logger = logging.getLogger(__name__)


class UniversalGNN(nn.Module):
    """Universal Graph Neural Network for any domain knowledge graph"""

    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_dim: int = MLModelConstants.DEFAULT_HIDDEN_DIM,
        num_layers: int = MLModelConstants.DEFAULT_NUM_LAYERS,
        dropout: float = MLModelConstants.DEFAULT_DROPOUT_RATE,
        conv_type: str = MLModelConstants.DEFAULT_CONV_TYPE,
    ):
        """
        Initialize Universal GNN

        Args:
            num_node_features: Number of input node features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution ("gcn", "gat", "sage")
        """
        super(UniversalGNN, self).__init__()

        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()

        # First layer
        if conv_type == "gcn":
            self.conv_layers.append(GCNConv(num_node_features, hidden_dim))
        elif conv_type == "gat":
            self.conv_layers.append(GATConv(num_node_features, hidden_dim))
        elif conv_type == "sage":
            self.conv_layers.append(SAGEConv(num_node_features, hidden_dim))
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        # Hidden layers
        for _ in range(num_layers - 1):
            if conv_type == "gcn":
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == "gat":
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim))
            elif conv_type == "sage":
                self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Output classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        logger.info(
            f"UniversalGNN initialized: {conv_type}, {num_layers} layers, "
            f"hidden_dim={hidden_dim}, dropout={dropout}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the GNN

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            Graph-level predictions [batch_size, num_classes]
        """
        # Graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:  # Not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if batch is not None:
            # Use batch information for pooling
            x = self._global_pool(x, batch)
        else:
            # Simple mean pooling for single graph
            x = torch.mean(x, dim=0, keepdim=True)

        # Classification
        x = self.classifier(x)

        return x

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Global pooling using batch information"""
        # Group by batch
        unique_batches = torch.unique(batch)
        pooled_features = []

        for batch_idx in unique_batches:
            mask = batch == batch_idx
            batch_features = x[mask]
            # Mean pooling for each batch
            batch_pooled = torch.mean(batch_features, dim=0)
            pooled_features.append(batch_pooled)

        return torch.stack(pooled_features)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings (without classification layer)"""
        # Graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:  # Not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def predict_node_classes(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Predict node-level classes"""
        embeddings = self.get_embeddings(x, edge_index)
        return self.classifier(embeddings)


class UniversalGNNConfig:
    """Configuration for Universal GNN model"""

    def __init__(
        self,
        hidden_dim: int = MLModelConstants.DEFAULT_HIDDEN_DIM,
        num_layers: int = MLModelConstants.DEFAULT_NUM_LAYERS,
        dropout: float = MLModelConstants.DEFAULT_DROPOUT_RATE,
        conv_type: str = MLModelConstants.DEFAULT_CONV_TYPE,
        learning_rate: float = MLModelConstants.DEFAULT_LEARNING_RATE,
        weight_decay: float = MLModelConstants.DEFAULT_WEIGHT_DECAY,
    ):
        """
        Initialize GNN configuration

        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "conv_type": self.conv_type,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UniversalGNNConfig":
        """Create config from dictionary"""
        return cls(
            hidden_dim=config_dict.get(
                "hidden_dim", MLModelConstants.DEFAULT_HIDDEN_DIM
            ),
            num_layers=config_dict.get(
                "num_layers", MLModelConstants.DEFAULT_NUM_LAYERS
            ),
            dropout=config_dict.get("dropout", MLModelConstants.DEFAULT_DROPOUT_RATE),
            conv_type=config_dict.get("conv_type", MLModelConstants.DEFAULT_CONV_TYPE),
            learning_rate=config_dict.get(
                "learning_rate", MLModelConstants.DEFAULT_LEARNING_RATE
            ),
            weight_decay=config_dict.get(
                "weight_decay", MLModelConstants.DEFAULT_WEIGHT_DECAY
            ),
        )


def create_gnn_model(
    num_node_features: int, num_classes: int, config: UniversalGNNConfig
) -> UniversalGNN:
    """
    Factory function to create Universal GNN model

    Args:
        num_node_features: Number of input node features
        num_classes: Number of output classes
        config: GNN configuration

    Returns:
        Configured Universal GNN model
    """
    return UniversalGNN(
        num_node_features=num_node_features,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        conv_type=config.conv_type,
    )
