#!/usr/bin/env python3
"""
Real Graph Attention Network Model
Matches the architecture of the trained model for loading weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class RealGraphAttentionNetwork(nn.Module):
    """Real Graph Attention Network that matches the trained model architecture"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 heads: int = 8,
                 dropout: float = 0.2):
        """
        Initialize Real Graph Attention Network

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            output_dim: Number of output classes
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)

        # Hidden layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Output layer
        self.conv_out = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)

        # Additional layers
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim * heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT network

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Input layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Hidden layers
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)
            # Residual connection if dimensions match
            if residual.size() == x.size():
                x = x + residual

        # Output layer
        x = self.conv_out(x, edge_index)

        return F.log_softmax(x, dim=1)

    def predict_node_classes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict node classes

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Class predictions [num_nodes, output_dim]
        """
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return torch.exp(logits)  # Convert log_softmax back to probabilities

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings from the last hidden layer

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, hidden_dim * heads]
        """
        # Input layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Hidden layers
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)
            # Residual connection if dimensions match
            if residual.size() == x.size():
                x = x + residual

        return x


def load_trained_gnn_model(model_info_path: str, weights_path: str) -> RealGraphAttentionNetwork:
    """
    Load the trained GNN model

    Args:
        model_info_path: Path to model info JSON file
        weights_path: Path to model weights PT file

    Returns:
        Loaded GNN model
    """
    import json

    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    # Create model with same architecture
    model = RealGraphAttentionNetwork(
        input_dim=model_info['model_architecture']['input_dim'],
        hidden_dim=model_info['model_architecture']['hidden_dim'],
        output_dim=model_info['model_architecture']['output_dim'],
        num_layers=model_info['model_architecture']['num_layers'],
        heads=model_info['model_architecture']['attention_heads'],
        dropout=model_info['model_architecture']['dropout']
    )

    # Load trained weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    return model


def test_model_loading():
    """Test loading the trained model"""
    print("🧪 Testing GNN Model Loading")
    print("=" * 40)

    model_info_path = "data/gnn_models/real_gnn_model_full_20250727_045556.json"
    weights_path = "data/gnn_models/real_gnn_weights_full_20250727_045556.pt"

    try:
        # Load model
        model = load_trained_gnn_model(model_info_path, weights_path)
        print("✅ Model loaded successfully!")

        # Test with dummy data
        num_nodes = 10
        num_edges = 20
        input_dim = 1540

        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Test forward pass
        with torch.no_grad():
            output = model(x, edge_index)
            embeddings = model.get_embeddings(x, edge_index)
            predictions = model.predict_node_classes(x, edge_index)

        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Embeddings shape: {embeddings.shape}")
        print(f"   - Predictions shape: {predictions.shape}")

        return model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_model_loading()
