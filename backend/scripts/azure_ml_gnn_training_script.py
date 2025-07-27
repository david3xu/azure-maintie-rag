#!/usr/bin/env python3
"""
Azure ML GNN Training Script
Runs in Azure ML compute - no local dependencies
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import mlflow
import mlflow.pytorch
from pathlib import Path


class AzureMLGraphAttentionNetwork(torch.nn.Module):
    """Graph Attention Network for Azure ML training"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GAT layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--output_path", type=str, help="Path to save model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()
    
    # Start MLflow tracking
    mlflow.start_run()
    
    print("🚀 Azure ML GNN Training Started")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    
    # Load data
    data_files = list(Path(args.data_path).glob("gnn_training_data_full_*.npz"))
    metadata_files = list(Path(args.data_path).glob("gnn_metadata_full_*.json"))
    
    if not data_files or not metadata_files:
        raise FileNotFoundError("No training data found")
    
    latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
    latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    # Load training data
    training_data = np.load(latest_data_file)
    with open(latest_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Prepare PyTorch Geometric data
    node_features = torch.FloatTensor(training_data['node_features'])
    edge_index = torch.LongTensor(training_data['edge_index'])
    node_labels = torch.LongTensor(training_data['node_labels'])
    
    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    
    print(f"📊 Data loaded:")
    print(f"   Nodes: {data.x.size(0)}")
    print(f"   Features: {data.x.size(1)}")
    print(f"   Edges: {data.edge_index.size(1)}")
    print(f"   Classes: {metadata['num_classes']}")
    
    # Create data splits
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create model
    model = AzureMLGraphAttentionNetwork(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=metadata['num_classes']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.NLLLoss()
    
    print(f"🧠 Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[train_mask], data.y[train_mask])
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[val_mask], data.y[val_mask])
            
            train_pred = out[train_mask].argmax(dim=1)
            val_pred = val_out[val_mask].argmax(dim=1)
            
            train_accuracy = (train_pred == data.y[train_mask]).float().mean().item()
            val_accuracy = (val_pred == data.y[val_mask]).float().mean().item()
        
        # Logging
        mlflow.log_metrics({
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        }, step=epoch)
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            model_path = Path(args.output_path) / "best_model.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss={train_loss.item():.4f}, Val Acc={val_accuracy:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[test_mask].argmax(dim=1)
        test_accuracy = (test_pred == data.y[test_mask]).float().mean().item()
    
    print(f"✅ Training completed:")
    print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"   Final test accuracy: {test_accuracy:.4f}")
    
    # Log final results
    mlflow.log_metrics({
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy
    })
    
    # Register model
    mlflow.pytorch.log_model(model, "gnn_model")
    
    mlflow.end_run()
    
    return test_accuracy


if __name__ == "__main__":
    main()
