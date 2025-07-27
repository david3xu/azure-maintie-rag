#!/usr/bin/env python3
"""
REAL Graph Neural Network Training with Azure ML
NO SIMULATION - Actual PyTorch Geometric training with real data
"""

import json
import numpy as np
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import time

sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.utils import train_test_split_edges
    print(f"✅ PyTorch Geometric {torch_geometric.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ PyTorch Geometric not installed: {e}")
    print("Install with: pip install torch-geometric")
    sys.exit(1)

from config.settings import settings


class RealGraphAttentionNetwork(torch.nn.Module):
    """Real Graph Attention Network - no simulation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, heads: int = 8, dropout: float = 0.2):
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


class RealGNNTrainer:
    """Real GNN Trainer using PyTorch Geometric - NO SIMULATION"""
    
    def __init__(self, use_partial: bool = False):
        self.use_partial = use_partial
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
        self.model_config = {
            "hidden_dim": 256,
            "num_layers": 3,
            "heads": 8,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "weight_decay": 5e-4,
            "epochs": 100,
            "patience": 10
        }
        
    def load_training_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load real GNN training data"""
        
        data_dir = Path(__file__).parent.parent / "data" / "gnn_training"
        
        # Find most recent training data
        data_pattern = "gnn_training_data_partial_*.npz" if self.use_partial else "gnn_training_data_full_*.npz"
        metadata_pattern = "gnn_metadata_partial_*.json" if self.use_partial else "gnn_metadata_full_*.json"
        
        data_files = list(data_dir.glob(data_pattern))
        metadata_files = list(data_dir.glob(metadata_pattern))
        
        if not data_files or not metadata_files:
            raise FileNotFoundError(f"No training data found. Run prepare_gnn_training_features.py first")
        
        # Get most recent files
        latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
        latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        print(f"📄 Loading training data:")
        print(f"   • Data: {latest_data_file.name}")
        print(f"   • Metadata: {latest_metadata_file.name}")
        
        # Load data
        data = np.load(latest_data_file)
        with open(latest_metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return dict(data), metadata
    
    def prepare_torch_geometric_data(self, training_data: Dict[str, Any], metadata: Dict[str, Any]) -> Data:
        """Convert numpy data to PyTorch Geometric Data object"""
        
        # Extract data
        node_features = torch.FloatTensor(training_data['node_features'])
        edge_index = torch.LongTensor(training_data['edge_index'])
        node_labels = torch.LongTensor(training_data['node_labels'])
        
        print(f"📊 Data conversion:")
        print(f"   • Nodes: {node_features.size(0)}")
        print(f"   • Features: {node_features.size(1)}")
        print(f"   • Edges: {edge_index.size(1)}")
        print(f"   • Classes: {metadata['num_classes']}")
        
        # Create PyTorch Geometric data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=node_labels
        )
        
        return data
    
    def create_data_splits(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/validation/test splits"""
        
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
        
        print(f"📊 Data splits created:")
        print(f"   • Training: {train_mask.sum().item()} nodes ({train_mask.sum().item() / num_nodes * 100:.1f}%)")
        print(f"   • Validation: {val_mask.sum().item()} nodes ({val_mask.sum().item() / num_nodes * 100:.1f}%)")
        print(f"   • Test: {test_mask.sum().item()} nodes ({test_mask.sum().item() / num_nodes * 100:.1f}%)")
        
        return train_mask, val_mask, test_mask
    
    def train_real_gnn(self, data: Data, train_mask: torch.Tensor, val_mask: torch.Tensor, 
                      test_mask: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """REAL GNN training using PyTorch Geometric"""
        
        print(f"🧠 Creating REAL GNN model architecture...")
        
        # Create model
        model = RealGraphAttentionNetwork(
            input_dim=data.x.size(1),
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=metadata['num_classes'],
            num_layers=self.model_config["num_layers"],
            heads=self.model_config["heads"],
            dropout=self.model_config["dropout"]
        ).to(self.device)
        
        # Move data to device
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.model_config["learning_rate"],
            weight_decay=self.model_config["weight_decay"]
        )
        criterion = torch.nn.NLLLoss()
        
        print(f"✅ Model architecture created:")
        print(f"   • Type: RealGraphAttentionNetwork")
        print(f"   • Input dim: {data.x.size(1)}")
        print(f"   • Hidden dim: {self.model_config['hidden_dim']}")
        print(f"   • Output classes: {metadata['num_classes']}")
        print(f"   • Layers: {self.model_config['num_layers']}")
        print(f"   • Attention heads: {self.model_config['heads']}")
        print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print(f"\n🚀 STARTING REAL GNN TRAINING")
        print("=" * 50)
        
        # Training loop
        model.train()
        training_history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.model_config["epochs"]):
            epoch_start = time.time()
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            train_loss = criterion(out[train_mask], data.y[train_mask])
            
            train_loss.backward()
            optimizer.step()
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask])
                
                # Calculate accuracies
                train_pred = out[train_mask].argmax(dim=1)
                val_pred = val_out[val_mask].argmax(dim=1)
                
                train_accuracy = (train_pred == data.y[train_mask]).float().mean().item()
                val_accuracy = (val_pred == data.y[val_mask]).float().mean().item()
            
            # Record metrics
            training_history["epochs"].append(epoch + 1)
            training_history["train_loss"].append(train_loss.item())
            training_history["val_loss"].append(val_loss.item())
            training_history["train_accuracy"].append(train_accuracy)
            training_history["val_accuracy"].append(val_accuracy)
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1:3d}: Train Loss={train_loss.item():.4f}, "
                      f"Val Acc={val_accuracy:.4f}, Time={epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= self.model_config["patience"]:
                print(f"   Early stopping at epoch {epoch + 1} (patience={patience_counter})")
                break
        
        # Final evaluation on test set
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            test_pred = test_out[test_mask].argmax(dim=1)
            test_accuracy = (test_pred == data.y[test_mask]).float().mean().item()
        
        total_time = time.time() - start_time
        
        print(f"\n✅ REAL Training completed successfully!")
        print(f"   • Total time: {total_time:.1f}s")
        print(f"   • Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"   • Final test accuracy: {test_accuracy:.4f}")
        print(f"   • Epochs trained: {len(training_history['epochs'])}")
        
        return {
            "model_state_dict": best_model_state,
            "training_history": training_history,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy,
            "total_training_time": total_time,
            "final_epoch": len(training_history['epochs'])
        }
    
    def save_real_model(self, model_results: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Save real trained model"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_type = "partial" if self.use_partial else "full"
        
        # Save directory
        models_dir = Path(__file__).parent.parent / "data" / "gnn_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model metadata
        model_info = {
            "model_info": {
                "model_type": "RealGraphAttentionNetwork",
                "training_timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "real_training": True,
                "pytorch_geometric": True,
                "device_used": str(self.device)
            },
            "model_architecture": {
                "model_type": "RealGraphAttentionNetwork",
                "input_dim": metadata['graph_info']['feature_dimension'],
                "hidden_dim": self.model_config["hidden_dim"],
                "output_dim": metadata['num_classes'],
                "num_layers": self.model_config["num_layers"],
                "attention_heads": self.model_config["heads"],
                "dropout": self.model_config["dropout"]
            },
            "training_results": {
                "best_val_accuracy": model_results["best_val_accuracy"],
                "test_accuracy": model_results["test_accuracy"],
                "total_training_time": model_results["total_training_time"],
                "final_epoch": model_results["final_epoch"],
                "training_config": self.model_config
            },
            "training_history": model_results["training_history"]
        }
        
        # Save model metadata
        model_file = models_dir / f"real_gnn_model_{data_type}_{timestamp}.json"
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save model weights
        weights_file = models_dir / f"real_gnn_weights_{data_type}_{timestamp}.pt"
        torch.save(model_results["model_state_dict"], weights_file)
        
        print(f"💾 Real trained model saved:")
        print(f"   • Metadata: {model_file}")
        print(f"   • Weights: {weights_file}")
        
        return str(model_file)


def main():
    """Main function for REAL GNN training"""
    
    parser = argparse.ArgumentParser(description="REAL GNN training with Azure ML")
    parser.add_argument("--partial", action="store_true", help="Train with partial data")
    args = parser.parse_args()
    
    print("🧠 REAL GNN TRAINING WITH AZURE ML")
    print("=" * 50)
    print("NO SIMULATION - Actual PyTorch Geometric training")
    print(f"Data type: {'Partial dataset' if args.partial else 'Full dataset'}")
    print(f"Real training: ✅ Enabled")
    print(f"PyTorch Geometric: ✅ Enabled")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = RealGNNTrainer(use_partial=args.partial)
        
        # Load training data
        training_data, metadata = trainer.load_training_data()
        
        # Prepare PyTorch Geometric data
        data = trainer.prepare_torch_geometric_data(training_data, metadata)
        
        # Create data splits
        train_mask, val_mask, test_mask = trainer.create_data_splits(data)
        
        # Train real GNN model
        training_results = trainer.train_real_gnn(data, train_mask, val_mask, test_mask, metadata)
        
        # Save trained model
        model_file = trainer.save_real_model(training_results, metadata)
        
        print(f"\n{'='*80}")
        print("🎯 REAL GNN TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Final Results:")
        print(f"   • Model type: RealGraphAttentionNetwork")
        print(f"   • Test accuracy: {training_results['test_accuracy']:.4f}")
        print(f"   • Training time: {training_results['total_training_time']:.1f}s")
        print(f"   • Model file: {Path(model_file).name}")
        print(f"\n🚀 Ready for real inference and deployment!")
        
        return 0
        
    except Exception as e:
        print(f"❌ REAL GNN training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())