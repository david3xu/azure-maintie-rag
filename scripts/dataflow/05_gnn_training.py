#!/usr/bin/env python3
"""
GNN Training - Stage 5 of README Data Flow
PyTorch Geometric Data → GNN Training → Trained Model Storage

This script implements the optimized GNN training stage:
- Uses PyTorch Geometric file from Step 04 as direct input
- Trains Graph Neural Network models for maintenance domain
- Supports multiple GNN architectures (GCN, GraphSAGE, GAT)
- Saves trained models for inference and integration
- Eliminates redundant knowledge extraction (11+ minute savings)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric imports
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.async_pattern_manager import get_pattern_manager
from config.discovery_infrastructure_naming import get_discovery_naming
from config.dynamic_ml_config import get_dynamic_ml_config

logger = logging.getLogger(__name__)


class GNNModel(torch.nn.Module):
    """Graph Neural Network for maintenance domain node classification"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, model_type: str = "gcn"
    ):
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


class GNNTrainingStage:
    """Stage 5: PyTorch Geometric Data → GNN Training → Trained Model Storage"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_pytorch_geometric_data(self, source_path: str) -> Data:
        """Load PyTorch Geometric data from Step 04 output"""
        pytorch_file = Path(source_path)

        if not pytorch_file.exists():
            # Try default location
            pytorch_file = Path("data/outputs/step04/pytorch_geometric_maintenance.pt")

        if not pytorch_file.exists():
            raise FileNotFoundError(f"PyTorch Geometric file not found: {source_path}")

        logger.info(f"Loading PyTorch Geometric data from: {pytorch_file}")
        data_dict = torch.load(pytorch_file, weights_only=False)

        graph = data_dict["data"]
        domain = data_dict.get("domain", "unknown")

        logger.info(
            f"Loaded graph - Nodes: {graph.x.size(0)}, Edges: {graph.edge_index.size(1)}, Domain: {domain}"
        )
        return graph

    def create_train_test_split(self, graph: Data, train_ratio: float = 0.8) -> Data:
        """Create train/validation/test masks for node classification"""
        num_nodes = graph.x.size(0)

        # Create random permutation
        perm = torch.randperm(num_nodes)

        # Split indices
        train_size = int(train_ratio * num_nodes)
        val_size = int(0.1 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size : train_size + val_size]] = True
        test_mask[perm[train_size + val_size :]] = True

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

        logger.info(
            f"Split - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}"
        )
        return graph

    def train_model(
        self, model: GNNModel, graph: Data, epochs: int, learning_rate: float
    ) -> Dict[str, Any]:
        """Train the GNN model"""
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

        model.to(self.device)
        graph = graph.to(self.device)

        best_val_acc = 0.0
        training_history = []

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            out = model(graph.x, graph.edge_index, graph.edge_attr)
            loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])

            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, graph.edge_attr)

                # Training accuracy
                pred_train = out[graph.train_mask].argmax(dim=1)
                train_acc = (
                    (pred_train == graph.y[graph.train_mask]).float().mean().item()
                )

                # Validation accuracy
                pred_val = out[graph.val_mask].argmax(dim=1)
                val_acc = (pred_val == graph.y[graph.val_mask]).float().mean().item()

                # Test accuracy
                pred_test = out[graph.test_mask].argmax(dim=1)
                test_acc = (pred_test == graph.y[graph.test_mask]).float().mean().item()

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                }
            )

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
                )

        return {
            "best_val_accuracy": best_val_acc,
            "final_test_accuracy": training_history[-1]["test_acc"],
            "training_history": training_history,
        }

    def save_model(
        self, model: GNNModel, training_results: Dict[str, Any], domain: str
    ) -> str:
        """Save trained model to file"""
        output_dir = Path("data/outputs/step05")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_file = output_dir / f"gnn_model_{domain}.pt"

        # Save model state dict and metadata
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": model.model_type,
                "training_results": training_results,
                "domain": domain,
            },
            model_file,
        )

        logger.info(f"Model saved to: {model_file}")
        return str(model_file)

    async def execute(
        self,
        source_path: str,
        domain: str = "maintenance",
        epochs: int = 100,
        learning_rate: float = 0.01,
        model_type: str = "gcn",
    ) -> Dict[str, Any]:
        """
        Execute GNN training stage using PyTorch Geometric data

        Args:
            source_path: Path to PyTorch Geometric file from Step 04
            domain: Domain for training
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            model_type: GNN architecture (gcn, sage, gat)

        Returns:
            Dict with training results and model information
        """
        print("🧠 Stage 5: GNN Training - PyTorch Geometric → GNN Model → Trained Model")
        print("=" * 70)

        start_time = time.time()

        results = {
            "stage": "05_gnn_training",
            "source_path": str(source_path),
            "domain": domain,
            "training_parameters": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "model_type": model_type,
            },
            "training_metrics": {},
            "model_info": {},
            "success": False,
        }

        try:
            # Load PyTorch Geometric data
            print(f"📂 Loading PyTorch Geometric data from: {source_path}")
            graph = self.load_pytorch_geometric_data(source_path)

            print(
                f"✅ Graph loaded - Nodes: {graph.x.size(0)}, Edges: {graph.edge_index.size(1)}"
            )
            print(
                f"   Node features: {graph.x.size(1)}D, Edge features: {graph.edge_attr.size(1)}D"
            )
            print(f"   Classes: {len(torch.unique(graph.y))}")

            # Create train/test split
            print(f"🔀 Creating train/validation/test split...")
            graph = self.create_train_test_split(graph)

            # Get domain configuration
            pytorch_config = await (
                await get_dynamic_ml_config()
            ).get_learned_ml_config(domain)

            # Initialize model
            input_dim = graph.x.size(1)
            output_dim = len(torch.unique(graph.y))
            hidden_dim = (
                pytorch_config.node_feature_dim
                if hasattr(pytorch_config, "node_feature_dim")
                else 64
            )

            print(
                f"🏗️  Creating {model_type.upper()} model - Input: {input_dim}D, Hidden: {hidden_dim}D, Output: {output_dim}D"
            )
            model = GNNModel(input_dim, hidden_dim, output_dim, model_type)

            # Train model
            print(f"🚀 Starting GNN training for {epochs} epochs...")
            training_results = self.train_model(model, graph, epochs, learning_rate)

            # Save model
            print(f"💾 Saving trained model...")
            model_file = self.save_model(model, training_results, domain)

            # Update results
            results["training_metrics"] = {
                "best_val_accuracy": training_results["best_val_accuracy"],
                "final_test_accuracy": training_results["final_test_accuracy"],
                "epochs_trained": epochs,
                "model_type": model_type,
            }
            results["model_info"] = {
                "model_file": model_file,
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "model_saved": True,
            }
            results["graph_info"] = {
                "num_nodes": graph.x.size(0),
                "num_edges": graph.edge_index.size(1),
                "node_feature_dim": graph.x.size(1),
                "edge_feature_dim": graph.edge_attr.size(1),
                "num_classes": output_dim,
            }

            # Success
            duration = time.time() - start_time
            results["duration_seconds"] = round(duration, 2)
            results["success"] = True

            print(f"✅ Stage 5 Complete:")
            print(f"   🧠 Graph nodes: {graph.x.size(0)}")
            print(f"   🔗 Graph edges: {graph.edge_index.size(1)}")
            print(f"   🎯 Training epochs: {epochs}")
            print(
                f"   📈 Best validation accuracy: {training_results['best_val_accuracy']:.4f}"
            )
            print(
                f"   🎯 Final test accuracy: {training_results['final_test_accuracy']:.4f}"
            )
            print(f"   💾 Model saved: {model_file}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")

            return results

        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(time.time() - start_time, 2)
            print(f"❌ Stage 5 Failed: {e}")
            logger.error(f"GNN training failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for GNN training stage"""
    parser = argparse.ArgumentParser(
        description="Stage 5: GNN Training - Graph Data → Azure ML → Trained Model"
    )
    parser.add_argument(
        "--source",
        default="data/outputs/step04/pytorch_geometric_maintenance.pt",
        help="Path to PyTorch Geometric file from Step 04",
    )
    parser.add_argument("--domain", default="maintenance", help="Domain for training")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--model-type",
        choices=["gcn", "sage", "gat"],
        default="gcn",
        help="GNN architecture type",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate for training"
    )
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Execute stage
    stage = GNNTrainingStage()
    results = await stage.execute(
        source_path=args.source,
        domain=args.domain,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
    )

    # Save results if requested
    if args.output and results.get("success"):
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")

    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
