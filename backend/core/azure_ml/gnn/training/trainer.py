"""Universal GNN trainer for Azure Universal RAG system."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import json

from ..models.universal_gnn import UniversalGNN, UniversalGNNConfig
from ..data.loader import UnifiedGNNDataLoader
from core.models.universal_rag_models import UniversalTrainingResult

def create_gnn_model(num_features: int, num_classes: int, config: UniversalGNNConfig) -> UniversalGNN:
    """Create GNN model with configuration"""
    return UniversalGNN(
        num_node_features=num_features,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

logger = logging.getLogger(__name__)


class UniversalGNNTrainer:
    """Universal GNN trainer for any domain knowledge graph"""

    def __init__(self, config: UniversalGNNConfig, device: Optional[str] = None):
        """
        Initialize GNN trainer

        Args:
            config: GNN configuration
            device: Device to train on ("cpu", "cuda", or None for auto)
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None

        logger.info(f"UniversalGNNTrainer initialized on device: {self.device}")

    def setup_model(self, num_node_features: int, num_classes: int) -> UniversalGNN:
        """
        Setup GNN model

        Args:
            num_node_features: Number of input node features
            num_classes: Number of output classes

        Returns:
            Configured GNN model
        """
        self.model = create_gnn_model(num_node_features, num_classes, self.config)
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        logger.info(f"Model setup complete: {self.config.conv_type}, "
                   f"hidden_dim={self.config.hidden_dim}, "
                   f"num_layers={self.config.num_layers}")

        return self.model

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            criterion: Loss function

        Returns:
            Average loss and metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        metrics = {"train_loss": 0.0, "train_acc": 0.0}

        for batch in train_loader:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)

            # Compute loss
            loss = criterion(out, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            pred = out.argmax(dim=1)
            correct = pred.eq(batch.y).sum().item()
            accuracy = correct / batch.y.size(0)
            metrics["train_acc"] += accuracy

        # Average metrics
        avg_loss = total_loss / num_batches
        metrics["train_loss"] = avg_loss
        metrics["train_acc"] = metrics["train_acc"] / num_batches

        return avg_loss, metrics

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Validate model

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Average loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        metrics = {"val_loss": 0.0, "val_acc": 0.0}

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                out = self.model(batch.x, batch.edge_index, batch.batch)

                # Compute loss
                loss = criterion(out, batch.y)

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy
                pred = out.argmax(dim=1)
                correct = pred.eq(batch.y).sum().item()
                accuracy = correct / batch.y.size(0)
                metrics["val_acc"] += accuracy

        # Average metrics
        avg_loss = total_loss / num_batches
        metrics["val_loss"] = avg_loss
        metrics["val_acc"] = metrics["val_acc"] / num_batches

        return avg_loss, metrics

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              patience: int = 20,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the GNN model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            Training results and metrics
        """
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")

        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, criterion)

            # Validation
            val_loss, val_metrics = None, {}
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader, criterion)

            # Update learning rate
            if val_loss is not None:
                self.scheduler.step(val_loss)

            # Record metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **train_metrics,
                **val_metrics
            }
            training_history.append(epoch_metrics)

            # Log progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_metrics['train_acc']:.4f}, "
                       f"Time: {epoch_time:.2f}s")

            if val_loss is not None:
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['val_acc']:.4f}")

            # Early stopping
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final results
        results = {
            "training_history": training_history,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
            "early_stopped": patience_counter >= patience
        }

        logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
        return results

    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'device': self.device
        }, path)

    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions

        Args:
            data_loader: Data loader for prediction

        Returns:
            Predictions and true labels
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)

                predictions.extend(pred.cpu().numpy())
                true_labels.extend(batch.y.cpu().numpy())

        return np.array(predictions), np.array(true_labels)


def train_gnn_with_azure_ml(config_dict: Dict[str, Any],
                           data_path: str,
                           output_path: str) -> UniversalTrainingResult:
    """
    Train GNN with Azure ML integration

    Args:
        config_dict: Training configuration
        data_path: Path to training data
        output_path: Path to save model

    Returns:
        Training result
    """
    try:
        # Parse configuration
        gnn_config = UniversalGNNConfig.from_dict(config_dict)

        # Load data
        data_loader = UnifiedGNNDataLoader()
        train_loader, val_loader = data_loader.create_data_loaders(data_path)

        # Setup trainer
        trainer = UniversalGNNTrainer(gnn_config)

        # Get model dimensions from data
        sample_batch = next(iter(train_loader))
        num_node_features = sample_batch.x.size(1)
        num_classes = len(torch.unique(sample_batch.y))

        # Setup model
        trainer.setup_model(num_node_features, num_classes)

        # Train model
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config_dict.get("epochs", 100),
            patience=config_dict.get("patience", 20),
            save_path=output_path
        )

        # Create training result
        result = UniversalTrainingResult(
            model_id=f"gnn-{int(time.time())}",
            model_type="gnn",
            domain=config_dict.get("domain", "general"),
            training_metrics={
                "final_train_loss": training_results["training_history"][-1]["train_loss"],
                "final_train_acc": training_results["training_history"][-1]["train_acc"],
                "best_val_loss": training_results["best_val_loss"],
                "epochs_trained": training_results["final_epoch"]
            },
            validation_metrics={
                "final_val_loss": training_results["training_history"][-1].get("val_loss", 0.0),
                "final_val_acc": training_results["training_history"][-1].get("val_acc", 0.0)
            },
            model_path=output_path,
            training_time=time.time(),
            metadata={
                "gnn_config": gnn_config.to_dict(),
                "training_history": training_results["training_history"],
                "early_stopped": training_results["early_stopped"]
            }
        )

        return result

    except Exception as e:
        logger.error(f"GNN training failed: {e}")
        return UniversalTrainingResult(
            model_id="failed",
            model_type="gnn",
            domain=config_dict.get("domain", "general"),
            metadata={"error": str(e)}
        )