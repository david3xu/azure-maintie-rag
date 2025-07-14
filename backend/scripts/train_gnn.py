"""
Script to train GNN model on MaintIE data
Run this after GNN data preparation is complete
"""

import logging
import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.data_transformer import MaintIEDataTransformer
from src.gnn.data_preparation import MaintIEGNNDataProcessor
from src.gnn.gnn_models import MaintenanceGNNModel, GNNTrainer

def main():
    """Train GNN model on MaintIE data"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting GNN model training...")

    # Load data
    data_transformer = MaintIEDataTransformer()
    gnn_processor = MaintIEGNNDataProcessor(data_transformer)
    gnn_data = gnn_processor.prepare_gnn_data(use_cache=True)

    if gnn_data['full_data'] is None:
        logger.error("GNN data not available. Install PyTorch Geometric: pip install torch-geometric")
        return

    # Model configuration
    config = {
        'input_dim': gnn_data['full_data'].x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'num_entity_types': len(set(gnn_data['node_labels'].tolist())),
        'gnn_type': 'GraphSAGE',
        'dropout': 0.2
    }

    # Create model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaintenanceGNNModel(config)
    trainer = GNNTrainer(model, device)

    logger.info(f"Training on device: {device}")
    logger.info(f"Model config: {config}")

    # Train model
    train_data = gnn_data['train_data']
    val_data = gnn_data['val_data']
    train_labels = gnn_data['node_labels']
    val_labels = gnn_data['node_labels']

    results = trainer.train(train_data, val_data, train_labels, val_labels, num_epochs=200)

    # Save model
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "maintenance_gnn.pt"
    trainer.save_model(model_path)

    logger.info(f"Training completed. Model saved to {model_path}")
    logger.info(f"Final validation accuracy: {results['val_accuracies'][-1]:.4f}")

if __name__ == "__main__":
    main()