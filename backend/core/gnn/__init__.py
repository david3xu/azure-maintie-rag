"""
GNN Integration Module for Universal RAG
Neural Intelligence Implementation for Universal Domain Support
"""

from .universal_gnn_processor import UniversalGNNDataProcessor
from .gnn_models import MaintenanceGNNModel, GNNTrainer
from .graph_dataset import MaintIEGraphDataset

__all__ = [
    'UniversalGNNDataProcessor',
    'MaintenanceGNNModel',
    'GNNTrainer',
    'MaintIEGraphDataset'
]