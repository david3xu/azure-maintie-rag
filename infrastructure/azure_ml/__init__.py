"""Azure ML service integrations for Universal RAG"""
from .classification_client import (
    AzureClassificationPipeline as AzureMLClassificationService,
)
from .gnn_training_client import GNNTrainingClient
from .gnn_inference_client import GNNInferenceClient
from .gnn_model import UniversalGNN, UniversalGNNConfig, create_gnn_model

__all__ = [
    "AzureMLClassificationService",
    "GNNTrainingClient",
    "GNNInferenceClient", 
    "UniversalGNN",
    "UniversalGNNConfig",
    "create_gnn_model"
]
