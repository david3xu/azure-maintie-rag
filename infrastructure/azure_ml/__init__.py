"""Azure ML service integrations for Universal RAG"""
from .classification_client import (
    AzureClassificationPipeline as AzureMLClassificationService,
)

# from .gnn.training.orchestrator import UnifiedGNNTrainingOrchestrator as AzureMLGNNProcessor  # Disabled - GNN module removed

__all__ = [
    # 'AzureMLGNNProcessor',  # Disabled - GNN module removed
    "AzureMLClassificationService"
]
