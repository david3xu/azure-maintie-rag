"""Azure ML service integrations for Universal RAG"""
from .classification_client import AzureClassificationPipeline as AzureMLClassificationService
from .gnn.training.orchestrator import UnifiedGNNTrainingOrchestrator as AzureMLGNNProcessor

__all__ = [
    'AzureMLGNNProcessor', 
    'AzureMLClassificationService'
]