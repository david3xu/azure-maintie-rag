"""Azure ML service integrations for Universal RAG"""
from .gnn_processor import AzureMLGNNProcessor as AzureMLGNNProcessor
from .classification_service import UniversalClassificationPipeline as AzureMLClassificationService

__all__ = [
    'AzureMLGNNProcessor',
    'AzureMLClassificationService'
]