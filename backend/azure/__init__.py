"""Azure service integrations for Universal RAG system.

This package contains Azure-specific service clients for:
- Azure Blob Storage (data storage)
- Azure Cognitive Search (vector search)
- Azure Cosmos DB (knowledge graphs)
- Azure Machine Learning (model training)

All clients follow the existing integration patterns from azure_openai.py
"""

from .storage_client import AzureStorageClient
from .search_client import AzureCognitiveSearchClient
from .cosmos_client import AzureCosmosClient
from .ml_client import AzureMLClient

__all__ = [
    'AzureStorageClient',
    'AzureCognitiveSearchClient',
    'AzureCosmosClient',
    'AzureMLClient'
]