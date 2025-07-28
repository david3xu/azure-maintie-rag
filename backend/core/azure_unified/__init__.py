"""
Unified Azure Services Module
Consolidates all Azure service clients into cohesive interfaces
"""

from .openai_client import UnifiedAzureOpenAIClient
from .cosmos_client import UnifiedCosmosClient  
from .search_client import UnifiedSearchClient
from .storage_client import UnifiedStorageClient
from .base_client import BaseAzureClient

__all__ = [
    'UnifiedAzureOpenAIClient',
    'UnifiedCosmosClient', 
    'UnifiedSearchClient',
    'UnifiedStorageClient',
    'BaseAzureClient'
]