"""
Azure Search Integration - Consolidated Client
All Azure Search functionality consolidated in this module
"""

# Main unified client implementation
from .search_client import UnifiedSearchClient

# Maintain backwards compatibility with old class names
SearchClient = UnifiedSearchClient
AzureSearchVectorService = UnifiedSearchClient
AzureSearchQueryAnalyzer = UnifiedSearchClient
VectorService = UnifiedSearchClient

__all__ = [
    "UnifiedSearchClient",
    "SearchClient",
    "AzureSearchVectorService",
    "AzureSearchQueryAnalyzer",
    "VectorService",
]
