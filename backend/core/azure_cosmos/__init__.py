"""
Azure Cosmos DB Integration - Consolidated Client
All Azure Cosmos DB functionality consolidated in this module
"""

# Main unified client implementation
from .cosmos_client import UnifiedCosmosClient

# Keep the original implementation available
from .cosmos_gremlin_client import AzureCosmosGremlinClient

# Maintain backwards compatibility with old class names
GremlinClient = UnifiedCosmosClient
CosmosDbClient = UnifiedCosmosClient

__all__ = [
    'UnifiedCosmosClient',
    'AzureCosmosGremlinClient',
    'GremlinClient',
    'CosmosDbClient'
]