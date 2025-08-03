"""
Azure Cosmos DB Integration - Gremlin Graph Database
All Azure Cosmos DB functionality using the complete AzureCosmosGremlinClient
"""

# Main production client - complete implementation
from .cosmos_gremlin_client import AzureCosmosGremlinClient

# Aliases for backwards compatibility
UnifiedCosmosClient = AzureCosmosGremlinClient
GremlinClient = AzureCosmosGremlinClient
CosmosDbClient = AzureCosmosGremlinClient

__all__ = [
    "AzureCosmosGremlinClient",
    "UnifiedCosmosClient",
    "GremlinClient",
    "CosmosDbClient",
]
