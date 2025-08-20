"""
Azure Cosmos DB Integration - Gremlin Graph Database
All Azure Cosmos DB functionality using the simple client implementation
"""

# Main production client - simple implementation
from .cosmos_gremlin_client import SimpleCosmosGremlinClient

# Aliases for backwards compatibility and different use cases
AzureCosmosGremlinClient = SimpleCosmosGremlinClient
UnifiedCosmosClient = SimpleCosmosGremlinClient
GremlinClient = SimpleCosmosGremlinClient
CosmosDbClient = SimpleCosmosGremlinClient

__all__ = [
    "SimpleCosmosGremlinClient",
    "AzureCosmosGremlinClient",
    "UnifiedCosmosClient",
    "GremlinClient",
    "CosmosDbClient",
]
