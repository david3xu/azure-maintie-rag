"""
Azure Storage Integration - Simple Client
All Azure Storage functionality using the simple client implementation
"""

# Main unified client implementation
from .storage_client import SimpleStorageClient

# Maintain backwards compatibility with old class names
UnifiedStorageClient = SimpleStorageClient
StorageClient = SimpleStorageClient
BlobStorageClient = SimpleStorageClient
AzureStorageClient = SimpleStorageClient

__all__ = [
    "SimpleStorageClient",
    "UnifiedStorageClient",
    "StorageClient",
    "BlobStorageClient",
    "AzureStorageClient",
]
