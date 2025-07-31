"""
Azure Storage Integration - Consolidated Client
All Azure Storage functionality consolidated in this module
"""

# Main unified client implementation
from .storage_client import UnifiedStorageClient

# Maintain backwards compatibility with old class names
StorageClient = UnifiedStorageClient
BlobStorageClient = UnifiedStorageClient
AzureStorageClient = UnifiedStorageClient

__all__ = [
    'UnifiedStorageClient',
    'StorageClient',
    'BlobStorageClient', 
    'AzureStorageClient'
]