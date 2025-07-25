"""Azure Storage Factory for multiple storage accounts with different purposes."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from config.settings import settings
from .storage_client import AzureStorageClient

logger = logging.getLogger(__name__)


class AzureStorageFactory:
    """Factory for creating Azure Storage clients for different purposes"""

    def __init__(self):
        """Initialize storage factory"""
        self.clients: Dict[str, AzureStorageClient] = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize storage clients from configuration"""
        storage_configs = {
            'rag_data': {
                'account_name': settings.azure_storage_account,
                'account_key': settings.azure_storage_key,
                'container_name': settings.azure_blob_container,
                'connection_string': settings.azure_storage_connection_string
            },
            'ml_models': {
                'account_name': settings.azure_ml_storage_account,
                'account_key': settings.azure_ml_storage_key,
                'container_name': settings.azure_ml_blob_container,
                'connection_string': settings.azure_ml_storage_connection_string
            },
            'app_data': {
                'account_name': settings.azure_app_storage_account,
                'account_key': settings.azure_app_storage_key,
                'container_name': settings.azure_app_blob_container,
                'connection_string': settings.azure_app_storage_connection_string
            }
        }
        for client_type, config in storage_configs.items():
            if config['account_name'] and config['account_key']:
                self.clients[client_type] = AzureStorageClient(config)
                logger.info(f"{client_type} storage client initialized")

    def get_storage_client(self, client_type: str) -> AzureStorageClient:
        """Get storage client by type from configuration"""
        if client_type not in self.clients:
            raise ValueError(f"Storage client type '{client_type}' not configured")
        return self.clients[client_type]

    def get_rag_data_client(self):
        return self.get_storage_client('rag_data')

    def get_ml_models_client(self):
        return self.get_storage_client('ml_models')

    def get_app_data_client(self):
        return self.get_storage_client('app_data')

    def list_available_clients(self) -> Dict[str, str]:
        """List available storage clients and their purposes"""
        return {
            'rag_data': 'RAG documents, embeddings, and search data',
            'ml_models': 'ML models, training artifacts, and model metadata',
            'app_data': 'Application logs, cache, and runtime data'
        }

    def get_storage_status(self) -> Dict[str, Any]:
        """Get status of all storage clients"""
        status = {}

        for client_type, client in self.clients.items():
            try:
                connection_status = client.get_connection_status()
                status[client_type] = {
                    'initialized': True,
                    'connection_status': connection_status,
                    'account_name': client.account_name,
                    'container_name': client.container_name
                }
            except Exception as e:
                status[client_type] = {
                    'initialized': False,
                    'error': str(e),
                    'account_name': 'unknown',
                    'container_name': 'unknown'
                }

        return status

    async def upload_file(self, local_path: Path, blob_name: str, client_type: str) -> Dict[str, Any]:
        """Upload file to a specific storage client"""
        client = self.get_storage_client(client_type)
        return await client.upload_file(local_path, blob_name)

    async def download_text(self, blob_name: str, client_type: str) -> Dict[str, Any]:
        """Download text from a specific storage client"""
        client = self.get_storage_client(client_type)
        return await client.download_text(client.container_name, blob_name)


# Global storage factory instance
storage_factory = AzureStorageFactory()


def get_storage_factory() -> AzureStorageFactory:
    """Get global storage factory instance"""
    return storage_factory


def get_storage_client(client_type: str) -> AzureStorageClient:
    """Get storage client by type from configuration"""
    return storage_factory.get_storage_client(client_type)