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
        """Initialize storage clients for different purposes"""
        try:
            # RAG Data Storage Client
            if settings.azure_storage_account and settings.azure_storage_key:
                rag_config = {
                    'account_name': settings.azure_storage_account,
                    'account_key': settings.azure_storage_key,
                    'container_name': settings.azure_blob_container,
                    'connection_string': settings.azure_storage_connection_string
                }
                self.clients['rag_data'] = AzureStorageClient(rag_config)
                logger.info(f"RAG Data Storage client initialized: {settings.azure_storage_account}")

            # ML Storage Client
            if settings.azure_ml_storage_account and settings.azure_ml_storage_key:
                ml_config = {
                    'account_name': settings.azure_ml_storage_account,
                    'account_key': settings.azure_ml_storage_key,
                    'container_name': settings.azure_ml_blob_container,
                    'connection_string': settings.azure_ml_storage_connection_string
                }
                self.clients['ml_models'] = AzureStorageClient(ml_config)
                logger.info(f"ML Storage client initialized: {settings.azure_ml_storage_account}")

            # Application Storage Client
            if settings.azure_app_storage_account and settings.azure_app_storage_key:
                app_config = {
                    'account_name': settings.azure_app_storage_account,
                    'account_key': settings.azure_app_storage_key,
                    'container_name': settings.azure_app_blob_container,
                    'connection_string': settings.azure_app_storage_connection_string
                }
                self.clients['app_data'] = AzureStorageClient(app_config)
                logger.info(f"Application Storage client initialized: {settings.azure_app_storage_account}")

        except Exception as e:
            logger.error(f"Failed to initialize storage clients: {e}")
            raise

    def get_rag_data_client(self) -> AzureStorageClient:
        """Get RAG data storage client"""
        if 'rag_data' not in self.clients:
            raise RuntimeError("RAG data storage client not initialized")
        return self.clients['rag_data']

    def get_ml_models_client(self) -> AzureStorageClient:
        """Get ML models storage client"""
        if 'ml_models' not in self.clients:
            raise RuntimeError("ML models storage client not initialized")
        return self.clients['ml_models']

    def get_app_data_client(self) -> AzureStorageClient:
        """Get application data storage client"""
        if 'app_data' not in self.clients:
            raise RuntimeError("Application data storage client not initialized")
        return self.clients['app_data']

    def get_client(self, client_type: str) -> AzureStorageClient:
        """Get storage client by type"""
        if client_type not in self.clients:
            raise RuntimeError(f"Storage client '{client_type}' not initialized")
        return self.clients[client_type]

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

    async def upload_rag_document(self, local_path: Path, blob_name: str) -> Dict[str, Any]:
        """Upload document to RAG data storage"""
        client = self.get_rag_data_client()
        return await client.upload_file(local_path, blob_name)

    async def upload_ml_model(self, local_path: Path, blob_name: str) -> Dict[str, Any]:
        """Upload ML model to ML storage"""
        client = self.get_ml_models_client()
        return await client.upload_file(local_path, blob_name)

    async def upload_app_data(self, local_path: Path, blob_name: str) -> Dict[str, Any]:
        """Upload application data to app storage"""
        client = self.get_app_data_client()
        return await client.upload_file(local_path, blob_name)

    async def download_rag_document(self, blob_name: str) -> Dict[str, Any]:
        """Download document from RAG data storage"""
        client = self.get_rag_data_client()
        return await client.download_text(client.container_name, blob_name)

    async def download_ml_model(self, blob_name: str) -> Dict[str, Any]:
        """Download ML model from ML storage"""
        client = self.get_ml_models_client()
        return await client.download_text(client.container_name, blob_name)

    async def download_app_data(self, blob_name: str) -> Dict[str, Any]:
        """Download application data from app storage"""
        client = self.get_app_data_client()
        return await client.download_text(client.container_name, blob_name)


# Global storage factory instance
storage_factory = AzureStorageFactory()


def get_storage_factory() -> AzureStorageFactory:
    """Get global storage factory instance"""
    return storage_factory


def get_rag_storage_client() -> AzureStorageClient:
    """Get RAG data storage client"""
    return storage_factory.get_rag_data_client()


def get_ml_storage_client() -> AzureStorageClient:
    """Get ML models storage client"""
    return storage_factory.get_ml_models_client()


def get_app_storage_client() -> AzureStorageClient:
    """Get application data storage client"""
    return storage_factory.get_app_data_client()