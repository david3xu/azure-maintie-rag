"""Azure Blob Storage client for Universal RAG system."""

import logging
from typing import Dict, List, Any, Optional, IO
from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError

from backend.config.azure_settings import azure_settings

logger = logging.getLogger(__name__)


class AzureStorageClient:
    """Universal Azure Blob Storage client - follows azure_openai.py pattern"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure Storage client - follows existing pattern"""
        self.config = config or {}

        # Load from environment if not in config (matches azure_openai.py pattern)
        self.account_name = self.config.get('account_name') or azure_settings.azure_storage_account
        self.account_key = self.config.get('account_key') or azure_settings.azure_storage_key
        self.container_name = self.config.get('container_name') or azure_settings.azure_blob_container

        if not self.account_name or not self.account_key:
            raise ValueError("Azure Storage account name and key are required")

        # Initialize client (follows azure_openai.py error handling pattern)
        try:
            self.blob_service_client = BlobServiceClient(
                account_url=f"https://{self.account_name}.blob.core.windows.net",
                credential=self.account_key
            )
            self._ensure_container_exists()
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage client: {e}")
            raise

        logger.info(f"AzureStorageClient initialized for container: {self.container_name}")

    def _ensure_container_exists(self):
        """Ensure container exists - create if needed"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
        except AzureError as e:
            logger.error(f"Container operation failed: {e}")
            raise

    def upload_file(self, local_path: Path, blob_name: str, overwrite: bool = True) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage - matches azure_openai.py return pattern"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)

            return {
                "success": True,
                "blob_name": blob_name,
                "container": self.container_name,
                "size_bytes": local_path.stat().st_size
            }

        except Exception as e:
            logger.error(f"File upload failed for {blob_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "blob_name": blob_name
            }

    def download_file(self, blob_name: str, local_path: Path) -> Dict[str, Any]:
        """Download file from Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            return {
                "success": True,
                "blob_name": blob_name,
                "local_path": str(local_path),
                "size_bytes": local_path.stat().st_size
            }

        except Exception as e:
            logger.error(f"File download failed for {blob_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "blob_name": blob_name
            }

    def list_blobs(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List blobs in container with optional prefix"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = []

            for blob in container_client.list_blobs(name_starts_with=prefix):
                blob_list.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "content_type": blob.get('content_settings', {}).get('content_type', 'unknown')
                })

            return blob_list

        except Exception as e:
            logger.error(f"Blob listing failed: {e}")
            return []

    def get_connection_status(self) -> Dict[str, Any]:
        """Get storage connection status - follows azure_openai.py pattern"""
        try:
            # Test connection by getting account info
            account_info = self.blob_service_client.get_account_information()

            return {
                "status": "healthy",
                "account_name": self.account_name,
                "container_name": self.container_name,
                "account_kind": account_info.get('account_kind', 'unknown')
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "account_name": self.account_name
            }