"""Azure Blob Storage client for Universal RAG system."""

import logging
import time
from typing import Dict, List, Any, Optional, IO
from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError

from config.settings import azure_settings

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

        if not self.account_name:
            raise ValueError("Azure Storage account name is required")

        # Initialize client (follows azure_openai.py error handling pattern)
        try:
            self.credential = self._get_azure_credential()
            self.blob_service_client = BlobServiceClient(
                account_url=f"https://{self.account_name}.blob.core.windows.net",
                credential=self.credential
            )
            self._ensure_container_exists()
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage client: {e}")
            raise

        logger.info(f"AzureStorageClient initialized for container: {self.container_name}")

    def _get_azure_credential(self):
        """Enterprise credential management - data-driven from config"""
        if azure_settings.azure_use_managed_identity and azure_settings.azure_managed_identity_client_id:
            from azure.identity import ManagedIdentityCredential
            return ManagedIdentityCredential(client_id=azure_settings.azure_managed_identity_client_id)

        # Fallback to account key if available
        if self.account_key:
            return self.account_key

        # Final fallback to DefaultAzureCredential
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential()

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

    async def create_container(self, container_name: str) -> bool:
        """Create a new container in Azure Blob Storage, returns True if created or already exists"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {container_name}")
            else:
                logger.info(f"Container already exists: {container_name}")
            return True
        except Exception as e:
            logger.error(f"Container creation failed: {e}")
            return False

    async def upload_text(self, container_name: str, blob_name: str, text: str) -> str:
        """Upload text content to Azure Blob Storage, returns blob name on success"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            text_bytes = text.encode('utf-8')
            blob_client.upload_blob(text_bytes, overwrite=True)
            logger.info(f"Uploaded text to blob: {blob_name} in container: {container_name}")
            return blob_name
        except Exception as e:
            logger.error(f"Text upload failed: {e}")
            return ""

    async def download_text(self, container_name: str, blob_name: str) -> str:
        """Download text content from Azure Blob Storage - enterprise text operations pattern"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            # Download as text with UTF-8 encoding
            download_stream = blob_client.download_blob()
            text_content = download_stream.readall().decode('utf-8')
            logger.info(f"Downloaded text from blob: {blob_name} in container: {container_name}")
            return text_content
        except Exception as e:
            logger.error(f"Text download failed for {blob_name}: {e}")
            return ""

    def upload_file(self, local_path: Path, blob_name: str, overwrite: bool = True) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage with telemetry - data-driven monitoring"""
        start_time = time.time()

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)

            # Add telemetry data
            operation_time = time.time() - start_time

            return {
                "success": True,
                "blob_name": blob_name,
                "container": self.container_name,
                "size_bytes": local_path.stat().st_size,
                "operation_duration_ms": operation_time * 1000,  # Data-driven metrics
                "telemetry": {
                    "service": "azure_storage",
                    "operation": "upload_file",
                    "timestamp": time.time()
                }
            }

        except Exception as e:
            # Add error telemetry
            operation_time = time.time() - start_time
            logger.error(f"File upload failed for {blob_name}: {e}")

            return {
                "success": False,
                "error": str(e),
                "blob_name": blob_name,
                "operation_duration_ms": operation_time * 1000,
                "telemetry": {
                    "service": "azure_storage",
                    "operation": "upload_file_failed",
                    "error_type": type(e).__name__
                }
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

    def list_blobs(self, container_name: str = None, prefix: str = "") -> List[Dict[str, Any]]:
        """List blobs in specified container with optional prefix"""
        try:
            # Data-driven container selection
            target_container = container_name or self.container_name
            container_client = self.blob_service_client.get_container_client(target_container)
            blob_list = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blob_list.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "content_type": getattr(blob, 'content_settings', None) and getattr(blob.content_settings, 'content_type', 'unknown') or 'unknown'
                })
            return blob_list
        except Exception as e:
            logger.error(f"Blob listing failed for container {target_container}: {e}")
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

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob from Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_name} from container: {container_name}")
            return True
        except Exception as e:
            logger.error(f"Blob deletion failed: {e}")
            return False

    async def delete_all_blobs(self, container_name: str) -> int:
        """Delete all blobs in a container, returns number of deleted blobs"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            deleted_count = 0

            # List all blobs and delete them
            blobs = container_client.list_blobs()
            for blob in blobs:
                try:
                    container_client.delete_blob(blob.name)
                    deleted_count += 1
                    logger.info(f"Deleted blob: {blob.name}")
                except Exception as e:
                    logger.error(f"Failed to delete blob {blob.name}: {e}")

            logger.info(f"Deleted {deleted_count} blobs from container: {container_name}")
            return deleted_count
        except Exception as e:
            logger.error(f"Bulk blob deletion failed: {e}")
            return 0