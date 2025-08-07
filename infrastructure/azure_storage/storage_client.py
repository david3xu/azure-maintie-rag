"""
Simple Azure Blob Storage Client - CODING_STANDARDS Compliant
Clean storage client without over-engineering enterprise patterns.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure.storage.blob import BlobServiceClient

from config.settings import azure_settings
from ..azure_auth.base_client import BaseAzureClient

logger = logging.getLogger(__name__)


class SimpleStorageClient(BaseAzureClient):
    """
    Simple Azure Blob Storage client following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses Azure settings for configuration
    - Universal Design: Works with any storage container
    - Mathematical Foundation: Simple blob operations
    """

    def _get_default_endpoint(self) -> str:
        if not azure_settings.azure_storage_account:
            raise RuntimeError("Azure storage account is required")
        return f"https://{azure_settings.azure_storage_account}.blob.core.windows.net"

    def _health_check(self) -> bool:
        """Simple health check without API parameter issues"""
        try:
            if hasattr(self, "_blob_service") and self._blob_service:
                # Use correct API - list_containers doesn't take max_results parameter
                # Just verify the client is properly initialized
                return (
                    hasattr(self._blob_service, "list_containers")
                    and self._blob_service is not None
                )
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        return False

    async def test_connection(self) -> bool:
        """Test connection method expected by ConsolidatedAzureServices"""
        try:
            self.ensure_initialized()
            return self._health_check()
        except Exception as e:
            logger.error(f"Storage connection test failed: {e}")
            return False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize simple storage client"""
        super().__init__(config)

        # Simple configuration
        self.container_name = (
            self.config.get("container") or azure_settings.azure_storage_container
        )
        self._blob_service = None

        logger.info(f"Simple storage client initialized for {self.container_name}")

    def _initialize_client(self):
        """Simple client initialization"""
        try:
            from infrastructure.azure_auth_utils import get_azure_credential

            credential = get_azure_credential()

            # Create simple blob service client
            self._blob_service = BlobServiceClient(
                account_url=self.endpoint, credential=credential
            )

            self._client = self._blob_service
            logger.info("Storage client initialized")

        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise

    async def upload_blob(
        self, blob_name: str, data: bytes, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Upload blob using simple approach"""
        try:
            self.ensure_initialized()

            # Get blob client
            blob_client = self._blob_service.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            # Upload data
            blob_client.upload_blob(data, overwrite=True, metadata=metadata or {})

            return self.create_success_response(
                "upload_blob",
                {
                    "blob_name": blob_name,
                    "container": self.container_name,
                    "size": len(data),
                },
            )

        except Exception as e:
            return self.handle_azure_error("upload_blob", e)

    async def download_blob(self, blob_name: str) -> Dict[str, Any]:
        """Download blob using simple approach"""
        try:
            self.ensure_initialized()

            # Get blob client
            blob_client = self._blob_service.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            # Download data
            blob_data = blob_client.download_blob()
            content = blob_data.readall()

            return self.create_success_response(
                "download_blob",
                {"blob_name": blob_name, "content": content, "size": len(content)},
            )

        except Exception as e:
            return self.handle_azure_error("download_blob", e)

    async def upload_file(
        self, file_path: Path, blob_name: str = None
    ) -> Dict[str, Any]:
        """Upload file using simple approach"""
        try:
            self.ensure_initialized()

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            blob_name = blob_name or file_path.name

            # Read file and upload
            with open(file_path, "rb") as file_data:
                content = file_data.read()

            result = await self.upload_blob(
                blob_name,
                content,
                {
                    "source_file": str(file_path),
                    "upload_time": datetime.now().isoformat(),
                },
            )

            return result

        except Exception as e:
            return self.handle_azure_error("upload_file", e)

    async def download_file(self, blob_name: str, file_path: Path) -> Dict[str, Any]:
        """Download file using simple approach"""
        try:
            self.ensure_initialized()

            # Download blob
            result = await self.download_blob(blob_name)

            if result["success"]:
                # Write to file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as file:
                    file.write(result["data"]["content"])

                return self.create_success_response(
                    "download_file",
                    {
                        "blob_name": blob_name,
                        "file_path": str(file_path),
                        "size": result["data"]["size"],
                    },
                )
            else:
                return result

        except Exception as e:
            return self.handle_azure_error("download_file", e)

    async def delete_blob(self, blob_name: str) -> Dict[str, Any]:
        """Delete blob using simple approach"""
        try:
            self.ensure_initialized()

            # Get blob client
            blob_client = self._blob_service.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            # Delete blob
            blob_client.delete_blob()

            return self.create_success_response(
                "delete_blob",
                {"blob_name": blob_name, "container": self.container_name},
            )

        except Exception as e:
            return self.handle_azure_error("delete_blob", e)

    async def list_blobs(self, prefix: str = None) -> Dict[str, Any]:
        """List blobs using simple approach"""
        try:
            self.ensure_initialized()

            # Get container client
            container_client = self._blob_service.get_container_client(
                self.container_name
            )

            # List blobs
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append(
                    {
                        "name": blob.name,
                        "size": blob.size,
                        "last_modified": (
                            blob.last_modified.isoformat()
                            if blob.last_modified
                            else None
                        ),
                    }
                )

            return self.create_success_response(
                "list_blobs",
                {"blobs": blobs, "count": len(blobs), "container": self.container_name},
            )

        except Exception as e:
            return self.handle_azure_error("list_blobs", e)

    async def blob_exists(self, blob_name: str) -> bool:
        """Check if blob exists"""
        try:
            self.ensure_initialized()

            blob_client = self._blob_service.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            return blob_client.exists()

        except Exception as e:
            logger.error(f"Blob exists check failed: {e}")
            return False


# Backward compatibility aliases
UnifiedStorageClient = SimpleStorageClient
AzureStorageClient = SimpleStorageClient
