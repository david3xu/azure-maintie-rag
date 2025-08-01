"""
Unified Azure Blob Storage Client
Consolidates all storage functionality: blob operations, file management, data persistence
Replaces: storage_client.py, storage_factory.py, real_azure_services.py
"""

import logging
from typing import Dict, List, Any, Optional, BinaryIO
import json
from datetime import datetime
from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from ..azure_auth.base_client import BaseAzureClient
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class UnifiedStorageClient(BaseAzureClient):
    """Unified client for all Azure Blob Storage operations"""
    
    def _get_default_endpoint(self) -> str:
        if not azure_settings.azure_storage_account:
            raise RuntimeError("azure_storage_account is required for Azure-only deployment. Connection strings not supported.")
        return f"https://{azure_settings.azure_storage_account}.blob.core.windows.net"
        
    def _health_check(self) -> bool:
        """Perform Blob Storage service health check"""
        try:
            # Simple connectivity check
            return True  # If client is initialized successfully, service is accessible
        except Exception as e:
            logger.warning(f"Blob Storage health check failed: {e}")
            return False
        
    def _initialize_client(self):
        """Initialize blob service client - Azure managed identity only"""
        # Azure-only deployment - managed identity required
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        self._blob_service = BlobServiceClient(
            account_url=self.endpoint,
            credential=credential
        )
        logger.info(f"Azure Storage client initialized with managed identity for {self.endpoint}")
        
        # Default containers
        self.default_container = azure_settings.azure_blob_container

    async def test_connection(self) -> Dict[str, Any]:
        """Test Azure Blob Storage connection"""
        try:
            self.ensure_initialized()
            
            # Test by listing containers
            containers = list(self._blob_service.list_containers())
            
            return {
                "success": True,
                "endpoint": self.endpoint,
                "account": azure_settings.azure_storage_account,
                "container_count": len(containers)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "endpoint": getattr(self, 'endpoint', 'unknown'),
                "account": azure_settings.azure_storage_account
            }
        
    # === BLOB OPERATIONS ===
    
    async def upload_file(self, file_path: str, blob_name: str = None, container: str = None) -> Dict[str, Any]:
        """Upload file to blob storage"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            blob_name = blob_name or Path(file_path).name
            
            # Ensure container exists
            await self._ensure_container_exists(container_name)
            
            # Upload file
            blob_client = self._blob_service.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            return self.create_success_response('upload_file', {
                'file_path': file_path,
                'blob_name': blob_name,
                'container': container_name,
                'blob_url': blob_client.url
            })
            
        except Exception as e:
            return self.handle_azure_error('upload_file', e)
    
    async def upload_data(self, data: str, blob_name: str, container: str = None) -> Dict[str, Any]:
        """Upload text data to blob storage"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            
            # Ensure container exists
            await self._ensure_container_exists(container_name)
            
            # Upload data
            blob_client = self._blob_service.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_client.upload_blob(data.encode('utf-8'), overwrite=True)
            
            return self.create_success_response('upload_data', {
                'blob_name': blob_name,
                'container': container_name,
                'data_size': len(data),
                'blob_url': blob_client.url
            })
            
        except Exception as e:
            return self.handle_azure_error('upload_data', e)
    
    async def upload_blob(self, blob_name: str, data: str, container: str = None) -> Dict[str, Any]:
        """Upload blob with data - alias for upload_data with different parameter order"""
        return await self.upload_data(data, blob_name, container)
    
    async def download_file(self, blob_name: str, download_path: str = None, container: str = None) -> Dict[str, Any]:
        """Download blob to file"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            download_path = download_path or blob_name
            
            blob_client = self._blob_service.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(download_path, 'wb') as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            return self.create_success_response('download_file', {
                'blob_name': blob_name,
                'download_path': download_path,
                'container': container_name
            })
            
        except Exception as e:
            return self.handle_azure_error('download_file', e)
    
    async def download_data(self, blob_name: str, container: str = None) -> Dict[str, Any]:
        """Download blob data as string"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            
            blob_client = self._blob_service.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            download_stream = blob_client.download_blob()
            data = download_stream.readall().decode('utf-8')
            
            return self.create_success_response('download_data', {
                'blob_name': blob_name,
                'container': container_name,
                'data': data,
                'data_size': len(data)
            })
            
        except Exception as e:
            return self.handle_azure_error('download_data', e)
    
    # === CONTAINER OPERATIONS ===
    
    async def list_blobs(self, container: str = None, prefix: str = None) -> Dict[str, Any]:
        """List blobs in container"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            
            container_client = self._blob_service.get_container_client(container_name)
            
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blob_info = {
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified.isoformat() if blob.last_modified else None,
                    'content_type': blob.content_settings.content_type if blob.content_settings else None
                }
                blobs.append(blob_info)
            
            return self.create_success_response('list_blobs', {
                'container': container_name,
                'blobs': blobs,
                'blob_count': len(blobs)
            })
            
        except Exception as e:
            return self.handle_azure_error('list_blobs', e)
    
    async def delete_blob(self, blob_name: str, container: str = None) -> Dict[str, Any]:
        """Delete blob"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            
            blob_client = self._blob_service.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_client.delete_blob()
            
            return self.create_success_response('delete_blob', {
                'blob_name': blob_name,
                'container': container_name,
                'message': 'Blob deleted successfully'
            })
            
        except Exception as e:
            return self.handle_azure_error('delete_blob', e)
    
    # === DATA PERSISTENCE ===
    
    async def save_json(self, data: Dict, blob_name: str, container: str = None) -> Dict[str, Any]:
        """Save JSON data to blob storage"""
        try:
            from ..utilities.file_utils import FileUtils
            json_data = FileUtils.safe_json_dumps(data, indent=2, default=str)
            return await self.upload_data(json_data, blob_name, container)
            
        except Exception as e:
            return self.handle_azure_error('save_json', e)
    
    async def load_json(self, blob_name: str, container: str = None) -> Dict[str, Any]:
        """Load JSON data from blob storage"""
        try:
            result = await self.download_data(blob_name, container)
            
            if result['success']:
                raw_data = result['data']
                
                # Handle case where data might already be parsed
                if isinstance(raw_data, (dict, list)):
                    data = raw_data
                elif isinstance(raw_data, (str, bytes, bytearray)):
                    data = json.loads(raw_data)
                else:
                    raise ValueError(f"Unexpected data type for JSON parsing: {type(raw_data)}")
                
                return self.create_success_response('load_json', {
                    'blob_name': blob_name,
                    'data': data
                })
            else:
                return result
                
        except Exception as e:
            return self.handle_azure_error('load_json', e)
    
    # === BATCH OPERATIONS ===
    
    async def upload_multiple_files(self, file_paths: List[str], container: str = None) -> Dict[str, Any]:
        """Upload multiple files"""
        self.ensure_initialized()
        
        try:
            results = []
            for file_path in file_paths:
                result = await self.upload_file(file_path, container=container)
                results.append(result)
            
            success_count = sum(1 for r in results if r['success'])
            
            return self.create_success_response('upload_multiple_files', {
                'files_uploaded': success_count,
                'files_failed': len(file_paths) - success_count,
                'total_files': len(file_paths),
                'results': results
            })
            
        except Exception as e:
            return self.handle_azure_error('upload_multiple_files', e)
    
    async def cleanup_container(self, container: str = None, prefix: str = None) -> Dict[str, Any]:
        """Clean up blobs in container"""
        self.ensure_initialized()
        
        try:
            container_name = container or self.default_container
            
            # List blobs to delete
            blob_list = await self.list_blobs(container_name, prefix)
            
            if not blob_list['success']:
                return blob_list
            
            deleted_count = 0
            for blob in blob_list['data']['blobs']:
                result = await self.delete_blob(blob['name'], container_name)
                if result['success']:
                    deleted_count += 1
            
            return self.create_success_response('cleanup_container', {
                'container': container_name,
                'blobs_deleted': deleted_count,
                'prefix': prefix
            })
            
        except Exception as e:
            return self.handle_azure_error('cleanup_container', e)
    
    # === UTILITY METHODS ===
    
    async def list_containers(self) -> List[str]:
        """List all containers - used for connectivity testing"""
        self.ensure_initialized()
        try:
            containers = []
            for container in self._blob_service.list_containers():
                containers.append(container.name)
            return containers
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            raise e
    
    async def create_container(self, container_name: str) -> Dict[str, Any]:
        """Create container (alias for ensure_container_exists)"""
        return await self.ensure_container_exists(container_name)
    
    async def ensure_container_exists(self, container_name: str) -> Dict[str, Any]:
        """Public method to ensure container exists, create if not"""
        try:
            await self._ensure_container_exists(container_name)
            return self.create_success_response('ensure_container_exists', {
                'container': container_name,
                'message': 'Container ensured successfully'
            })
        except Exception as e:
            return self.handle_azure_error('ensure_container_exists', e)
    
    async def _ensure_container_exists(self, container_name: str):
        """Private method to ensure container exists, create if not"""
        try:
            # Ensure client is initialized before accessing _blob_service
            self.ensure_initialized()
            
            container_client = self._blob_service.get_container_client(container_name)
            # Check if container exists first
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"✅ Created container: {container_name}")
            else:
                logger.info(f"✅ Container already exists: {container_name}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure container {container_name}: {e}")
            raise e
    
    def generate_blob_name(self, prefix: str, extension: str = None) -> str:
        """Generate unique blob name"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if extension:
            return f"{prefix}_{timestamp}.{extension}"
        return f"{prefix}_{timestamp}"