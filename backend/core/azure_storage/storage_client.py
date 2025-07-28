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
        return azure_settings.azure_storage_connection_string.split(';')[0].replace('DefaultEndpointsProtocol=https;AccountName=', 'https://') + '.blob.core.windows.net'
        
    def _get_default_key(self) -> str:
        # Extract account key from connection string
        parts = azure_settings.azure_storage_connection_string.split(';')
        for part in parts:
            if part.startswith('AccountKey='):
                return part.replace('AccountKey=', '')
        return ""
        
    def _initialize_client(self):
        """Initialize blob service client"""
        self._blob_service = BlobServiceClient.from_connection_string(
            azure_settings.azure_storage_connection_string
        )
        
        # Default containers
        self.default_container = azure_settings.azure_blob_container
        
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
            json_data = json.dumps(data, indent=2, default=str)
            return await self.upload_data(json_data, blob_name, container)
            
        except Exception as e:
            return self.handle_azure_error('save_json', e)
    
    async def load_json(self, blob_name: str, container: str = None) -> Dict[str, Any]:
        """Load JSON data from blob storage"""
        try:
            result = await self.download_data(blob_name, container)
            
            if result['success']:
                data = json.loads(result['data'])
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
    
    async def _ensure_container_exists(self, container_name: str):
        """Ensure container exists, create if not"""
        try:
            container_client = self._blob_service.get_container_client(container_name)
            container_client.create_container()
        except Exception:
            # Container might already exist
            pass
    
    def generate_blob_name(self, prefix: str, extension: str = None) -> str:
        """Generate unique blob name"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if extension:
            return f"{prefix}_{timestamp}.{extension}"
        return f"{prefix}_{timestamp}"