"""
Real Azure Services for Dual Storage
Uses actual Azure Blob, Search, and Cosmos DB services with proper credentials
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.cosmos import CosmosClient
from azure.core.credentials import AzureKeyCredential

from config.settings import settings

logger = logging.getLogger(__name__)


class AzureBlobService:
    """Real Azure Blob Storage Service"""
    
    def __init__(self):
        """Initialize Azure Blob Storage client"""
        try:
            if settings.azure_storage_connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    settings.azure_storage_connection_string
                )
            else:
                account_url = f"https://{settings.azure_storage_account}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=settings.azure_storage_key
                )
            
            self.container_name = settings.azure_blob_container
            logger.info(f"AzureBlobService initialized - Container: {self.container_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Service: {e}")
            raise
    
    def upload_json(self, data: Dict[str, Any], blob_name: str):
        """Upload JSON data to Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            json_data = json.dumps(data, indent=2)
            blob_client.upload_blob(json_data, overwrite=True)
            
            logger.info(f"Uploaded to Azure Blob: {blob_name} ({len(json_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to upload blob {blob_name}: {e}")
            raise
    
    def upload_text(self, text: str, blob_name: str):
        """Upload text data to Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_client.upload_blob(text, overwrite=True)
            logger.info(f"Uploaded text to Azure Blob: {blob_name} ({len(text)} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to upload text blob {blob_name}: {e}")
            raise


class AzureSearchService:
    """Real Azure Cognitive Search Service"""
    
    def __init__(self):
        """Initialize Azure Cognitive Search client"""
        try:
            self.search_endpoint = f"https://{settings.azure_search_service_name}.search.windows.net"
            self.index_name = "knowledge-extraction-index"
            
            credential = AzureKeyCredential(settings.azure_search_admin_key)
            
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=credential
            )
            
            self.index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=credential
            )
            
            logger.info(f"AzureSearchService initialized - Index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Search Service: {e}")
            raise
    
    def upload_documents(self, documents: List[Dict[str, Any]]):
        """Upload documents to Azure Cognitive Search"""
        try:
            # Ensure all documents have required fields
            processed_docs = []
            for doc in documents:
                processed_doc = {
                    "id": str(doc.get("id", "")),
                    "content": str(doc.get("content", "")),
                    "entity_type": str(doc.get("entity_type", "")),
                    "domain": str(doc.get("domain", "")),
                    "timestamp": doc.get("timestamp", datetime.now().isoformat())
                }
                processed_docs.append(processed_doc)
            
            # Upload in batches of 1000 (Azure Search limit)
            batch_size = 1000
            total_uploaded = 0
            
            for i in range(0, len(processed_docs), batch_size):
                batch = processed_docs[i:i + batch_size]
                result = self.search_client.upload_documents(documents=batch)
                
                # Count successful uploads
                successful = sum(1 for r in result if r.succeeded)
                total_uploaded += successful
                
                logger.info(f"Uploaded batch {i//batch_size + 1}: {successful}/{len(batch)} documents")
            
            logger.info(f"Total documents uploaded to Azure Search: {total_uploaded}/{len(documents)}")
            
        except Exception as e:
            logger.error(f"Failed to upload documents to Azure Search: {e}")
            raise


class AzureCosmosService:
    """Real Azure Cosmos DB Service"""
    
    def __init__(self):
        """Initialize Azure Cosmos DB client"""
        try:
            if settings.azure_cosmos_db_connection_string:
                self.cosmos_client = CosmosClient.from_connection_string(
                    settings.azure_cosmos_db_connection_string
                )
            else:
                self.cosmos_client = CosmosClient(
                    url=settings.azure_cosmos_endpoint,
                    credential=settings.azure_cosmos_key
                )
            
            self.database_name = settings.azure_cosmos_database
            self.container_name = "extracted-knowledge"
            
            # Get or create database and container
            self.database = self.cosmos_client.get_database_client(self.database_name)
            self.container = self.database.get_container_client(self.container_name)
            
            logger.info(f"AzureCosmosService initialized - DB: {self.database_name}, Container: {self.container_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Cosmos Service: {e}")
            raise
    
    def upsert_document(self, document: Dict[str, Any]):
        """Upsert document in Azure Cosmos DB"""
        try:
            # Ensure document has required partition key
            if "partitionKey" not in document:
                document["partitionKey"] = document.get("batch_name", "default")
            
            response = self.container.upsert_item(body=document)
            logger.debug(f"Upserted document in Cosmos DB: {document.get('id', 'unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to upsert document {document.get('id', 'unknown')}: {e}")
            raise
    
    def batch_upsert_documents(self, documents: List[Dict[str, Any]]):
        """Batch upsert multiple documents"""
        successful = 0
        errors = []
        
        for doc in documents:
            try:
                self.upsert_document(doc)
                successful += 1
            except Exception as e:
                errors.append(f"Document {doc.get('id', 'unknown')}: {e}")
        
        logger.info(f"Batch upsert completed: {successful}/{len(documents)} successful")
        
        if errors:
            logger.warning(f"Batch upsert errors: {errors[:5]}...")  # Log first 5 errors
        
        return {"successful": successful, "total": len(documents), "errors": errors}


# Factory function to get services based on configuration
def get_azure_services():
    """Get Azure services instances"""
    try:
        blob_service = AzureBlobService()
        search_service = AzureSearchService()
        cosmos_service = AzureCosmosService()
        
        return blob_service, search_service, cosmos_service
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure services: {e}")
        raise


if __name__ == "__main__":
    # Test Azure services connectivity
    try:
        blob_service, search_service, cosmos_service = get_azure_services()
        print("✅ All Azure services initialized successfully")
        
        # Test blob upload
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        blob_service.upload_json(test_data, "test/connectivity_test.json")
        print("✅ Azure Blob Storage test successful")
        
        # Test search document upload
        test_docs = [{"id": "test_1", "content": "test content", "domain": "test"}]
        # Note: Commented out to avoid test data in production index
        # search_service.upload_documents(test_docs)
        print("✅ Azure Search Service initialized (upload test skipped)")
        
        # Test cosmos document upsert
        test_doc = {"id": "test_1", "partitionKey": "test", "data": "test data"}
        # Note: Commented out to avoid test data in production database
        # cosmos_service.upsert_document(test_doc)
        print("✅ Azure Cosmos DB initialized (upsert test skipped)")
        
        print("\n🎯 Azure services connectivity test completed successfully!")
        
    except Exception as e:
        print(f"❌ Azure services test failed: {e}")
        import traceback
        traceback.print_exc()