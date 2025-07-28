"""
Azure service mocks for testing
Provides mock implementations of Azure services
"""

from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List, Optional
import asyncio


class MockAzureOpenAIClient:
    """Mock Azure OpenAI client for testing"""
    
    def __init__(self):
        self.calls = []
    
    async def generate_completion(self, prompt: str, max_tokens: int = 100, **kwargs) -> Dict[str, Any]:
        """Mock completion generation"""
        self.calls.append({"method": "generate_completion", "prompt": prompt})
        
        return {
            "content": f"Mock response for: {prompt[:50]}...",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(prompt.split()) + 10
            }
        }
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Mock embedding generation"""
        self.calls.append({"method": "generate_embeddings", "text": text})
        
        # Return mock 1536-dimensional vector
        return [0.1] * 1536


class MockAzureSearchClient:
    """Mock Azure Cognitive Search client for testing"""
    
    def __init__(self):
        self.indices = {}
        self.documents = {}
    
    async def create_index(self, index_name: str, schema: Dict[str, Any] = None) -> bool:
        """Mock index creation"""
        self.indices[index_name] = {"schema": schema or {}, "documents": {}}
        return True
    
    async def index_exists(self, index_name: str) -> bool:
        """Mock index existence check"""
        return index_name in self.indices
    
    async def index_document(self, index_name: str, document: Dict[str, Any]) -> bool:
        """Mock document indexing"""
        if index_name not in self.indices:
            return False
        
        doc_id = document.get("id", f"doc_{len(self.documents)}")
        if index_name not in self.documents:
            self.documents[index_name] = {}
        self.documents[index_name][doc_id] = document
        return True
    
    async def search_documents(self, index_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Mock document search"""
        if index_name not in self.documents:
            return []
        
        # Simple mock search - return all documents
        results = []
        for doc_id, doc in self.documents[index_name].items():
            results.append({
                "@search.score": 0.9,  # Mock score
                **doc
            })
        
        return results[:top_k]
    
    async def get_document_count(self, index_name: str) -> int:
        """Mock document count"""
        if index_name not in self.documents:
            return 0
        return len(self.documents[index_name])


class MockAzureStorageClient:
    """Mock Azure Storage client for testing"""
    
    def __init__(self):
        self.containers = {}
        self.blobs = {}
    
    async def create_container(self, container_name: str) -> bool:
        """Mock container creation"""
        self.containers[container_name] = {"created": True}
        self.blobs[container_name] = {}
        return True
    
    async def container_exists(self, container_name: str) -> bool:
        """Mock container existence check"""
        return container_name in self.containers
    
    async def upload_text(self, container_name: str, blob_name: str, content: str, **kwargs) -> bool:
        """Mock text upload"""
        if container_name not in self.containers:
            return False
        
        self.blobs[container_name][blob_name] = {
            "content": content,
            "content_type": kwargs.get("content_type", "text/plain"),
            "metadata": kwargs.get("metadata", {})
        }
        return True
    
    async def download_text(self, container_name: str, blob_name: str) -> Optional[str]:
        """Mock text download"""
        if container_name in self.blobs and blob_name in self.blobs[container_name]:
            return self.blobs[container_name][blob_name]["content"]
        return None
    
    async def list_blobs(self, container_name: str) -> List[str]:
        """Mock blob listing"""
        if container_name not in self.blobs:
            return []
        return list(self.blobs[container_name].keys())
    
    async def delete_container(self, container_name: str) -> bool:
        """Mock container deletion"""
        if container_name in self.containers:
            del self.containers[container_name]
            del self.blobs[container_name]
            return True
        return False


class MockAzureCosmosClient:
    """Mock Azure Cosmos DB client for testing"""
    
    def __init__(self):
        self.databases = {}
        self.containers = {}
        self.documents = {}
    
    async def create_database(self, database_name: str) -> bool:
        """Mock database creation"""
        self.databases[database_name] = {"created": True}
        return True
    
    async def database_exists(self, database_name: str) -> bool:
        """Mock database existence check"""
        return database_name in self.databases
    
    async def create_container(self, database_name: str, container_name: str) -> bool:
        """Mock container creation"""
        if database_name not in self.databases:
            return False
        
        key = f"{database_name}/{container_name}"
        self.containers[key] = {"created": True}
        self.documents[key] = {}
        return True
    
    async def container_exists(self, database_name: str, container_name: str) -> bool:
        """Mock container existence check"""
        key = f"{database_name}/{container_name}"
        return key in self.containers
    
    async def add_entity(self, entity: Dict[str, Any], domain: str) -> bool:
        """Mock entity addition"""
        key = f"rag-metadata-{domain}/entities"
        if key not in self.documents:
            self.documents[key] = {}
        
        entity_id = entity.get("id", f"entity_{len(self.documents[key])}")
        self.documents[key][entity_id] = entity
        return True
    
    async def count_entities(self, database_name: str, container_name: str) -> int:
        """Mock entity count"""
        key = f"{database_name}/{container_name}"
        if key not in self.documents:
            return 0
        return len(self.documents[key])
    
    async def count_relationships(self, database_name: str, container_name: str) -> int:
        """Mock relationship count - simplified"""
        return 0  # Simplified for testing


class MockInfrastructureService:
    """Mock infrastructure service for testing"""
    
    def __init__(self):
        self.openai_client = MockAzureOpenAIClient()
        self.search_service = MockAzureSearchClient()
        self.cosmos_client = MockAzureCosmosClient()
        self.storage_client = MockAzureStorageClient()
        self.ml_client = Mock()
        self.app_insights = Mock()
    
    def get_service(self, service_name: str):
        """Get mock service by name"""
        services = {
            "openai": self.openai_client,
            "search": self.search_service,
            "cosmos": self.cosmos_client,
            "storage": self.storage_client,
            "ml": self.ml_client
        }
        return services.get(service_name)
    
    def get_rag_storage_client(self):
        """Get mock RAG storage client"""
        return self.storage_client
    
    def check_all_services_health(self) -> Dict[str, Any]:
        """Mock health check"""
        return {
            "overall_status": "healthy",
            "services": {
                "openai": {"status": "healthy"},
                "search": {"status": "healthy"},
                "cosmos": {"status": "healthy"},
                "storage": {"status": "healthy"}
            },
            "summary": {
                "total_services": 4,
                "healthy_services": 4,
                "unhealthy_services": 0
            }
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Mock configuration validation"""
        return {
            "valid": True,
            "all_configured": True,
            "errors": [],
            "warnings": []
        }


def create_mock_infrastructure():
    """Create a complete mock infrastructure for testing"""
    return MockInfrastructureService()


def create_async_mock(return_value=None):
    """Create an async mock that returns a value"""
    mock = AsyncMock()
    if return_value is not None:
        mock.return_value = return_value
    return mock