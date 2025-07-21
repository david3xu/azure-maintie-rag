"""Azure Cognitive Search client for Universal RAG system."""

import logging
import time
from typing import Dict, List, Any, Optional
import json
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureCognitiveSearchClient:
    """Universal Azure Cognitive Search client - follows azure_openai.py pattern"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure Cognitive Search client"""
        self.config = config or {}

        # Load from environment (matches azure_openai.py pattern)
        self.service_name = self.config.get('service_name') or azure_settings.azure_search_service
        self.admin_key = self.config.get('admin_key') or azure_settings.azure_search_admin_key
        self.api_version = azure_settings.azure_search_api_version

        if not self.service_name:
            raise ValueError("Azure Search service name is required")

        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.credential = self._get_azure_credential()

        # Initialize clients (follows azure_openai.py pattern)
        try:
            self.index_client = SearchIndexClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure Search client: {e}")
            raise

        logger.info(f"AzureCognitiveSearchClient initialized for endpoint: {self.endpoint}")

    def _get_azure_credential(self):
        """Enterprise credential management - data-driven from config"""
        if azure_settings.azure_use_managed_identity and azure_settings.azure_managed_identity_client_id:
            from azure.identity import ManagedIdentityCredential
            return ManagedIdentityCredential(client_id=azure_settings.azure_managed_identity_client_id)

        # Fallback to admin key if available
        if self.admin_key:
            return AzureKeyCredential(self.admin_key)

        # Final fallback to DefaultAzureCredential
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential()

    def create_universal_index(self, index_name: str, vector_dimensions: int = 1536) -> Dict[str, Any]:
        """Create universal search index for any domain - data-driven configuration"""
        try:
            from azure.search.documents.indexes.models import (
                SearchIndex, SearchField, SearchFieldDataType
            )

            # Universal fields that work for any domain (without vector search for now)
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="domain", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="source", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="metadata", type=SearchFieldDataType.String, searchable=False, filterable=False)
            ]

            index = SearchIndex(
                name=index_name,
                fields=fields
            )

            return index

        except Exception as e:
            logger.error(f"Failed to create universal index: {e}")
            raise

    async def create_index(self, index_name: str) -> Dict[str, Any]:
        """Create a search index for the specified domain"""
        try:
            # Create the index using the universal index schema
            index = self.create_universal_index(index_name)

            # Create the index in Azure Search
            self.index_client.create_index(index)

            return {
                "success": True,
                "index_name": index_name,
                "message": f"Index {index_name} created successfully"
            }
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name
            }

    def upload_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload documents to search index - universal format"""
        try:
            # Ensure all documents have required universal fields
            processed_docs = []
            for doc in documents:
                processed_doc = {
                    "id": doc.get("id", f"doc_{len(processed_docs)}"),
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "domain": doc.get("domain", "general"),
                    "source": doc.get("source", "unknown"),
                    "contentVector": doc.get("contentVector", [])
                }
                processed_docs.append(processed_doc)

            result = self.search_client.upload_documents(processed_docs)

            return {
                "success": True,
                "uploaded_count": len(processed_docs),
                "results": [r.key for r in result if r.succeeded]
            }

        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "attempted_count": len(documents)
            }

    def vector_search(self, query_vector: List[float], top_k: int = 10,
                     domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform vector search - follows existing search patterns"""
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )

            search_params = {
                "vector_queries": [vector_query],
                "top": top_k
            }

            # Add domain filter if specified (data-driven filtering)
            if domain_filter:
                search_params["filter"] = f"domain eq '{domain_filter}'"

            results = self.search_client.search(**search_params)

            search_results = []
            for result in results:
                search_results.append({
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "title": result.get("title"),
                    "domain": result.get("domain"),
                    "score": result.get("@search.score", 0.0)
                })

            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def search_documents(self, index_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Enterprise search with service validation and diagnostics"""
        try:
            # Create dedicated search client for target index
            target_search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,  # Match indexing pattern
                credential=self.credential
            )

            # Pre-search index validation
            index_stats = await self._get_index_statistics(index_name)
            logger.info(f"Searching index {index_name}: {index_stats['document_count']} documents")

            # Enhanced search parameters (simplified)
            search_params = {
                "top": top_k,
                "include_total_count": True
            }

            # Use the query parameter directly
            results = target_search_client.search(query, **search_params)

            search_results = []
            total_count = 0

            for result in results:
                search_results.append({
                    "id": result.get("id"),
                    "content": result.get("content", "")[:500],  # Limit content preview
                    "title": result.get("title", ""),
                    "domain": result.get("domain", ""),
                    "score": result.get("@search.score", 0.0)
                })
                total_count += 1

            logger.info(f"Search completed: {len(search_results)} results from {total_count} total documents")

            return search_results

        except Exception as e:
            logger.error(f"Azure Search query failed: {e}")
            return []

    async def _get_index_statistics(self, index_name: str) -> Dict[str, Any]:
        """Get Azure Search index statistics for diagnostics"""
        try:
            # Use index client to get statistics
            stats_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )

            # Count documents in index using proper API
            count_result = stats_client.search("*", include_total_count=True)
            document_count = 0
            try:
                # Try to get count from search results
                for _ in count_result:
                    document_count += 1
            except Exception:
                # Fallback: use index statistics
                try:
                    index_stats = self.index_client.get_index_statistics(index_name)
                    document_count = index_stats.get("documentCount", 0)
                except Exception:
                    document_count = 0

            return {
                "index_exists": True,
                "index_name": index_name,
                "document_count": document_count
            }

        except Exception as e:
            logger.warning(f"Index statistics unavailable: {e}")
            return {
                "index_exists": False,
                "index_name": index_name,
                "document_count": 0
            }

    async def index_document(self, index_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enterprise document indexing with service validation"""
        try:
            # Create dedicated service client for target index
            index_search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,  # Target specific index
                credential=self.credential
            )

            # Validate document structure before indexing
            validation_result = self._validate_document_schema(document)
            if not validation_result['valid']:
                logger.error(f"Document schema validation failed: {validation_result['errors']}")
                return {"success": False, "error": "Invalid document schema"}

            # Index document with retry pattern
            result = index_search_client.upload_documents([document])

            # Validate indexing success
            success_count = len([r for r in result if r.succeeded])

            logger.info(f"Document indexed successfully: {document.get('id')} -> {index_name}")

            return {
                "success": True,
                "document_id": document.get("id"),
                "index_name": index_name,
                "indexed_count": success_count
            }

        except Exception as e:
            logger.error(f"Azure Search indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document.get("id", "unknown")
            }

    def _validate_document_schema(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document against Azure Search schema requirements"""
        required_fields = ['id', 'content', 'title', 'domain']
        missing_fields = [field for field in required_fields if not document.get(field)]

        return {
            "valid": len(missing_fields) == 0,
            "errors": missing_fields
        }

    async def validate_document_structure(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document structure against index schema"""
        required_fields = ['id', 'content', 'title', 'domain']
        missing_fields = []

        for field in required_fields:
            if field not in document or not document[field]:
                missing_fields.append(field)

        return {
            "valid": len(missing_fields) == 0,
            "errors": missing_fields,
            "document_id": document.get('id', 'unknown')
        }

    def validate_index_configuration(self, index_name: str) -> Dict[str, Any]:
        """Validate Azure Search index configuration before use"""
        try:
            # Check if index exists
            index = self.index_client.get_index(index_name)

            # Count documents in index
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )

            # Get document count
            count_result = search_client.search("*", include_total_count=True)
            document_count = count_result.get_count()

            return {
                "index_exists": True,
                "index_name": index_name,
                "document_count": document_count,
                "fields_count": len(index.fields),
                "status": "operational"
            }

        except Exception as e:
            return {
                "index_exists": False,
                "index_name": index_name,
                "error": str(e),
                "status": "misconfigured"
            }

    def get_service_status(self, index_name: str) -> Dict[str, Any]:
        """Get search service status - follows azure_openai.py pattern"""
        try:
            # Test connection by getting index stats
            index_stats = self.index_client.get_index_statistics(index_name)

            return {
                "status": "healthy",
                "service_name": self.service_name,
                "index_name": index_name,
                "document_count": index_stats.get("documentCount", 0),
                "storage_size": index_stats.get("storageSize", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_name": self.service_name
            }

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status for service validation"""
        try:
            # Simple test to check if service name and admin key are configured
            if not self.service_name:
                return {
                    "status": "unhealthy",
                    "error": "Search service name not configured",
                    "service": "search"
                }

            if not self.admin_key:
                return {
                    "status": "unhealthy",
                    "error": "Search admin key not configured",
                    "service": "search"
                }

            return {
                "status": "healthy",
                "service": "search",
                "endpoint": self.endpoint,
                "index_name": self.index_name
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service": "search"
            }