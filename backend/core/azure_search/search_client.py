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
        self.index_name = self.config.get('index_name') or azure_settings.azure_search_index
        self.api_version = azure_settings.azure_search_api_version

        if not self.service_name:
            raise ValueError("Azure Search service name is required")

        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.credential = self._get_azure_credential()

        # Initialize clients (follows azure_openai.py pattern)
        try:
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
            self.index_client = SearchIndexClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure Search client: {e}")
            raise

        logger.info(f"AzureCognitiveSearchClient initialized for index: {self.index_name}")

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

    def create_universal_index(self, vector_dimensions: int = 1536) -> Dict[str, Any]:
        """Create universal search index for any domain - data-driven configuration"""
        try:
            from azure.search.documents.indexes.models import (
                SearchIndex, SearchField, SearchFieldDataType, VectorSearch,
                VectorSearchProfile, VectorSearchAlgorithmConfiguration
            )

            # Universal fields that work for any domain
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="domain", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="source", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                           searchable=True, vector_search_dimensions=vector_dimensions,
                           vector_search_profile_name="universal-vector-profile")
            ]

            # Vector search configuration
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(
                    name="universal-vector-profile",
                    algorithm_configuration_name="universal-algorithm"
                )],
                algorithms=[VectorSearchAlgorithmConfiguration(
                    name="universal-algorithm",
                    kind="hnsw"
                )]
            )

            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )

            result = self.index_client.create_or_update_index(index)

            return {
                "success": True,
                "index_name": self.index_name,
                "fields_count": len(fields),
                "vector_dimensions": vector_dimensions
            }

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": self.index_name
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

    def get_service_status(self) -> Dict[str, Any]:
        """Get search service status - follows azure_openai.py pattern"""
        try:
            # Test connection by getting index stats
            index_stats = self.index_client.get_index_statistics(self.index_name)

            return {
                "status": "healthy",
                "service_name": self.service_name,
                "index_name": self.index_name,
                "document_count": index_stats.get("documentCount", 0),
                "storage_size": index_stats.get("storageSize", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_name": self.service_name
            }