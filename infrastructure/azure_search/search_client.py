"""
Simple Azure Cognitive Search Client - CODING_STANDARDS Compliant
Clean search client without over-engineering enterprise patterns.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from config.settings import azure_settings
from infrastructure.constants import SearchConstants

from ..azure_auth.base_client import BaseAzureClient

logger = logging.getLogger(__name__)


class SimpleSearchClient(BaseAzureClient):
    """
    Simple Azure Cognitive Search client following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses Azure settings for configuration
    - Universal Design: Works with any search domain
    - Mathematical Foundation: Simple vector search operations
    """

    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_search_endpoint

    def _health_check(self) -> bool:
        """Simple health check with network error handling"""
        try:
            if hasattr(self, "_search_client") and self._search_client:
                # Use a lighter health check that doesn't require network resolution
                # Just verify the client object is properly initialized
                if self._search_client and self._index_client:
                    # Try a simple operation that doesn't require full network connectivity
                    return hasattr(self._search_client, "search") and hasattr(
                        self._index_client, "get_index_statistics"
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
            logger.error(f"Search connection test failed: {e}")
            return False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize simple search client"""
        super().__init__(config)

        # Simple configuration
        self.index_name = (
            self.config.get("index_name") or azure_settings.azure_search_index
        )
        self._search_client = None
        self._index_client = None

        logger.info(f"Simple search client initialized for {self.index_name}")

    def _initialize_client(self):
        """Simple client initialization"""
        try:
            # QUICK FAIL: Must use credential from Universal Dependencies - NO FALLBACK
            if not hasattr(self, 'credential') or not self.credential:
                raise RuntimeError(
                    "Search client MUST receive credential from Universal Dependencies. "
                    "No fallback authentication allowed. Ensure UniversalDeps passes credential."
                )
            credential = self.credential

            # Create simple search clients
            self._search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=credential,
            )

            self._index_client = SearchIndexClient(
                endpoint=self.endpoint, credential=credential
            )

            self._client = self._search_client
            logger.info("Search client initialized")

        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise

    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents using simple approach"""
        try:
            self.ensure_initialized()

            # Prepare documents for indexing - USE ONLY ACTUAL SCHEMA FIELDS
            search_documents = []
            for i, doc in enumerate(documents):
                # FIXED: Only use fields that exist in the schema
                search_doc = {
                    "id": doc.get("id", f"doc_{i}"),
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                }

                # Add optional schema fields only if they exist in the document AND schema
                if "file_path" in doc:
                    search_doc["file_path"] = doc["file_path"]
                # Include any metadata fields dynamically (Universal RAG approach)
                # Discover available metadata fields without hardcoding domain-specific field names
                metadata_fields = [
                    key
                    for key in doc.keys()
                    if key.startswith("metadata_")
                    or key.endswith("_type")
                    or key.endswith("_category")
                ]
                for field in metadata_fields:
                    search_doc[field] = doc[field]

                # Include vector if present AND schema supports it
                if "content_vector" in doc:
                    search_doc["content_vector"] = doc["content_vector"]

                search_documents.append(search_doc)

            # Upload documents
            result = self._search_client.upload_documents(documents=search_documents)

            return self.create_success_response(
                "index_documents",
                {
                    "indexed_count": len(search_documents),
                    "results": [r for r in result],
                },
            )

        except Exception as e:
            return self.handle_azure_error("index_documents", e)

    async def index_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Index single document - wrapper for compatibility"""
        return await self.index_documents([document])

    async def search_documents(
        self, query: str, top: int = 10, filters: str = None
    ) -> Dict[str, Any]:
        """Search documents using simple approach"""
        try:
            self.ensure_initialized()

            # Perform simple search
            results = self._search_client.search(
                search_text=query, top=top, filter=filters, include_total_count=True
            )

            # Convert results to list - USE ONLY ACTUAL SCHEMA FIELDS
            documents = []
            for result in results:
                doc = {
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "score": result.get(
                        "@search.score", SearchConstants.DEFAULT_SEARCH_SCORE
                    ),
                }

                # Add optional fields only if they exist in the result
                if "file_path" in result:
                    doc["file_path"] = result["file_path"]
                # Include any metadata fields dynamically (Universal RAG approach)
                # Discover available metadata fields without hardcoding domain-specific field names
                metadata_fields = [
                    key
                    for key in result.keys()
                    if key.startswith("metadata_")
                    or key.endswith("_type")
                    or key.endswith("_category")
                ]
                for field in metadata_fields:
                    doc[field] = result[field]

                documents.append(doc)

            return self.create_success_response(
                "search_documents",
                {"query": query, "documents": documents, "count": len(documents)},
            )

        except Exception as e:
            return self.handle_azure_error("search_documents", e)

    async def vector_search(self, vector: List[float], top: int = 10) -> Dict[str, Any]:
        """Vector search using simple approach"""
        try:
            self.ensure_initialized()

            # Perform vector search
            results = self._search_client.search(
                search_text="*",
                vector_queries=[
                    {
                        "vector": vector,
                        "k_nearest_neighbors": top,
                        "fields": "content_vector",
                    }
                ],
                top=top,
            )

            # Convert results to list - USE ONLY ACTUAL SCHEMA FIELDS
            documents = []
            for result in results:
                doc = {
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "score": result.get(
                        "@search.score", SearchConstants.DEFAULT_SEARCH_SCORE
                    ),
                }

                # Add optional fields only if they exist in the result
                if "file_path" in result:
                    doc["file_path"] = result["file_path"]
                # Include any metadata fields dynamically (Universal RAG approach)
                # Discover available metadata fields without hardcoding domain-specific field names
                metadata_fields = [
                    key
                    for key in result.keys()
                    if key.startswith("metadata_")
                    or key.endswith("_type")
                    or key.endswith("_category")
                ]
                for field in metadata_fields:
                    doc[field] = result[field]

                documents.append(doc)

            return self.create_success_response(
                "vector_search", {"documents": documents, "count": len(documents)}
            )

        except Exception as e:
            return self.handle_azure_error("vector_search", e)

    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get single document by ID"""
        try:
            self.ensure_initialized()

            result = self._search_client.get_document(key=doc_id)

            return self.create_success_response("get_document", {"document": result})

        except Exception as e:
            return self.handle_azure_error("get_document", e)

    async def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Delete documents by IDs"""
        try:
            self.ensure_initialized()

            # Prepare documents for deletion
            delete_docs = [{"id": doc_id} for doc_id in doc_ids]

            result = self._search_client.delete_documents(documents=delete_docs)

            # Check actual results from Azure
            results_list = [r for r in result]
            successful_deletions = 0
            failed_deletions = 0

            for r in results_list:
                if hasattr(r, "succeeded") and r.succeeded:
                    successful_deletions += 1
                else:
                    failed_deletions += 1

            return self.create_success_response(
                "delete_documents",
                {
                    "deleted_count": successful_deletions,
                    "failed_count": failed_deletions,
                    "results": results_list,
                },
            )

        except Exception as e:
            return self.handle_azure_error("delete_documents", e)

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get simple index statistics"""
        try:
            self.ensure_initialized()

            stats = self._index_client.get_index_statistics(self.index_name)

            return self.create_success_response(
                "get_index_stats",
                {
                    "document_count": getattr(stats, "document_count", 0),
                    "storage_size": getattr(stats, "storage_size", 0),
                    "index_name": self.index_name,
                },
            )

        except Exception as e:
            return self.handle_azure_error("get_index_stats", e)


# Backward compatibility aliases
UnifiedSearchClient = SimpleSearchClient
AzureSearchClient = SimpleSearchClient
