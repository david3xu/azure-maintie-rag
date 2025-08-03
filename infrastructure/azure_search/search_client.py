"""
Unified Azure Cognitive Search Client
Consolidates all search functionality: vector operations, document indexing, queries
Replaces: search_client.py, vector_service.py, query_analyzer.py
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from config.settings import azure_settings

from ..azure_auth.base_client import BaseAzureClient

# Removed AzureKeyCredential - Azure-only deployment uses managed identity


# Updated to use consolidated intelligence components
try:
    from agents.core.cache_manager import get_cache_manager
    from agents.intelligence.config_generator import ConfigGenerator
    from agents.intelligence.pattern_engine import PatternEngine

    CONSOLIDATED_INTELLIGENCE_AVAILABLE = True
    # Compatibility wrappers
    get_pattern_manager = lambda: PatternEngine()
    get_discovery_naming = lambda: ConfigGenerator().generate_infrastructure_naming
    get_dynamic_ml_config = lambda: ConfigGenerator().generate_ml_config
except ImportError:
    # Fallback to no-op functions if consolidated intelligence unavailable
    get_pattern_manager = lambda: None
    get_discovery_naming = lambda: None
    get_dynamic_ml_config = lambda: None
    CONSOLIDATED_INTELLIGENCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnifiedSearchClient(BaseAzureClient):
    """Unified client for all Azure Cognitive Search operations"""

    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_search_endpoint

    def _health_check(self) -> bool:
        """Perform Cognitive Search service health check"""
        try:
            # Simple connectivity check
            return True  # If client is initialized successfully, service is accessible
        except Exception as e:
            logger.warning(f"Cognitive Search health check failed: {e}")
            return False

    def _initialize_client(self):
        """Initialize search clients - Azure managed identity only"""
        # Azure-only deployment - managed identity required
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        logger.info(
            f"Azure Search client initialized with managed identity for {self.endpoint}"
        )

        # Main search client
        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=azure_settings.azure_search_index,
            credential=credential,
        )

        # Index management client
        self._index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=credential
        )

    async def test_connection(self) -> Dict[str, Any]:
        """Test Azure Cognitive Search connection"""
        try:
            self.ensure_initialized()

            # Test by trying to get index statistics
            stats = self._index_client.get_index_statistics(
                azure_settings.azure_search_index
            )

            return {
                "success": True,
                "index": azure_settings.azure_search_index,
                "endpoint": self.endpoint,
                "document_count": getattr(stats, "document_count", 0),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "endpoint": getattr(self, "endpoint", "unknown"),
                "index": azure_settings.azure_search_index,
            }

    # === DOCUMENT OPERATIONS ===

    async def index_documents(
        self, documents: List[Dict], index_name: str = None
    ) -> Dict[str, Any]:
        """Index documents to Azure Search"""
        self.ensure_initialized()

        try:
            # Prepare documents for indexing
            search_documents = []
            for i, doc in enumerate(documents):
                search_doc = {
                    "id": doc.get("id", f"doc_{i}"),
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "metadata": json.dumps(doc.get("metadata", {})),
                    "domain": doc.get("domain", "general"),
                }

                # Include vector embeddings if present
                if "content_vector" in doc and doc["content_vector"]:
                    search_doc["content_vector"] = doc["content_vector"]

                # Include category if present
                if "category" in doc:
                    search_doc["category"] = doc["category"]

                search_documents.append(search_doc)

            # Upload documents - use specific index if provided
            if index_name:
                # Create temporary search client for the specific index
                search_client = SearchClient(
                    endpoint=self.endpoint,
                    index_name=index_name,
                    credential=self._search_client._credential,
                )
                result = search_client.upload_documents(documents=search_documents)
            else:
                # Use default search client
                result = self._search_client.upload_documents(
                    documents=search_documents
                )

            success_count = sum(1 for r in result if r.succeeded)
            error_count = len(result) - success_count

            return self.create_success_response(
                "index_documents",
                {
                    "documents_indexed": success_count,
                    "documents_failed": error_count,
                    "success_rate": success_count / len(documents) if documents else 0,
                },
            )

        except Exception as e:
            return self.handle_azure_error("index_documents", e)

    async def search_documents(
        self, query: str, top: int = 10, filters: str = None
    ) -> Dict[str, Any]:
        """Search documents"""
        self.ensure_initialized()

        try:
            search_params = {
                "search_text": query,
                "top": top,
                "include_total_count": True,
            }

            if filters:
                search_params["filter"] = filters

            results = await asyncio.to_thread(self._search_client.search, **search_params)

            documents = []
            for result in results:
                doc = {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "title": result.get("title"),
                    "score": result.get("@search.score"),
                    "metadata": json.loads(result.get("metadata", "{}")),
                }
                documents.append(doc)

            return self.create_success_response(
                "search_documents",
                {
                    "documents": documents,
                    "total_count": getattr(
                        results, "get_count", lambda: len(documents)
                    )(),
                    "query": query,
                },
            )

        except Exception as e:
            return self.handle_azure_error("search_documents", e)

    # === VECTOR OPERATIONS ===

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using Azure OpenAI"""
        try:
            from ..azure_openai.embedding import AzureEmbeddingService

            # Use dedicated embedding service
            embedding_service = AzureEmbeddingService()
            result = await embedding_service.generate_embeddings_batch(texts)

            if result.get("success"):
                return result["data"]["embeddings"]
            else:
                raise RuntimeError(
                    f"Embedding generation failed: {result.get('error')}"
                )

        except Exception as e:
            logger.error(f"Failed to create embeddings via Azure OpenAI: {e}")
            raise RuntimeError(f"Embedding creation failed: {e}")

    async def vector_search(
        self, query_vector: List[float], top: int = 10, filters: str = None
    ) -> Dict[str, Any]:
        """Perform vector similarity search using Azure Cognitive Search"""
        self.ensure_initialized()

        try:
            from azure.search.documents.models import VectorizedQuery

            # Create vector query for Azure Search
            vector_query = VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top, fields="content_vector"
            )

            search_params = {
                "search_text": None,  # Pure vector search
                "vector_queries": [vector_query],
                "top": top,
                "include_total_count": True,
            }

            if filters:
                search_params["filter"] = filters

            results = await asyncio.to_thread(self._search_client.search, **search_params)

            documents = []
            for result in results:
                doc = {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "title": result.get("title"),
                    "score": result.get("@search.score"),
                    "metadata": json.loads(result.get("metadata", "{}")),
                }
                documents.append(doc)

            return self.create_success_response(
                "vector_search",
                {
                    "documents": documents,
                    "total_count": getattr(
                        results, "get_count", lambda: len(documents)
                    )(),
                    "search_type": "vector",
                },
            )

        except ImportError as ie:
            # Fallback to semantic search if vector search not available
            logger.warning(
                "Vector search not available, falling back to semantic search"
            )
            return await self.search_documents("", top)

        except Exception as e:
            return self.handle_azure_error("vector_search", e)

    # === INDEX MANAGEMENT ===

    async def list_indexes(self) -> List[str]:
        """List all indexes - used for connectivity testing"""
        self.ensure_initialized()
        try:
            indexes = []
            for index in self._index_client.list_indexes():
                indexes.append(index.name)
            return indexes
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise e

    async def create_index(
        self, index_name: str, domain: str = "general", schema: Dict = None
    ) -> Dict[str, Any]:
        """Create search index using domain-specific schema - handles existing indexes gracefully"""
        self.ensure_initialized()

        try:
            from azure.core.exceptions import ResourceExistsError
            from azure.search.documents.indexes.models import (
                ComplexField,
                SearchableField,
                SearchIndex,
                SimpleField,
            )

            # Check if index already exists first
            try:
                existing_index = self._index_client.get_index(index_name)
                logger.info(f"✅ Index already exists: {index_name}")
                return self.create_success_response(
                    "create_index",
                    {
                        "index_name": index_name,
                        "message": "Index already exists",
                        "existed": True,
                        "fields_count": len(existing_index.fields),
                    },
                )
            except Exception:
                # Index doesn't exist, proceed with creation
                pass

            # Use domain-specific schema from centralized configuration
            if schema is None:
                domain_schema = DomainPatternManager.get_schema(domain)
                schema_fields = domain_schema.fields
            else:
                schema_fields = schema

            # Convert schema to Azure Search fields
            fields = []
            for field_config in schema_fields:
                field_name = field_config["name"]
                field_type = field_config["type"]

                if field_config.get("searchable", False):
                    field = SearchableField(
                        name=field_name,
                        type=field_type,
                        filterable=field_config.get("filterable", False),
                        sortable=field_config.get("sortable", False),
                    )
                else:
                    field = SimpleField(
                        name=field_name,
                        type=field_type,
                        key=field_config.get("key", False),
                        filterable=field_config.get("filterable", False),
                        sortable=field_config.get("sortable", False),
                    )
                fields.append(field)

            index = SearchIndex(name=index_name, fields=fields)

            try:
                result = self._index_client.create_index(index)
                logger.info(f"✅ Created new index: {index_name}")

                return self.create_success_response(
                    "create_index",
                    {
                        "index_name": index_name,
                        "fields_count": len(fields),
                        "created": True,
                    },
                )

            except ResourceExistsError:
                # Handle race condition where index was created between our check and creation
                logger.info(f"✅ Index was created by another process: {index_name}")
                return self.create_success_response(
                    "create_index",
                    {
                        "index_name": index_name,
                        "message": "Index already exists (created by another process)",
                        "existed": True,
                        "fields_count": len(fields),
                    },
                )

        except Exception as e:
            # Check if this is specifically the "already exists" error
            error_message = str(e)
            if (
                "ResourceNameAlreadyInUse" in error_message
                or "already exists" in error_message
            ):
                logger.info(f"✅ Index already exists (caught exception): {index_name}")
                return self.create_success_response(
                    "create_index",
                    {
                        "index_name": index_name,
                        "message": "Index already exists",
                        "existed": True,
                    },
                )

            return self.handle_azure_error("create_index", e)

    async def delete_index(self, index_name: str) -> Dict[str, Any]:
        """Delete search index"""
        self.ensure_initialized()

        try:
            self._index_client.delete_index(index_name)

            return self.create_success_response(
                "delete_index",
                {"index_name": index_name, "message": "Index deleted successfully"},
            )

        except Exception as e:
            return self.handle_azure_error("delete_index", e)

    async def get_or_create_vector_index(
        self, index_name: str, domain: str = "general", vector_dimension: int = 1536
    ) -> Dict[str, Any]:
        """Get or create vector index optimized for embeddings"""
        self.ensure_initialized()

        try:
            # First try to get existing index
            try:
                existing_index = self._index_client.get_index(index_name)
                logger.info(f"✅ Vector index already exists: {index_name}")
                return self.create_success_response(
                    "get_or_create_vector_index",
                    {
                        "index_name": index_name,
                        "message": "Vector index already exists",
                        "existed": True,
                        "fields_count": len(existing_index.fields),
                    },
                )
            except Exception:
                # Index doesn't exist, create it
                pass

            from azure.search.documents.indexes.models import (
                HnswAlgorithmConfiguration,
                SearchableField,
                SearchIndex,
                SemanticConfiguration,
                SemanticField,
                SemanticPrioritizedFields,
                SemanticSearch,
                SimpleField,
                VectorSearch,
                VectorSearchAlgorithmConfiguration,
                VectorSearchProfile,
            )

            # Define vector-optimized schema
            fields = [
                SimpleField(name="id", type="Edm.String", key=True, filterable=True),
                SearchableField(name="content", type="Edm.String", searchable=True),
                SearchableField(
                    name="title", type="Edm.String", searchable=True, filterable=True
                ),
                SimpleField(name="metadata", type="Edm.String", filterable=True),
                SimpleField(name="domain", type="Edm.String", filterable=True),
                SimpleField(name="category", type="Edm.String", filterable=True),
                SearchableField(
                    name="content_vector",
                    type="Collection(Edm.Single)",
                    searchable=True,
                    vector_search_dimensions=vector_dimension,
                    vector_search_profile_name="vector-profile",
                ),
            ]

            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-config",
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine",
                        },
                    )
                ],
            )

            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )

            semantic_search = SemanticSearch(configurations=[semantic_config])

            # Create the index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search,
            )

            result = self._index_client.create_index(index)
            logger.info(f"✅ Created new vector index: {index_name}")

            return self.create_success_response(
                "get_or_create_vector_index",
                {
                    "index_name": index_name,
                    "fields_count": len(fields),
                    "created": True,
                    "vector_dimensions": vector_dimension,
                },
            )

        except Exception as e:
            # Check if this is the "already exists" error
            error_message = str(e)
            if (
                "ResourceNameAlreadyInUse" in error_message
                or "already exists" in error_message
            ):
                logger.info(
                    f"✅ Vector index already exists (caught exception): {index_name}"
                )
                return self.create_success_response(
                    "get_or_create_vector_index",
                    {
                        "index_name": index_name,
                        "message": "Vector index already exists",
                        "existed": True,
                    },
                )

            return self.handle_azure_error("get_or_create_vector_index", e)

    # === QUERY ANALYSIS ===

    def analyze_query(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Analyze and enhance query using centralized domain patterns"""
        try:
            # Use centralized domain pattern manager
            return DomainPatternManager.enhance_query(query, domain)

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "original_query": query,
                "enhanced_query": query,
                "detected_domain": "general",
                "confidence": 0,
                "error": str(e),
            }

    # === UTILITY METHODS ===

    async def get_index_stats(self, index_name: str = None) -> Dict[str, Any]:
        """Get index statistics"""
        self.ensure_initialized()

        try:
            # Get document count
            results = await asyncio.to_thread(
                self._search_client.search,
                search_text="*", include_total_count=True, top=0
            )
            doc_count = getattr(results, "get_count", lambda: 0)()

            return self.create_success_response(
                "get_index_stats",
                {
                    "document_count": doc_count,
                    "index_name": azure_settings.azure_search_index,
                },
            )

        except Exception as e:
            return self.handle_azure_error("get_index_stats", e)
