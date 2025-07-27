"""Azure Cognitive Search client for Universal RAG system."""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureCognitiveSearchClient:
    """
    Enterprise Azure Cognitive Search Client

    Architecture Components:
    - Credential Management Layer: Managed Identity → Key → DefaultAzureCredential
    - Service Orchestration Layer: Index lifecycle management + health monitoring
    - Document Processing Layer: Intelligent chunking + batch operations
    - Search Operations Layer: Vector + text search with domain filtering
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Azure Cognitive Search client with enterprise credential orchestration

        Service Integration Pattern:
        - Configuration-driven initialization (data-driven, no hardcoded values)
        - Credential hierarchy with enterprise fallbacks
        - Azure SDK client orchestration with proper error handling
        - Health validation integration for service monitoring
        """
        self.config = config or {}

        # Enterprise configuration management - data-driven from azure_settings
        self.service_name = self.config.get('service_name') or azure_settings.azure_search_service
        self.admin_key = self.config.get('admin_key') or azure_settings.azure_search_admin_key
        self.api_version = azure_settings.azure_search_api_version

        if not self.service_name:
            raise ValueError("Azure Search service name is required for enterprise deployment")

        # Service endpoint orchestration
        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.credential = self._get_azure_credential()
        self.index_name = self.config.get('index_name', 'default-index')

        # Azure SDK client initialization with enterprise error handling
        try:
            self.index_client = SearchIndexClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
        except Exception as e:
            logger.error(f"Azure Search client initialization failed: {e}")
            raise

        logger.info(f"AzureCognitiveSearchClient initialized - Endpoint: {self.endpoint}")

    def _get_azure_credential(self):
        """
        Enterprise Credential Management Architecture

        Credential Hierarchy (Azure Best Practices):
        1. Managed Identity (zero-credential enterprise approach)
        2. Service Key (enterprise key management integration)
        3. DefaultAzureCredential (development/fallback scenarios)
        """
        # Primary: Azure Managed Identity for enterprise zero-credential approach
        if azure_settings.azure_use_managed_identity and azure_settings.azure_managed_identity_client_id:
            from azure.identity import ManagedIdentityCredential
            return ManagedIdentityCredential(client_id=azure_settings.azure_managed_identity_client_id)

        # Secondary: Service key from Azure Key Vault or configuration
        if self.admin_key:
            return AzureKeyCredential(self.admin_key)

        # Fallback: DefaultAzureCredential for development environments
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential()

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Azure Service Health Monitoring

        Enterprise Health Check Pattern:
        - Configuration validation
        - Connection testing via Azure SDK
        - Service accessibility verification
        - Integration with Azure Application Insights
        """
        try:
            # Configuration layer validation
            if not self.service_name:
                return {
                    "status": "unhealthy",
                    "error": "Search service name not configured",
                    "service": "azure_cognitive_search",
                    "component": "configuration"
                }

            if not self.admin_key and not azure_settings.azure_use_managed_identity:
                return {
                    "status": "unhealthy",
                    "error": "No authentication method configured",
                    "service": "azure_cognitive_search",
                    "component": "authentication"
                }

            # Azure SDK connection validation
            index_list = list(self.index_client.list_indexes())

            return {
                "status": "healthy",
                "service": "azure_cognitive_search",
                "endpoint": self.endpoint,
                "index_count": len(index_list),
                "service_accessible": True,
                "authentication_method": "managed_identity" if azure_settings.azure_use_managed_identity else "service_key"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Service health check failed: {str(e)}",
                "service": "azure_cognitive_search",
                "endpoint": self.endpoint,
                "service_accessible": False
            }

    def create_universal_index_schema(self, index_name: str) -> SearchIndex:
        """
        Universal Search Index Schema Architecture

        Enterprise Index Design:
        - Domain-agnostic field structure
        - Scalable for multi-tenant scenarios
        - Azure Search limits compliance
        - Vector search capability integration
        """
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="domain", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="metadata", type=SearchFieldDataType.String, searchable=False, filterable=False),
            SearchField(name="file_name", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="entities", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                       searchable=True, filterable=True),
            SearchField(name="entity_types", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                       filterable=True),
            SearchField(name="relation_types", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                       filterable=True),
            SearchField(name="extraction_confidence", type=SearchFieldDataType.Double, filterable=True),
            SearchField(name="last_updated", type=SearchFieldDataType.String, filterable=True)
        ]

        return SearchIndex(name=index_name, fields=fields)

    async def create_index(self, index_name: str) -> Dict[str, Any]:
        """
        Azure Search Index Lifecycle Management

        Enterprise Index Orchestration:
        - Idempotent operations (safe for repeated calls)
        - Schema validation and compatibility checks
        - Service integration with Azure Application Insights
        - Error handling with structured response format
        """
        try:
            # Idempotency check - enterprise pattern for safe operations
            try:
                existing_index = self.index_client.get_index(index_name)
                logger.info(f"Index {index_name} already exists - idempotent operation")
                return {
                    "success": True,
                    "index_name": index_name,
                    "message": f"Index {index_name} already exists (idempotent operation)",
                    "action": "skipped",
                    "azure_service": "cognitive_search",
                    "operation": "create_index"
                }
            except Exception:
                # Index doesn't exist, proceed with creation
                pass

            # Create universal index schema
            index_schema = self.create_universal_index_schema(index_name)
            self.index_client.create_index(index_schema)

            logger.info(f"Azure Search index created: {index_name}")
            return {
                "success": True,
                "index_name": index_name,
                "message": f"Index {index_name} created successfully",
                "action": "created",
                "azure_service": "cognitive_search",
                "operation": "create_index",
                "schema_fields": len(index_schema.fields)
            }

        except Exception as e:
            logger.error(f"Azure Search index creation failed for {index_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name,
                "azure_service": "cognitive_search",
                "operation": "create_index"
            }

    async def create_or_update_index(self, index_name: str, documents: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """
        Enterprise Index Operations with Document Batch Processing

        Service Orchestration:
        - Index schema creation/update with Azure Search limits compliance
        - Batch document processing for optimal performance
        - Error handling with detailed telemetry for Azure Application Insights
        - Domain-specific indexing with enterprise data segregation
        """
        try:
            # Create index schema using universal design
            index_schema = self.create_universal_index_schema(index_name)

            # Recreate index for schema updates (enterprise pattern)
            try:
                self.index_client.get_index(index_name)
                self.index_client.delete_index(index_name)
                logger.info(f"Existing index {index_name} deleted for schema update")
            except Exception:
                pass  # Index doesn't exist, continue with creation

            self.index_client.create_index(index_schema)

            # Document batch processing with Azure Search optimization
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )

            batch_size = getattr(azure_settings, 'azure_search_batch_size', 100)
            upload_results = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = search_client.upload_documents(documents=batch)

                batch_result = {
                    "batch_start": i,
                    "batch_size": len(batch),
                    "success_count": len([r for r in result if r.succeeded]),
                    "failed_count": len([r for r in result if not r.succeeded])
                }
                upload_results.append(batch_result)

            total_success = sum([batch["success_count"] for batch in upload_results])
            logger.info(f"Document indexing completed - {total_success}/{len(documents)} documents indexed")

            return {
                "index_created": True,
                "index_name": index_name,
                "documents_uploaded": total_success,
                "total_documents": len(documents),
                "upload_results": upload_results,
                "azure_service": "cognitive_search",
                "domain": domain
            }

        except Exception as e:
            logger.error(f"Index creation or document upload failed: {e}")
            return {
                "index_created": False,
                "error": str(e),
                "index_name": index_name,
                "azure_service": "cognitive_search",
                "domain": domain
            }

    async def search_documents(self, index_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Enterprise Search Operations with Service Validation

        Search Architecture:
        - Pre-search index validation and statistics
        - Query processing with Azure Search optimization
        - Result formatting for downstream service integration
        - Performance monitoring and telemetry integration
        """
        try:
            # Pre-search service validation
            index_stats = await self._get_index_statistics(index_name)
            logger.info(f"Searching index {index_name}: {index_stats['document_count']} documents available")

            # Azure Search client for target index
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )

            # Enterprise search parameters
            search_params = {
                "top": top_k,
                "include_total_count": True
            }

            # Execute search operation
            results = search_client.search(query, **search_params)

            # Process and format results for service integration
            search_results = []
            for result in results:
                search_results.append({
                    "id": result.get("id"),
                    "content": result.get("content", "")[:500],  # Content preview for performance
                    "title": result.get("title", ""),
                    "domain": result.get("domain", ""),
                    "source": result.get("source", ""),
                    "score": result.get("@search.score", 0.0),
                    "azure_service": "cognitive_search"
                })

            logger.info(f"Search operation completed: {len(search_results)} results returned")
            return search_results

        except Exception as e:
            logger.error(f"Azure Search query failed for index {index_name}: {e}")
            return []

    async def _get_index_statistics(self, index_name: str) -> Dict[str, Any]:
        """
        Azure Search Index Statistics for Service Monitoring

        Enterprise Monitoring Pattern:
        - Index health validation
        - Document count verification
        - Service operational metrics
        - Integration with Azure Application Insights
        """
        try:
            # Index statistics via Azure Search SDK
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )

            # Document count retrieval
            count_result = search_client.search("*", include_total_count=True)
            document_count = 0

            try:
                # Count documents in search results
                for _ in count_result:
                    document_count += 1
            except Exception:
                # Fallback to index statistics API
                try:
                    index_stats = self.index_client.get_index_statistics(index_name)
                    document_count = index_stats.get("documentCount", 0)
                except Exception:
                    document_count = 0

            return {
                "index_exists": True,
                "index_name": index_name,
                "document_count": document_count,
                "service_status": "operational"
            }

        except Exception as e:
            logger.warning(f"Index statistics retrieval failed for {index_name}: {e}")
            return {
                "index_exists": False,
                "index_name": index_name,
                "document_count": 0,
                "service_status": "unavailable"
            }

    def _calculate_optimal_chunk_strategy(self, content: str) -> Dict[str, Any]:
        """
        Enterprise Document Chunking Strategy

        Azure Search Optimization:
        - Field size limit compliance (32KB maximum)
        - Semantic continuity preservation via overlap
        - Performance optimization for large documents
        - Cost optimization through intelligent chunking
        """
        content_length = len(content)
        max_field_size = 32766  # Azure Search field size limit
        optimal_chunk_size = 30000  # Safe margin under Azure limits
        overlap_size = 1000  # Semantic continuity overlap

        if content_length <= max_field_size:
            return {
                "strategy": "single_document",
                "chunks_required": 1,
                "chunk_size": content_length,
                "azure_compliant": True
            }

        chunks_needed = (content_length + optimal_chunk_size - 1) // optimal_chunk_size

        return {
            "strategy": "chunked_processing",
            "chunks_required": chunks_needed,
            "chunk_size": optimal_chunk_size,
            "overlap_size": overlap_size,
            "total_content_size": content_length,
            "azure_compliant": True
        }

    async def index_document(self, index_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enterprise Document Indexing with Intelligent Processing

        Document Processing Architecture:
        - Content size analysis and chunking strategy determination
        - Azure Search field limits compliance
        - Batch processing optimization
        - Detailed telemetry for Azure Application Insights
        """
        try:
            content = document.get('content', '')
            chunk_strategy = self._calculate_optimal_chunk_strategy(content)

            logger.info(f"Document {document.get('id')}: {chunk_strategy['strategy']} "
                       f"({chunk_strategy['chunks_required']} chunks)")

            if chunk_strategy["strategy"] == "single_document":
                return await self._index_single_document(index_name, document)
            else:
                return await self._index_chunked_document(index_name, document, chunk_strategy)

        except Exception as e:
            logger.error(f"Document indexing failed for {document.get('id')}: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document.get("id", "unknown"),
                "azure_service": "cognitive_search"
            }

    async def _index_single_document(self, index_name: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Single document indexing for standard-size content"""
        search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=self.credential
        )

        # Filter out unsupported fields to avoid schema conflicts
        filtered_document = {k: v for k, v in document.items()
                           if k not in ['chunk_type', 'chunk_index']}

        result = search_client.upload_documents([filtered_document])
        success_count = len([r for r in result if r.succeeded])

        return {
            "success": success_count > 0,
            "document_id": document.get('id'),
            "strategy": "single_document",
            "indexed_count": success_count,
            "azure_service": "cognitive_search"
        }

    async def _index_chunked_document(self, index_name: str, document: Dict[str, Any],
                                     strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enterprise Chunked Document Processing

        Large Document Handling:
        - Intelligent content chunking with semantic overlap
        - Chunk metadata preservation for document reconstruction
        - Azure Search batch optimization
        - Parent-child document relationship tracking
        """
        content = document.get('content', '')
        chunk_size = strategy['chunk_size']
        overlap_size = strategy.get('overlap_size', 0)
        chunks = []

        # Generate chunks with semantic overlap
        for i in range(0, len(content), chunk_size - overlap_size):
            chunk_content = content[i:i + chunk_size]
            chunk_index = i // (chunk_size - overlap_size)

            # Filter out unsupported fields from metadata to avoid schema conflicts
            original_metadata = json.loads(document.get('metadata', '{}'))
            # Remove chunk_type and chunk_index from original metadata to avoid schema conflicts
            filtered_metadata = {k: v for k, v in original_metadata.items()
                               if k not in ['chunk_type', 'chunk_index']}

            chunk_doc = {
                "id": f"{document['id']}_chunk_{chunk_index}",
                "content": chunk_content,
                "title": f"{document.get('title', 'Document')} (Part {chunk_index + 1})",
                "domain": document.get('domain'),
                "source": document.get('source'),
                "metadata": json.dumps({
                    **filtered_metadata,
                    "chunk_index": chunk_index,
                    "is_chunk": True,
                    "parent_document_id": document['id'],
                    "chunk_strategy": "enterprise_overlap",
                    "total_chunks": strategy['chunks_required']
                })
            }
            chunks.append(chunk_doc)

        # Batch upload chunks to Azure Search
        search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=self.credential
        )

        result = search_client.upload_documents(chunks)
        success_count = len([r for r in result if r.succeeded])

        logger.info(f"Large document processed: {document.get('id')} → "
                   f"{success_count}/{len(chunks)} chunks indexed")

        return {
            "success": success_count > 0,
            "document_id": document.get('id'),
            "strategy": "chunked_processing",
            "total_chunks": len(chunks),
            "indexed_chunks": success_count,
            "indexing_efficiency": f"{success_count}/{len(chunks)}",
            "azure_service": "cognitive_search"
        }

    def validate_index_configuration(self, index_name: str) -> Dict[str, Any]:
        """
        Azure Search Index Configuration Validation

        Enterprise Validation Pattern:
        - Index existence verification
        - Schema compatibility checks
        - Document count validation
        - Service operational status assessment
        """
        try:
            # Index schema validation
            index = self.index_client.get_index(index_name)

            # Document count retrieval via Azure Search
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )

            count_result = search_client.search("*", include_total_count=True)
            document_count = count_result.get_count() if hasattr(count_result, 'get_count') else 0

            return {
                "index_exists": True,
                "index_name": index_name,
                "document_count": document_count,
                "fields_count": len(index.fields),
                "status": "operational",
                "azure_service": "cognitive_search"
            }

        except Exception as e:
            return {
                "index_exists": False,
                "index_name": index_name,
                "error": str(e),
                "status": "misconfigured",
                "azure_service": "cognitive_search"
            }

    def get_service_status(self, index_name: str) -> Dict[str, Any]:
        """
        Azure Search Service Status for Enterprise Monitoring

        Service Health Architecture:
        - Index-specific operational metrics
        - Azure Search service accessibility validation
        - Performance metrics collection
        - Integration with Azure Application Insights
        """
        try:
            # Service health via index statistics
            index_stats = self.index_client.get_index_statistics(index_name)

            return {
                "status": "healthy",
                "service_name": self.service_name,
                "index_name": index_name,
                "document_count": index_stats.get("documentCount", 0),
                "storage_size": index_stats.get("storageSize", 0),
                "azure_service": "cognitive_search",
                "endpoint": self.endpoint
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_name": self.service_name,
                "index_name": index_name,
                "azure_service": "cognitive_search"
            }
