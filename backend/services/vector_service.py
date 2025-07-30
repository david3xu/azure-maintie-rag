"""
Vector Service
High-level service for vector operations using Azure OpenAI embeddings
Integrates core embedding service with the infrastructure and data services
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

from core.azure_openai.embedding import AzureEmbeddingService
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class VectorService:
    """High-level vector service for integration with the system"""
    
    def __init__(self, infrastructure_service=None):
        """Initialize vector service"""
        self.infrastructure = infrastructure_service
        self._embedding_client = None
    
    @property
    def embedding_client(self) -> AzureEmbeddingService:
        """Lazy-loaded Azure embedding client"""
        if not self._embedding_client:
            self._embedding_client = AzureEmbeddingService()
        return self._embedding_client
    
    async def test_connectivity(self) -> Dict[str, Any]:
        """Test Azure OpenAI embedding service connectivity"""
        try:
            result = await self.embedding_client.test_embedding_connectivity()
            return result
        except Exception as e:
            logger.error(f"Embedding connectivity test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'service': 'EmbeddingService'
            }
    
    async def generate_embeddings_for_documents(self, documents: List[Dict[str, Any]], 
                                               content_field: str = 'content') -> Dict[str, Any]:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of document dictionaries
            content_field: Field name containing text to embed
            
        Returns:
            Dict with results and updated documents
        """
        try:
            # Extract text content from documents
            texts = []
            valid_docs = []
            
            for doc in documents:
                content = doc.get(content_field, '').strip()
                if content:
                    texts.append(content)
                    valid_docs.append(doc)
            
            if not texts:
                return {
                    'success': False,
                    'error': 'No valid text content found in documents',
                    'documents_processed': 0
                }
            
            logger.info(f"Generating embeddings for {len(texts)} documents")
            
            # Generate embeddings in batches
            embedding_result = await self.embedding_client.generate_embeddings_batch(texts)
            
            if not embedding_result.get('success'):
                return embedding_result
            
            embeddings = embedding_result['data']['embeddings']
            
            # Add embeddings to documents
            updated_documents = []
            for i, doc in enumerate(valid_docs):
                if i < len(embeddings) and embeddings[i] is not None:
                    doc_with_embedding = doc.copy()
                    doc_with_embedding['content_vector'] = embeddings[i]
                    updated_documents.append(doc_with_embedding)
                else:
                    logger.warning(f"No embedding generated for document {i}")
            
            return {
                'success': True,
                'documents_processed': len(updated_documents),
                'documents_with_embeddings': updated_documents,
                'embedding_stats': {
                    'total_texts': len(texts),
                    'successful_embeddings': embedding_result['data']['successful_count'],
                    'failed_embeddings': embedding_result['data']['failed_count'],
                    'dimensions': embedding_result['data']['dimensions'],
                    'model': embedding_result['data']['model']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0
            }
    
    async def add_embeddings_to_search_index(self, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Add vector embeddings to existing search index
        
        Args:
            domain: Domain to process
            
        Returns:
            Dict with results
        """
        try:
            if not self.infrastructure:
                raise RuntimeError("Infrastructure service required for search operations")
            
            search_service = self.infrastructure.search_service
            if not search_service:
                raise RuntimeError("Search service not available")
            
            # Get existing documents from search index
            from config.domain_patterns import DomainPatternManager
            index_name = DomainPatternManager.get_index_name(domain, azure_settings.azure_search_index)
            
            logger.info(f"Retrieving documents from index: {index_name}")
            
            search_result = await search_service.search_documents("*", top=1000)
            if not search_result.get('success'):
                raise RuntimeError(f"Failed to retrieve documents: {search_result.get('error')}")
            
            documents = search_result.get('data', {}).get('documents', [])
            logger.info(f"Found {len(documents)} documents to process")
            
            if not documents:
                return {
                    'success': True,
                    'message': 'No documents found to process',
                    'documents_processed': 0
                }
            
            # Generate embeddings for documents
            embedding_result = await self.generate_embeddings_for_documents(documents)
            
            if not embedding_result.get('success'):
                return embedding_result
            
            # Update documents with embeddings would require re-indexing
            # This is a complex operation that might need index schema updates
            
            return {
                'success': True,
                'message': 'Embeddings generated successfully',
                'documents_processed': embedding_result['documents_processed'],
                'embedding_stats': embedding_result['embedding_stats'],
                'note': 'Documents with embeddings ready for re-indexing'
            }
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to search index: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get vector service information"""
        return {
            'service': 'VectorService',
            'azure_client': 'AzureEmbeddingService',
            'embedding_model': 'text-embedding-ada-002',
            'dimensions': 1536,
            'max_batch_size': 100,
            'status': 'initialized' if self._embedding_client else 'not_initialized'
        }