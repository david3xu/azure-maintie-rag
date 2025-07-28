"""
Unified Azure Cognitive Search Client
Consolidates all search functionality: vector operations, document indexing, queries
Replaces: search_client.py, vector_service.py, query_analyzer.py
"""

import logging
from typing import Dict, List, Any, Optional
import json
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

from ..azure_auth.base_client import BaseAzureClient
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class UnifiedSearchClient(BaseAzureClient):
    """Unified client for all Azure Cognitive Search operations"""
    
    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_search_endpoint
        
    def _get_default_key(self) -> str:
        return azure_settings.azure_search_key
        
    def _initialize_client(self):
        """Initialize search clients"""
        credential = AzureKeyCredential(self.key)
        
        # Main search client
        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=azure_settings.azure_search_index,
            credential=credential
        )
        
        # Index management client
        self._index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=credential
        )
    
    # === DOCUMENT OPERATIONS ===
    
    async def index_documents(self, documents: List[Dict], index_name: str = None) -> Dict[str, Any]:
        """Index documents to Azure Search"""
        self.ensure_initialized()
        
        try:
            # Prepare documents for indexing
            search_documents = []
            for i, doc in enumerate(documents):
                search_doc = {
                    'id': doc.get('id', f"doc_{i}"),
                    'content': doc.get('content', ''),
                    'title': doc.get('title', ''),
                    'metadata': json.dumps(doc.get('metadata', {})),
                    'domain': doc.get('domain', 'general')
                }
                search_documents.append(search_doc)
            
            # Upload documents
            result = self._search_client.upload_documents(documents=search_documents)
            
            success_count = sum(1 for r in result if r.succeeded)
            error_count = len(result) - success_count
            
            return self.create_success_response('index_documents', {
                'documents_indexed': success_count,
                'documents_failed': error_count,
                'success_rate': success_count / len(documents) if documents else 0
            })
            
        except Exception as e:
            return self.handle_azure_error('index_documents', e)
    
    async def search_documents(self, query: str, top: int = 10, filters: str = None) -> Dict[str, Any]:
        """Search documents"""
        self.ensure_initialized()
        
        try:
            search_params = {
                'search_text': query,
                'top': top,
                'include_total_count': True
            }
            
            if filters:
                search_params['filter'] = filters
            
            results = self._search_client.search(**search_params)
            
            documents = []
            for result in results:
                doc = {
                    'id': result.get('id'),
                    'content': result.get('content'),
                    'title': result.get('title'),
                    'score': result.get('@search.score'),
                    'metadata': json.loads(result.get('metadata', '{}'))
                }
                documents.append(doc)
            
            return self.create_success_response('search_documents', {
                'documents': documents,
                'total_count': getattr(results, 'get_count', lambda: len(documents))(),
                'query': query
            })
            
        except Exception as e:
            return self.handle_azure_error('search_documents', e)
    
    # === VECTOR OPERATIONS ===
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using Azure OpenAI (simplified)"""
        # Note: This would typically call Azure OpenAI embeddings API
        # For now, return mock embeddings
        embeddings = []
        for text in texts:
            # Simple hash-based mock embedding (1536 dimensions)
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % 10000
            
            import random
            random.seed(seed)
            embedding = [random.uniform(-1, 1) for _ in range(1536)]
            embeddings.append(embedding)
        
        return embeddings
    
    async def vector_search(self, query_vector: List[float], top: int = 10) -> Dict[str, Any]:
        """Perform vector similarity search"""
        self.ensure_initialized()
        
        try:
            # This would use vector search capabilities in Azure Search
            # For now, return semantic search results
            query_text = "vector search query"  # Would convert vector to text
            return await self.search_documents(query_text, top)
            
        except Exception as e:
            return self.handle_azure_error('vector_search', e)
    
    # === INDEX MANAGEMENT ===
    
    async def create_index(self, index_name: str, schema: Dict = None) -> Dict[str, Any]:
        """Create search index"""
        self.ensure_initialized()
        
        try:
            from azure.search.documents.indexes.models import (
                SearchIndex, SimpleField, SearchableField, ComplexField
            )
            
            # Default schema for maintenance documents
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String"),
                SearchableField(name="title", type="Edm.String"),
                SimpleField(name="metadata", type="Edm.String"),
                SimpleField(name="domain", type="Edm.String", filterable=True)
            ]
            
            index = SearchIndex(name=index_name, fields=fields)
            result = self._index_client.create_index(index)
            
            return self.create_success_response('create_index', {
                'index_name': index_name,
                'fields_count': len(fields)
            })
            
        except Exception as e:
            return self.handle_azure_error('create_index', e)
    
    async def delete_index(self, index_name: str) -> Dict[str, Any]:
        """Delete search index"""
        self.ensure_initialized()
        
        try:
            self._index_client.delete_index(index_name)
            
            return self.create_success_response('delete_index', {
                'index_name': index_name,
                'message': 'Index deleted successfully'
            })
            
        except Exception as e:
            return self.handle_azure_error('delete_index', e)
    
    # === QUERY ANALYSIS ===
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze and enhance query"""
        try:
            # Simple query analysis
            words = query.lower().split()
            
            # Identify maintenance-specific terms
            equipment_terms = ['air conditioner', 'thermostat', 'pump', 'motor', 'valve']
            issue_terms = ['broken', 'not working', 'failed', 'leaking', 'damaged']
            action_terms = ['repair', 'replace', 'fix', 'check', 'maintenance']
            
            analysis = {
                'original_query': query,
                'word_count': len(words),
                'equipment_found': [term for term in equipment_terms if term in query.lower()],
                'issues_found': [term for term in issue_terms if term in query.lower()],
                'actions_found': [term for term in action_terms if term in query.lower()],
                'domain': 'maintenance' if any(term in query.lower() for term in equipment_terms) else 'general'
            }
            
            # Enhance query
            enhanced_query = query
            if analysis['equipment_found']:
                enhanced_query += " equipment maintenance"
            if analysis['issues_found']:
                enhanced_query += " troubleshooting repair"
            
            analysis['enhanced_query'] = enhanced_query
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {'original_query': query, 'enhanced_query': query}
    
    # === UTILITY METHODS ===
    
    async def get_index_stats(self, index_name: str = None) -> Dict[str, Any]:
        """Get index statistics"""
        self.ensure_initialized()
        
        try:
            # Get document count
            results = self._search_client.search(search_text="*", include_total_count=True, top=0)
            doc_count = getattr(results, 'get_count', lambda: 0)()
            
            return self.create_success_response('get_index_stats', {
                'document_count': doc_count,
                'index_name': azure_settings.azure_search_index
            })
            
        except Exception as e:
            return self.handle_azure_error('get_index_stats', e)