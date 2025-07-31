"""
Azure OpenAI Embedding Service
Core service for generating vector embeddings using Azure OpenAI
Part of the unified core architecture for semantic search capabilities
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from openai import AzureOpenAI

from ..azure_auth.base_client import BaseAzureClient
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureEmbeddingService(BaseAzureClient):
    """Core service for Azure OpenAI embeddings"""
    
    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_openai_endpoint
        
    def _get_default_key(self) -> str:
        return azure_settings.openai_api_key
        
    def _initialize_client(self):
        """Initialize Azure OpenAI client for embeddings"""
        if self.use_managed_identity:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            self._client = AzureOpenAI(
                azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint
            )
        else:
            self._client = AzureOpenAI(
                api_key=self.key,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint
            )
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """
        Generate embedding for single text
        
        Args:
            text: Text to embed
            model: Embedding model name
            
        Returns:
            Dict with embedding vector and metadata
        """
        self.ensure_initialized()
        
        try:
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            response = self._client.embeddings.create(
                model=model,
                input=text.strip()
            )
            
            embedding = response.data[0].embedding
            
            return self.create_success_response('generate_embedding', {
                'embedding': embedding,
                'dimensions': len(embedding),
                'model': model,
                'text_length': len(text),
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            })
            
        except Exception as e:
            return self.handle_azure_error('generate_embedding', e)
    
    async def generate_embeddings_batch(self, texts: List[str], model: str = "text-embedding-ada-002", 
                                      batch_size: int = 100) -> Dict[str, Any]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            batch_size: Number of texts per batch
            
        Returns:
            Dict with list of embeddings and metadata
        """
        self.ensure_initialized()
        
        try:
            if not texts:
                raise ValueError("Texts list cannot be empty")
            
            all_embeddings = []
            total_tokens = 0
            failed_count = 0
            
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_texts = [text.strip() for text in batch if text.strip()]
                
                if not batch_texts:
                    continue
                
                try:
                    logger.info(f"Processing embedding batch {i//batch_size + 1}: {len(batch_texts)} texts")
                    
                    response = self._client.embeddings.create(
                        model=model,
                        input=batch_texts
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    if hasattr(response, 'usage'):
                        total_tokens += response.usage.total_tokens
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as batch_error:
                    logger.error(f"Batch {i//batch_size + 1} failed: {batch_error}")
                    failed_count += len(batch_texts)
                    # Add empty embeddings for failed batch
                    all_embeddings.extend([None] * len(batch_texts))
            
            successful_embeddings = [emb for emb in all_embeddings if emb is not None]
            
            return self.create_success_response('generate_embeddings_batch', {
                'embeddings': all_embeddings,
                'successful_count': len(successful_embeddings),
                'failed_count': failed_count,
                'total_texts': len(texts),
                'dimensions': len(successful_embeddings[0]) if successful_embeddings else 0,
                'model': model,
                'total_tokens': total_tokens,
                'batches_processed': (len(texts) + batch_size - 1) // batch_size
            })
            
        except Exception as e:
            return self.handle_azure_error('generate_embeddings_batch', e)
    
    async def test_embedding_connectivity(self, model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """
        Test Azure OpenAI embedding service connectivity
        
        Args:
            model: Embedding model to test
            
        Returns:
            Dict with connectivity test results
        """
        try:
            test_result = await self.generate_embedding("test connectivity", model)
            
            if test_result.get('success'):
                return self.create_success_response('test_embedding_connectivity', {
                    'model': model,
                    'dimensions': test_result['data']['dimensions'],
                    'connectivity': 'verified'
                })
            else:
                return test_result
                
        except Exception as e:
            return self.handle_azure_error('test_embedding_connectivity', e)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding service configuration"""
        return {
            'service': 'AzureEmbeddingService',
            'endpoint': self.endpoint,
            'use_managed_identity': self.use_managed_identity,
            'api_version': azure_settings.openai_api_version,
            'default_model': 'text-embedding-ada-002',
            'dimensions': 1536,
            'max_input_tokens': 8191
        }