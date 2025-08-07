"""
Azure OpenAI Embedding Service
Core service for generating vector embeddings using Azure OpenAI
Part of the unified core architecture for semantic search capabilities
"""

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from config.settings import azure_settings

from ..azure_auth.base_client import BaseAzureClient
from infrastructure.constants import AzureServiceLimits

logger = logging.getLogger(__name__)


class AzureEmbeddingService(BaseAzureClient):
    """Core service for Azure OpenAI embeddings with caching for performance"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embedding_cache = {}  # Simple in-memory cache for repeated texts

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination"""
        content = f"{text.strip()}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_default_endpoint(self) -> str:
        return azure_settings.azure_openai_endpoint

    def _get_default_key(self) -> str:
        return azure_settings.openai_api_key

    def _initialize_client(self):
        """Initialize Azure OpenAI client for embeddings"""
        if self.use_managed_identity:
            from infrastructure.azure_auth_utils import get_azure_credential

            credential = get_azure_credential()
            self._client = AzureOpenAI(
                azure_ad_token_provider=lambda: credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                ).token,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint,
            )
        else:
            self._client = AzureOpenAI(
                api_key=self.key,
                api_version=azure_settings.openai_api_version,
                azure_endpoint=self.endpoint,
            )

    async def generate_embedding(
        self, text: str, model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
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

            # Check cache first for performance optimization
            cache_key = self._get_cache_key(text, model)
            if cache_key in self._embedding_cache:
                cached_result = self._embedding_cache[cache_key]
                logger.debug(f"Cache hit for text (length: {len(text)})")
                return self.create_success_response(
                    "generate_embedding",
                    {**cached_result, "cache_hit": True},
                )

            response = await asyncio.to_thread(
                self._client.embeddings.create, model=model, input=text.strip()
            )

            embedding = response.data[0].embedding

            # Cache the result for future use
            result_data = {
                "embedding": embedding,
                "dimensions": len(embedding),
                "model": model,
                "text_length": len(text),
                "tokens_used": response.usage.total_tokens
                if hasattr(response, "usage")
                else None,
                "cache_hit": False,
            }

            # Store in cache (limit cache size to prevent memory issues)
            if len(self._embedding_cache) < AzureServiceLimits.DEFAULT_EMBEDDING_CACHE_SIZE_THRESHOLD:  # Limit cache size
                self._embedding_cache[cache_key] = {
                    k: v for k, v in result_data.items() if k != "cache_hit"
                }

            return self.create_success_response("generate_embedding", result_data)

        except Exception as e:
            return self.handle_azure_error("generate_embedding", e)

    def _health_check(self) -> bool:
        """Perform embedding service health check"""
        try:
            self.ensure_initialized()
            return self._client is not None
        except Exception:
            return False

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        batch_size: int = AzureServiceLimits.DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> Dict[str, Any]:
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
                batch = texts[i : i + batch_size]
                batch_texts = [text.strip() for text in batch if text.strip()]

                if not batch_texts:
                    continue

                try:
                    logger.info(
                        f"Processing embedding batch {i//batch_size + 1}: {len(batch_texts)} texts"
                    )

                    response = await asyncio.to_thread(
                        self._client.embeddings.create, model=model, input=batch_texts
                    )

                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)

                    if hasattr(response, "usage"):
                        total_tokens += response.usage.total_tokens

                    # Small delay between batches
                    await asyncio.sleep(0.1)

                except Exception as batch_error:
                    logger.error(f"Batch {i//batch_size + 1} failed: {batch_error}")
                    failed_count += len(batch_texts)
                    # Add empty embeddings for failed batch
                    all_embeddings.extend([None] * len(batch_texts))

            successful_embeddings = [emb for emb in all_embeddings if emb is not None]

            return self.create_success_response(
                "generate_embeddings_batch",
                {
                    "embeddings": all_embeddings,
                    "successful_count": len(successful_embeddings),
                    "failed_count": failed_count,
                    "total_texts": len(texts),
                    "dimensions": len(successful_embeddings[0])
                    if successful_embeddings
                    else 0,
                    "model": model,
                    "total_tokens": total_tokens,
                    "batches_processed": (len(texts) + batch_size - 1) // batch_size,
                },
            )

        except Exception as e:
            return self.handle_azure_error("generate_embeddings_batch", e)

    async def test_embedding_connectivity(
        self, model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Test Azure OpenAI embedding service connectivity

        Args:
            model: Embedding model to test

        Returns:
            Dict with connectivity test results
        """
        try:
            test_result = await self.generate_embedding("test connectivity", model)

            if test_result.get("success"):
                return self.create_success_response(
                    "test_embedding_connectivity",
                    {
                        "model": model,
                        "dimensions": test_result["data"]["dimensions"],
                        "connectivity": "verified",
                    },
                )
            else:
                return test_result

        except Exception as e:
            return self.handle_azure_error("test_embedding_connectivity", e)

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding service configuration"""
        return {
            "service": "AzureEmbeddingService",
            "endpoint": self.endpoint,
            "use_managed_identity": self.use_managed_identity,
            "api_version": azure_settings.openai_api_version,
            "default_model": "text-embedding-ada-002",
            "dimensions": 1536,
            "max_input_tokens": 8191,
        }
