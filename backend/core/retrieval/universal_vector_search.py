"""
Universal Vector Search Module
Replaces MaintenanceVectorSearch with domain-agnostic semantic search
Works with any document type and content through universal models
"""

import os
import json
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import faiss
from openai import AzureOpenAI

from core.models.universal_models import UniversalSearchResult, UniversalDocument
from config.settings import settings

logger = logging.getLogger(__name__)


class UniversalVectorSearch:
    """Universal semantic vector search that works with any domain and document type"""

    def __init__(self, domain: str = "general", model_name: Optional[str] = None):
        """Initialize universal vector search with Azure OpenAI embedding model"""
        self.domain = domain
        self.api_key = settings.openai_api_key
        self.embedding_deployment = settings.embedding_deployment_name
        self.api_base = settings.embedding_api_base
        self.api_version = settings.embedding_api_version
        self.embedding_model = self.embedding_deployment

        # Universal data structures (no domain assumptions)
        self.faiss_index = None
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, UniversalDocument] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        self.index_to_doc_id: Dict[int, str] = {}

        # Azure OpenAI embedding client setup
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base
        )

        # Universal index paths (domain-specific but not hardcoded)
        self.index_dir = settings.indices_dir / self.domain
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index if available
        self._load_existing_index()

        logger.info(f"UniversalVectorSearch initialized for domain: {self.domain}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Azure OpenAI (universal - works with any text)"""
        try:
            # Clean and validate text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided for embedding")

            # Azure OpenAI embedding request
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_deployment
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed for text: {text[:100]}... Error: {e}")
            raise RuntimeError(f"Embedding retrieval failed: {e if e else 'Unknown error'}") from e

    def build_index_universal(self, documents: Dict[str, UniversalDocument]) -> Dict[str, Any]:
        """Build FAISS index from universal documents"""
        logger.info(f"Building universal vector index for {len(documents)} documents in domain: {self.domain}")

        self.documents = documents
        doc_texts = []
        doc_ids = []

        # Extract text from universal documents (works with any content)
        for doc_id, doc in documents.items():
            # Universal text extraction - works with any document structure
            full_text = f"{doc.title or ''} {doc.text}".strip()

            if full_text and len(full_text) > 10:
                doc_texts.append(full_text)
                doc_ids.append(doc_id)
            else:
                logger.warning(f"Skipping document {doc_id} with empty or too short text")

        if not doc_texts:
            logger.error("No valid documents found for indexing")
            return {"success": False, "error": "No valid documents"}

        try:
            # Batch processing respecting Azure OpenAI limits
            max_azure_batch = 2048
            configured_batch = getattr(settings, 'embedding_batch_size', 32)
            batch_size = min(configured_batch, max_azure_batch)

            logger.info(f"Processing {len(doc_texts)} documents in batches of {batch_size}")

            all_embeddings = []
            for i in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[i:i + batch_size]
                batch_doc_ids = doc_ids[i:i + batch_size]

                logger.info(f"Processing batch {i//batch_size + 1}/{(len(doc_texts) + batch_size - 1)//batch_size}")

                # Get embeddings for batch
                batch_embeddings = []
                for text in batch_texts:
                    embedding = self.get_embedding(text)
                    batch_embeddings.append(embedding)

                all_embeddings.extend(batch_embeddings)

                # Store individual embeddings
                for doc_id, embedding in zip(batch_doc_ids, batch_embeddings):
                    self.document_embeddings[doc_id] = embedding

                # Add small delay between batches
                if i + batch_size < len(doc_texts):
                    time.sleep(0.1)

            # Build FAISS index
            embeddings_matrix = np.array(all_embeddings)

            # Create FAISS index
            embedding_dim = embeddings_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine similarity)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            self.faiss_index.add(embeddings_matrix)

            # Update mappings
            self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
            self.index_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}

            # Save index
            self._save_index()

            result = {
                "success": True,
                "documents_indexed": len(doc_ids),
                "embedding_dimension": embedding_dim,
                "domain": self.domain,
                "index_type": "FAISS_IndexFlatIP"
            }

            logger.info(f"Universal vector index built successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"Universal index building failed: {e}")
            return {"success": False, "error": str(e)}

    def build_index_from_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """
        Build FAISS index from document list (Universal Solution)

        This method works with any document format and domain.
        Replaces the MaintIE-specific version with universal processing.

        Args:
            documents: List of dictionaries with 'doc_id', 'content', 'title', 'metadata'

        Returns:
            Dictionary with indexing results and success status
        """
        try:
            logger.info(f"Building universal FAISS index from {len(documents)} document dictionaries")

            # Convert dictionaries to UniversalDocument objects
            universal_docs = {}
            for doc_dict in documents:
                doc_id = doc_dict.get('doc_id', f'doc_{len(universal_docs)}')
                content = doc_dict.get('content', doc_dict.get('text', ''))
                title = doc_dict.get('title', f'Document {doc_id}')
                metadata = doc_dict.get('metadata', {})

                # Add domain information to metadata
                metadata['domain'] = self.domain
                metadata['processing_method'] = 'universal_vector_search'

                # Create UniversalDocument object
                universal_doc = UniversalDocument(
                    doc_id=doc_id,
                    text=content,
                    title=title,
                    metadata=metadata
                )
                universal_docs[doc_id] = universal_doc

            # Use universal build_index method
            return self.build_index_universal(universal_docs)

        except Exception as e:
            logger.error(f"Universal document indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_message": "Universal RAG indexing encountered an error"
            }

    def search_universal(self, query: str, top_k: int = 5) -> List[UniversalSearchResult]:
        """Universal semantic search that works with any domain"""

        if not self.faiss_index:
            logger.warning("No index available for search")
            return []

        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))

            # Convert to universal search results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                doc_id = self.index_to_doc_id.get(idx)
                if doc_id and doc_id in self.documents:
                    document = self.documents[doc_id]

                    result = UniversalSearchResult(
                        doc_id=doc_id,
                        content=document.text,
                        score=float(score),
                        metadata={
                            **document.metadata,
                            "title": document.title,
                            "domain": self.domain,
                            "search_method": "universal_vector_search"
                        },
                        entities=document.get_entity_texts() if hasattr(document, 'get_entity_texts') else [],
                        source="universal_vector_index"
                    )
                    results.append(result)

            logger.info(f"Universal search returned {len(results)} results for domain: {self.domain}")
            return results

        except Exception as e:
            logger.error(f"Universal search failed: {e}")
            return []

    def _save_index(self):
        """Save universal index to disk"""
        try:
            # Save FAISS index
            if self.faiss_index:
                index_path = self.index_dir / "universal_faiss.index"
                faiss.write_index(self.faiss_index, str(index_path))

            # Save metadata
            metadata = {
                "domain": self.domain,
                "document_count": len(self.documents),
                "doc_id_to_index": self.doc_id_to_index,
                "index_to_doc_id": self.index_to_doc_id,
                "embedding_model": self.embedding_model,
                "index_type": "universal_vector_search"
            }

            metadata_path = self.index_dir / "universal_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save document embeddings
            embeddings_path = self.index_dir / "universal_embeddings.pkl"
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.document_embeddings, f)

            logger.info(f"Universal index saved for domain: {self.domain}")

        except Exception as e:
            logger.error(f"Failed to save universal index: {e}")

    def _load_existing_index(self):
        """Load existing universal index if available"""
        try:
            index_path = self.index_dir / "universal_faiss.index"
            metadata_path = self.index_dir / "universal_metadata.json"
            embeddings_path = self.index_dir / "universal_embeddings.pkl"

            if all(path.exists() for path in [index_path, metadata_path, embeddings_path]):
                # Load FAISS index
                self.faiss_index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.doc_id_to_index = metadata.get("doc_id_to_index", {})
                self.index_to_doc_id = {int(k): v for k, v in metadata.get("index_to_doc_id", {}).items()}

                # Load embeddings
                with open(embeddings_path, 'rb') as f:
                    self.document_embeddings = pickle.load(f)

                logger.info(f"Loaded existing universal index for domain: {self.domain} with {len(self.doc_id_to_index)} documents")

        except Exception as e:
            logger.info(f"No existing universal index found for domain {self.domain}: {e}")

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get universal index statistics"""
        return {
            "domain": self.domain,
            "index_available": self.faiss_index is not None,
            "total_documents": len(self.documents),
            "total_embeddings": len(self.document_embeddings),
            "embedding_dimension": self.faiss_index.d if self.faiss_index else 0,
            "index_type": "universal_vector_search"
        }


# Legacy compatibility alias for gradual migration
MaintenanceVectorSearch = UniversalVectorSearch


def create_universal_vector_search(domain: str = "general") -> UniversalVectorSearch:
    """Factory function to create universal vector search"""
    return UniversalVectorSearch(domain)


if __name__ == "__main__":
    # Example usage
    search = UniversalVectorSearch("maintenance")

    # Sample documents
    docs = [
        {"doc_id": "doc1", "content": "Pump maintenance procedure", "title": "Pump Guide"},
        {"doc_id": "doc2", "content": "Motor troubleshooting steps", "title": "Motor Guide"}
    ]

    # Build index
    result = search.build_index_from_documents(docs)
    print(f"Index built: {result}")

    # Search
    results = search.search_universal("pump problem")
    print(f"Search results: {[r.doc_id for r in results]}")