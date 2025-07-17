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

    def chunk_text_for_embedding(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Chunk text to fit within embedding model token limits"""
        max_chars = max_tokens * 4  # Approximation: 1 token ≈ 4 characters

        if len(text) <= max_chars:
            return [text]

        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1

            if current_length + word_length > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with automatic chunking"""
        try:
            text_chunks = self.chunk_text_for_embedding(text, max_tokens=8000)

            if len(text_chunks) == 1:
                response = self.client.embeddings.create(
                    input=[text_chunks[0]],
                    model=self.embedding_deployment
                )
                return response.data[0].embedding
            else:
                chunk_embeddings = []
                for chunk in text_chunks:
                    if chunk.strip():
                        response = self.client.embeddings.create(
                            input=[chunk],
                            model=self.embedding_deployment
                        )
                        chunk_embeddings.append(response.data[0].embedding)

                if not chunk_embeddings:
                    raise ValueError("No valid chunks to embed")

                import numpy as np
                avg_embedding = np.mean(chunk_embeddings, axis=0)
                return avg_embedding.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed for text: {text[:100]}... Error: {e}")
            raise

    def build_index_universal(self, documents: Dict[str, UniversalDocument]) -> Dict[str, Any]:
        """Build FAISS index from universal documents"""
        logger.info(f"Building universal vector index for {len(documents)} documents in domain: {self.domain}")

        self.documents = documents
        doc_texts = []
        doc_ids = []

        # Extract text from universal documents
        for doc_id, doc in documents.items():
            full_text = f"{doc.title or ''} {doc.text}".strip()
            if full_text and len(full_text) > 10:
                doc_texts.append(full_text)
                doc_ids.append(doc_id)

        if not doc_texts:
            logger.error("No valid documents found for indexing")
            return {"success": False, "error": "No valid documents"}

        try:
            # Process documents individually to handle chunking
            all_embeddings = []

            for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts)):
                logger.info(f"Processing document {i+1}/{len(doc_texts)}: {doc_id}")

                try:
                    # Use text chunking to handle large documents
                    chunks = self.chunk_text_for_embedding(text, max_tokens=8000)

                    if len(chunks) == 1:
                        # Single chunk
                        embedding = self.get_embedding(chunks[0])
                    else:
                        # Multiple chunks - average embeddings
                        chunk_embeddings = []
                        for chunk in chunks:
                            if chunk.strip():
                                chunk_emb = self.get_embedding(chunk)
                                if isinstance(chunk_emb, list):
                                    chunk_emb = np.array(chunk_emb, dtype=np.float32)
                                chunk_embeddings.append(chunk_emb)

                        if chunk_embeddings:
                            embedding = np.mean(chunk_embeddings, axis=0)
                        else:
                            logger.warning(f"No valid chunks for document {doc_id}")
                            continue

                    # Ensure embedding is numpy array
                    if isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)

                    all_embeddings.append(embedding)
                    self.document_embeddings[doc_id] = embedding

                except Exception as e:
                    logger.error(f"Failed to process document {doc_id}: {e}")
                    continue

            if not all_embeddings:
                logger.error("No embeddings generated")
                return {"success": False, "error": "No embeddings generated"}

            # Create FAISS index
            embeddings_matrix = np.vstack(all_embeddings)

            # Fix: Ensure float32 type for FAISS
            embeddings_matrix = embeddings_matrix.astype(np.float32)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_matrix)

            # Create FAISS index
            dimension = embeddings_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.faiss_index.add(embeddings_matrix)

            # Update mappings
            self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids[:len(all_embeddings)])}
            self.index_to_doc_id = {i: doc_id for doc_id, i in self.doc_id_to_index.items()}

            # Save index
            self._save_index()

            logger.info(f"Universal FAISS index built successfully with {len(all_embeddings)} documents")
            return {
                "success": True,
                "total_documents": len(all_embeddings),
                "embedding_dimension": dimension,
                "index_type": "universal_vector_search"
            }

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

    def search_universal(self, query: str, top_k: int = 10) -> List[UniversalSearchResult]:
        """Universal semantic search across all documents"""
        if not self.faiss_index:
            logger.warning(f"No universal index available for domain: {self.domain}")
            return []

        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)

            # Fix: Ensure embedding is numpy array before reshape
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            elif not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)

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