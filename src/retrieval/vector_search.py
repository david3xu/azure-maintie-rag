"""
Vector-based semantic search module
Implements semantic similarity search using embeddings for maintenance documents
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.models.maintenance_models import SearchResult, MaintenanceDocument
from config.settings import settings


logger = logging.getLogger(__name__)


class MaintenanceVectorSearch:
    """Semantic vector search for maintenance documents"""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize vector search with embedding model"""
        self.model_name = model_name or settings.embedding_model
        self.embedding_model = None
        self.faiss_index = None
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, MaintenanceDocument] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        self.index_to_doc_id: Dict[int, str] = {}

        # Initialize model
        self._load_embedding_model()

        # Load existing index if available
        self._load_existing_index()

        logger.info(f"MaintenanceVectorSearch initialized with {self.model_name}")

    def _load_embedding_model(self) -> None:
        """Load sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _load_existing_index(self) -> None:
        """Load existing FAISS index and mappings if available"""
        try:
            index_dir = settings.indices_dir
            index_path = index_dir / "faiss_index.bin"
            mappings_path = index_dir / "doc_mappings.pkl"

            if index_path.exists() and mappings_path.exists():
                # Load FAISS index
                self.faiss_index = faiss.read_index(str(index_path))

                # Load document mappings
                with open(mappings_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self.doc_id_to_index = mappings['doc_id_to_index']
                    self.index_to_doc_id = mappings['index_to_doc_id']

                logger.info(f"Loaded existing index with {self.faiss_index.ntotal} documents")
            else:
                logger.info("No existing index found, will create new one")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")

    def build_index(self, documents: Dict[str, MaintenanceDocument]) -> None:
        """Build FAISS index from maintenance documents"""
        logger.info(f"Building vector index for {len(documents)} documents")

        self.documents = documents

        # Generate embeddings for all documents
        doc_texts = []
        doc_ids = []

        for doc_id, doc in documents.items():
            # Combine title and text for embedding
            full_text = f"{doc.title or ''} {doc.text}".strip()
            doc_texts.append(full_text)
            doc_ids.append(doc_id)

        # Generate embeddings in batches
        logger.info("Generating document embeddings...")
        from config.advanced_settings import advanced_settings

        embeddings = self.embedding_model.encode(
            doc_texts,
            batch_size=advanced_settings.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Create FAISS index
        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index with dimension {dimension}")

        # Use IndexFlatIP for cosine similarity (after normalization)
        self.faiss_index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        self.faiss_index.add(embeddings.astype(np.float32))

        # Create mappings
        self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.index_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}

        # Store embeddings
        for idx, doc_id in enumerate(doc_ids):
            self.document_embeddings[doc_id] = embeddings[idx]

        # Save index
        self._save_index()

        logger.info(f"Vector index built successfully with {self.faiss_index.ntotal} documents")

    def _save_index(self) -> None:
        """Save FAISS index and mappings to disk"""
        try:
            index_dir = settings.indices_dir
            index_dir.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = index_dir / "faiss_index.bin"
            faiss.write_index(self.faiss_index, str(index_path))

            # Save document mappings
            mappings_path = index_dir / "doc_mappings.pkl"
            mappings = {
                'doc_id_to_index': self.doc_id_to_index,
                'index_to_doc_id': self.index_to_doc_id
            }
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)

            # Save document embeddings
            embeddings_path = index_dir / "document_embeddings.pkl"
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.document_embeddings, f)

            logger.info("Vector index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for similar documents using vector similarity"""
        if not self.faiss_index or not self.embedding_model:
            logger.warning("Index or model not available for search")
            return []

        logger.info(f"Searching for: {query}")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search in FAISS index
            scores, indices = self.faiss_index.search(
                query_embedding.astype(np.float32),
                min(top_k, self.faiss_index.ntotal)
            )

            # Convert results to SearchResult objects
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue

                doc_id = self.index_to_doc_id.get(idx)
                if not doc_id or doc_id not in self.documents:
                    continue

                doc = self.documents[doc_id]

                result = SearchResult(
                    doc_id=doc_id,
                    title=doc.title or f"Document {doc_id}",
                    content=doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                    score=float(score),
                    source="vector",
                    metadata={
                        "embedding_model": self.model_name,
                        "similarity_type": "cosine",
                        "full_text_length": len(doc.text)
                    },
                    entities=doc.get_entity_texts()
                )

                results.append(result)

            logger.info(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def get_similarity_scores(self, query: str, doc_ids: List[str]) -> Dict[str, float]:
        """Get similarity scores for specific documents"""
        if not self.embedding_model:
            return {}

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            scores = {}
            for doc_id in doc_ids:
                if doc_id in self.document_embeddings:
                    doc_embedding = self.document_embeddings[doc_id].reshape(1, -1)
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding.T)[0, 0]
                    scores[doc_id] = float(similarity)

            return scores

        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            return {}

    def add_document(self, doc_id: str, document: MaintenanceDocument) -> bool:
        """Add a single document to the index"""
        try:
            # Generate embedding for new document
            full_text = f"{document.title or ''} {document.text}".strip()
            embedding = self.embedding_model.encode([full_text], convert_to_numpy=True)

            # Normalize embedding
            faiss.normalize_L2(embedding)

            # Add to FAISS index
            if self.faiss_index is None:
                # Create new index if none exists
                dimension = embedding.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)

            new_index = self.faiss_index.ntotal
            self.faiss_index.add(embedding.astype(np.float32))

            # Update mappings
            self.doc_id_to_index[doc_id] = new_index
            self.index_to_doc_id[new_index] = doc_id
            self.document_embeddings[doc_id] = embedding[0]
            self.documents[doc_id] = document

            logger.info(f"Added document {doc_id} to vector index")
            return True

        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False

    def update_document(self, doc_id: str, document: MaintenanceDocument) -> bool:
        """Update an existing document in the index"""
        # For simplicity, we'll remove and re-add the document
        # In production, consider more efficient update strategies
        try:
            if doc_id in self.doc_id_to_index:
                # Remove old document (mark as invalid)
                old_index = self.doc_id_to_index[doc_id]
                del self.doc_id_to_index[doc_id]
                del self.index_to_doc_id[old_index]
                del self.document_embeddings[doc_id]
                del self.documents[doc_id]

            # Add updated document
            return self.add_document(doc_id, document)

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific document"""
        return self.document_embeddings.get(doc_id)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query"""
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")

        embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding[0]

    def find_similar_documents(self, doc_id: str, top_k: int = 5) -> List[SearchResult]:
        """Find documents similar to a given document"""
        if doc_id not in self.document_embeddings:
            logger.warning(f"Document {doc_id} not found in embeddings")
            return []

        try:
            # Get document embedding
            doc_embedding = self.document_embeddings[doc_id].reshape(1, -1)

            # Search for similar documents
            scores, indices = self.faiss_index.search(
                doc_embedding.astype(np.float32),
                min(top_k + 1, self.faiss_index.ntotal)  # +1 to exclude self
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue

                similar_doc_id = self.index_to_doc_id.get(idx)
                if not similar_doc_id or similar_doc_id == doc_id:  # Skip self
                    continue

                if similar_doc_id in self.documents:
                    doc = self.documents[similar_doc_id]
                    result = SearchResult(
                        doc_id=similar_doc_id,
                        title=doc.title or f"Document {similar_doc_id}",
                        content=doc.text[:300] + "...",
                        score=float(score),
                        source="vector_similarity",
                        entities=doc.get_entity_texts()
                    )
                    results.append(result)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar documents for {doc_id}: {e}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        stats = {
            "total_documents": len(self.documents),
            "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "embedding_dimension": self.faiss_index.d if self.faiss_index else 0,
            "model_name": self.model_name,
            "index_type": type(self.faiss_index).__name__ if self.faiss_index else None
        }
        return stats


def create_vector_search(model_name: Optional[str] = None) -> MaintenanceVectorSearch:
    """Factory function to create vector search instance"""
    return MaintenanceVectorSearch(model_name)


def load_documents_from_processed_data() -> Dict[str, MaintenanceDocument]:
    """Load documents from processed data files"""
    try:
        documents_path = settings.processed_data_dir / "maintenance_documents.json"
        if not documents_path.exists():
            logger.warning(f"Documents file not found: {documents_path}")
            return {}

        import json
        with open(documents_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)

        documents = {}
        for doc_data in docs_data:
            doc = MaintenanceDocument(
                doc_id=doc_data["doc_id"],
                text=doc_data["text"],
                title=doc_data.get("title"),
                metadata=doc_data.get("metadata", {})
            )
            documents[doc.doc_id] = doc

        logger.info(f"Loaded {len(documents)} documents from processed data")
        return documents

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    vector_search = MaintenanceVectorSearch()

    # Load documents and build index
    documents = load_documents_from_processed_data()
    if documents:
        vector_search.build_index(documents)

        # Test search
        results = vector_search.search("pump seal failure", top_k=5)
        for result in results:
            print(f"Score: {result.score:.3f} - {result.title}")
            print(f"Content: {result.content[:200]}...")
            print()
