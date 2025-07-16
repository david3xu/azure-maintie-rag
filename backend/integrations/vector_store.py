"""Vector store integration for Universal RAG system."""

from typing import Dict, List, Any, Optional
import numpy as np


class VectorStoreClient:
    """Universal vector store client that works with any domain."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vector store client."""
        self.config = config or {}
        self.store_type = self.config.get('type', 'in_memory')
        self.vectors = {}  # Simple in-memory store for now

    def store_vector(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Store a vector with metadata."""
        self.vectors[id] = {
            'vector': np.array(vector),
            'metadata': metadata or {}
        }

    def search_similar(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        query_np = np.array(query_vector)
        results = []

        for id, data in self.vectors.items():
            similarity = np.dot(query_np, data['vector']) / (
                np.linalg.norm(query_np) * np.linalg.norm(data['vector'])
            )
            results.append({
                'id': id,
                'similarity': float(similarity),
                'metadata': data['metadata']
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def get_vector(self, id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID."""
        return self.vectors.get(id)

    def delete_vector(self, id: str) -> bool:
        """Delete vector by ID."""
        if id in self.vectors:
            del self.vectors[id]
            return True
        return False