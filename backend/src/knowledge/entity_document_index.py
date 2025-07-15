"""
Entity-Document Index for O(1) entity lookups
Simple, professional implementation that builds on existing data_transformer.py
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

from src.models.maintenance_models import MaintenanceEntity, MaintenanceDocument
from src.knowledge.data_transformer import MaintIEDataTransformer
from config.settings import settings

logger = logging.getLogger(__name__)

class EntityDocumentIndex:
    """Fast entity-to-document and document-to-entity index for graph operations"""

    def __init__(self, data_transformer: Optional[MaintIEDataTransformer] = None):
        """Initialize with data transformer"""
        self.data_transformer = data_transformer

        # Index data structures
        self.entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self.doc_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self.entity_frequency: Dict[str, int] = defaultdict(int)

        # Caching
        self.cache_dir = settings.processed_data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_cache_path = self.cache_dir / "entity_document_index.pkl"

        self.index_built = False
        logger.info("EntityDocumentIndex initialized")

    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build entity-document index from transformer data"""

        # Check cache first
        if not force_rebuild and self._load_from_cache():
            return self._get_index_stats()

        logger.info("Building entity-document index...")

        if not self.data_transformer:
            logger.error("Data transformer not provided to EntityDocumentIndex!")
            raise ValueError("Data transformer not provided")

        # Clear existing data
        self.entity_to_docs.clear()
        self.doc_to_entities.clear()
        self.entity_frequency.clear()

        # Build index from transformer data
        documents_processed = 0
        entities_processed = 0

        try:
            # Process documents and their entities
            if hasattr(self.data_transformer, 'documents'):
                for doc_id, document in self.data_transformer.documents.items():
                    documents_processed += 1

                    # Extract entities from document content and metadata
                    doc_entities = self._extract_document_entities(document)

                    for entity_text in doc_entities:
                        # Normalize entity text
                        entity_key = entity_text.lower().strip()

                        # Update indices
                        self.entity_to_docs[entity_key].add(doc_id)
                        self.doc_to_entities[doc_id].add(entity_key)
                        self.entity_frequency[entity_key] += 1
                        entities_processed += 1

            # Process explicit entity-document relationships if available
            if hasattr(self.data_transformer, 'entities'):
                for entity_id, entity in self.data_transformer.entities.items():
                    entity_key = entity.text.lower().strip()

                    # Find documents containing this entity
                    containing_docs = self._find_documents_for_entity(entity)

                    for doc_id in containing_docs:
                        self.entity_to_docs[entity_key].add(doc_id)
                        self.doc_to_entities[doc_id].add(entity_key)

            self.index_built = True

            # Cache the results
            self._save_to_cache()

            stats = {
                "documents_processed": documents_processed,
                "entities_processed": entities_processed,
                "unique_entities": len(self.entity_to_docs),
                "entity_document_pairs": sum(len(docs) for docs in self.entity_to_docs.values()),
                "index_built": True
            }

            logger.info(f"Entity-document index built: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error building entity-document index: {e}")
            self.index_built = False
            return {"index_built": False, "error": str(e)}

    def check_index_status(self):
        """Diagnostic method to print index build status and stats"""
        print(f"Entity index built: {self.index_built}")
        stats = self._get_index_stats()
        print(f"Index stats: {stats}")

    def get_documents_for_entity(self, entity_text: str) -> List[str]:
        """Get document IDs containing the entity (O(1) lookup)"""
        if not self.index_built:
            logger.warning("Index not built, building now...")
            self.build_index()

        entity_key = entity_text.lower().strip()
        return list(self.entity_to_docs.get(entity_key, set()))

    def get_entities_for_document(self, doc_id: str) -> List[str]:
        """Get entities in the document (O(1) lookup)"""
        if not self.index_built:
            logger.warning("Index not built, building now...")
            self.build_index()

        return list(self.doc_to_entities.get(doc_id, set()))

    def get_entity_frequency(self, entity_text: str) -> int:
        """Get frequency of entity across all documents"""
        entity_key = entity_text.lower().strip()
        return self.entity_frequency.get(entity_key, 0)

    def get_popular_entities(self, min_frequency: int = 5) -> List[str]:
        """Get entities that appear frequently across documents"""
        return [entity for entity, freq in self.entity_frequency.items()
                if freq >= min_frequency]

    def _extract_document_entities(self, document: MaintenanceDocument) -> Set[str]:
        """Extract entities from document content"""
        entities = set()

        # Extract from content using simple tokenization
        content_text = f"{document.title} {document.content}".lower()

        # Use existing entity vocabulary if available
        if (self.data_transformer and
            hasattr(self.data_transformer, 'entities')):

            for entity_id, entity in self.data_transformer.entities.items():
                entity_text = entity.text.lower()
                if entity_text in content_text:
                    entities.add(entity.text)

        return entities

    def _find_documents_for_entity(self, entity: MaintenanceEntity) -> Set[str]:
        """Find documents containing the entity"""
        containing_docs = set()

        if (self.data_transformer and
            hasattr(self.data_transformer, 'documents')):

            for doc_id, document in self.data_transformer.documents.items():
                content_text = f"{document.title} {document.content}".lower()
                if entity.text.lower() in content_text:
                    containing_docs.add(doc_id)

        return containing_docs

    def _save_to_cache(self) -> None:
        """Save index to cache for fast loading"""
        try:
            cache_data = {
                "entity_to_docs": dict(self.entity_to_docs),
                "doc_to_entities": dict(self.doc_to_entities),
                "entity_frequency": dict(self.entity_frequency),
                "index_built": self.index_built
            }

            with open(self.index_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"Entity-document index cached to {self.index_cache_path}")

        except Exception as e:
            logger.error(f"Error caching index: {e}")

    def _load_from_cache(self) -> bool:
        """Load index from cache"""
        try:
            if not self.index_cache_path.exists():
                return False

            with open(self.index_cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.entity_to_docs = defaultdict(set, {
                k: set(v) for k, v in cache_data["entity_to_docs"].items()
            })
            self.doc_to_entities = defaultdict(set, {
                k: set(v) for k, v in cache_data["doc_to_entities"].items()
            })
            self.entity_frequency = defaultdict(int, cache_data["entity_frequency"])
            self.index_built = cache_data["index_built"]

            logger.info(f"Entity-document index loaded from cache")
            return True

        except Exception as e:
            logger.warning(f"Error loading index from cache: {e}")
            return False

    def _get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        return {
            "unique_entities": len(self.entity_to_docs),
            "unique_documents": len(self.doc_to_entities),
            "entity_document_pairs": sum(len(docs) for docs in self.entity_to_docs.values()),
            "index_built": self.index_built
        }