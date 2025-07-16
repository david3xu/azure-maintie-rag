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
import re

from core.models.maintenance_models import MaintenanceEntity, MaintenanceDocument
from core.knowledge.data_transformer import MaintIEDataTransformer
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

    def _get_document_text(self, document) -> str:
        """Get text content from document regardless of structure"""
        if isinstance(document, dict):
            return document.get('text', '') or document.get('content', '')
        return getattr(document, 'text', '') or getattr(document, 'content', '')

    def _get_document_title(self, document) -> str:
        """Get title from document regardless of structure"""
        if isinstance(document, dict):
            return document.get('title', '')
        return getattr(document, 'title', '')

    def _extract_document_entities(self, document) -> set:
        """Extract entities from document content with robust matching"""
        entities = set()
        document_text = self._get_document_text(document)
        title = self._get_document_title(document)
        full_text = f"{title} {document_text}".lower()
        if self.data_transformer and hasattr(self.data_transformer, 'entities'):
            for entity_id, entity in self.data_transformer.entities.items():
                entity_text = entity.text.lower().strip()
                # Enhanced text matching with word boundaries
                if re.search(r'\b' + re.escape(entity_text) + r'\b', full_text):
                    entities.add(entity.text)
        return entities

    def build_index(self, force_rebuild: bool = False) -> dict:
        """Build entity-document index from transformer data with optimizations"""
        import time
        start_time = time.time()
        if not force_rebuild and self._load_from_cache():
            print(f"[build_index] Loaded from cache in {time.time() - start_time:.2f}s")
            return self._get_index_stats()
        logger.info("Building entity-document index...")
        print("[build_index] Starting index build...")
        if not self.data_transformer:
            logger.error("Data transformer not provided to EntityDocumentIndex!")
            raise ValueError("Data transformer not provided")
        self.entity_to_docs.clear()
        self.doc_to_entities.clear()
        self.entity_frequency.clear()
        documents_processed = 0
        entities_processed = 0
        try:
            # Process documents and their entities
            if hasattr(self.data_transformer, 'documents'):
                total_docs = len(self.data_transformer.documents)
                print(f"[build_index] Processing {total_docs} documents...")
                for doc_id, document in self.data_transformer.documents.items():
                    documents_processed += 1
                    if documents_processed % 100 == 0:
                        print(f"[build_index] Processed {documents_processed}/{total_docs} documents...")
                    doc_entities = self._extract_document_entities(document)
                    for entity_text in doc_entities:
                        entity_key = entity_text.lower().strip()
                        self.entity_to_docs[entity_key].add(doc_id)
                        self.doc_to_entities[doc_id].add(entity_key)
                        self.entity_frequency[entity_key] += 1
                        entities_processed += 1
            print(f"[build_index] Finished document/entity pass. Entities processed: {entities_processed}")
            # Process explicit entity-document relationships from entity metadata
            if hasattr(self.data_transformer, 'entities'):
                total_entities = len(self.data_transformer.entities)
                print(f"[build_index] Processing {total_entities} entities for explicit relationships...")
                for entity_id, entity in self.data_transformer.entities.items():
                    entity_key = entity.text.lower().strip()
                    if hasattr(entity, 'metadata') and entity.metadata:
                        doc_id = entity.metadata.get('doc_id')
                        if doc_id:
                            self.entity_to_docs[entity_key].add(doc_id)
                            self.doc_to_entities[doc_id].add(entity_key)
                            self.entity_frequency[entity_key] += 1
                print(f"[build_index] Finished explicit entity-document relationships.")
            self.index_built = True
            self._save_to_cache()
            stats = {
                "documents_processed": documents_processed,
                "entities_processed": entities_processed,
                "unique_entities": len(self.entity_to_docs),
                "entity_document_pairs": sum(len(docs) for docs in self.entity_to_docs.values()),
                "index_built": True
            }
            elapsed = time.time() - start_time
            print(f"[build_index] Index built in {elapsed:.2f}s. Stats: {stats}")
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

    def _find_documents_for_entity(self, entity: MaintenanceEntity) -> Set[str]:
        """Find documents containing the entity"""
        containing_docs = set()

        if (self.data_transformer and
            hasattr(self.data_transformer, 'documents')):

            for doc_id, document in self.data_transformer.documents.items():
                # Safe attribute access
                title = getattr(document, 'title', '') or ''
                text = getattr(document, 'text', '') or getattr(document, 'content', '') or ''
                content_text = f"{title} {text}".lower()

                # Enhanced matching with word boundaries
                import re
                entity_text = entity.text.lower().strip()
                if re.search(r'\b' + re.escape(entity_text) + r'\b', content_text):
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