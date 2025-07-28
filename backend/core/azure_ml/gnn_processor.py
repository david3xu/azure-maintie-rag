"""
Universal GNN Data Processor
Replaces MaintIEGNNDataProcessor with domain-agnostic GNN data preparation
Works with AzureOpenAITextProcessor and universal models for any domain
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    torch = None
    Data = None

from ..azure_openai import AzureOpenAITextProcessor
from ..models.universal_rag_models import UniversalEntity, UniversalRelation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)


class AzureMLGNNProcessor:

    async def enhance_search_results(
        self,
        search_results: List[Dict[str, Any]],
        analysis_results: Dict[str, Any],
        knowledge_graph: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Enterprise GNN-powered search enhancement using pre-computed embeddings
        Integrates with Azure Cosmos DB for embedding retrieval and similarity computation
        """
        import numpy as np
        import logging
        logger = logging.getLogger(__name__)
        if not search_results:
            logger.info("No search results to enhance")
            return search_results
        try:
            enhancement_weight = 0.3
            vector_weight = 0.7
            query_entities = analysis_results.get("entities_detected", [])
            query_concepts = analysis_results.get("concepts_detected", [])
            if not query_entities and not query_concepts:
                logger.info("No entities or concepts detected for GNN enhancement")
                return search_results
            logger.info(f"Enhancing {len(search_results)} search results using {len(query_entities)} entities and {len(query_concepts)} concepts")
            query_embeddings = await self._retrieve_query_embeddings(query_entities + query_concepts)
            if not query_embeddings:
                logger.info("No GNN embeddings found for query entities")
                return search_results
            enhanced_results = []
            for result in search_results:
                try:
                    enhanced_result = result.copy()
                    doc_entities = result.get("entities", [])
                    doc_content = result.get("content", "")
                    if not doc_entities and doc_content:
                        doc_entities = await self._extract_entities_from_content(doc_content)
                    gnn_similarity = await self._calculate_gnn_similarity(query_embeddings, doc_entities)
                    original_score = result.get("score", 0.0)
                    enhanced_score = (
                        original_score * vector_weight +
                        gnn_similarity * enhancement_weight
                    )
                    enhanced_result.update({
                        "gnn_similarity": gnn_similarity,
                        "enhanced_score": enhanced_score,
                        "original_score": original_score,
                        "enhancement_method": "gnn_pre_computed",
                        "doc_entities": doc_entities,
                        "enhancement_weight": enhancement_weight
                    })
                    enhanced_results.append(enhanced_result)
                except Exception as e:
                    logger.warning(f"Failed to enhance result {result.get('doc_id', 'unknown')}: {e}")
                    enhanced_results.append(result)
            enhanced_results.sort(key=lambda x: x.get("enhanced_score", x.get("score", 0.0)), reverse=True)
            original_scores = [r.get("score", 0.0) for r in search_results]
            enhanced_scores = [r.get("enhanced_score", r.get("score", 0.0)) for r in enhanced_results]
            avg_improvement = (
                (sum(enhanced_scores) / len(enhanced_scores)) -
                (sum(original_scores) / len(original_scores))
            ) if enhanced_scores and original_scores else 0.0
            logger.info(f"GNN enhancement completed: average score improvement = {avg_improvement:.4f}")
            return enhanced_results
        except Exception as e:
            logger.error(f"GNN search enhancement failed: {e}")
            return search_results

    async def _retrieve_query_embeddings(self, entities: List[str]) -> Dict[str, Any]:
        """
        Retrieve pre-computed GNN embeddings from Azure Cosmos DB
        Uses batch querying for performance optimization
        """
        if not entities:
            return {}
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            cosmos_client = AzureCosmosGremlinClient()
            embeddings = {}
            batch_size = 10
            entity_batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
            for batch in entity_batches:
                try:
                    entity_filters = " or ".join([f"has('text', '{entity}')" for entity in batch])
                    batch_query = f"""
                        g.V().has('domain', '{self.domain}')
                            .where({entity_filters})
                            .by('gnn_embeddings')
                    """
                    batch_results = cosmos_client._execute_gremlin_query_safe(batch_query, timeout_seconds=30)
                    if batch_results:
                        for result in batch_results:
                            entity_text = result.get('text')
                            embedding = result.get('gnn_embeddings')
                            if entity_text and embedding is not None:
                                embeddings[entity_text] = embedding
                except Exception as e:
                    continue
            return embeddings
        except Exception as e:
            return {}

    async def _calculate_gnn_similarity(
        self,
        query_embeddings: Dict[str, Any],
        doc_entities: List[str]
    ) -> float:
        """
        Calculate similarity using pre-computed GNN embeddings
        Implements vectorized computation for performance
        """
        import numpy as np
        if not query_embeddings or not doc_entities:
            return 0.0
        try:
            doc_embeddings = await self._retrieve_query_embeddings(doc_entities)
            if not doc_embeddings:
                return 0.0
            query_vecs = list(query_embeddings.values())
            doc_vecs = list(doc_embeddings.values())
            if not query_vecs or not doc_vecs:
                return 0.0
            query_avg = np.mean(query_vecs, axis=0)
            doc_avg = np.mean(doc_vecs, axis=0)
            dot_product = np.dot(query_avg, doc_avg)
            query_norm = np.linalg.norm(query_avg)
            doc_norm = np.linalg.norm(doc_avg)
            if query_norm == 0 or doc_norm == 0:
                return 0.0
            similarity = dot_product / (query_norm * doc_norm)
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            return float(similarity)
        except Exception as e:
            return 0.0

    async def _extract_entities_from_content(self, content: str) -> List[str]:
        """
        Extract entities from document content using existing knowledge extractor
        Fallback method when entities are not available in search metadata
        """
        try:
            if not content or len(content.strip()) < 10:
                return []
            from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
            extractor = AzureOpenAIKnowledgeExtractor(self.domain)
            extraction_results = await extractor.extract_knowledge_from_texts([content], ["search_result"])
            if extraction_results.get("success", False):
                knowledge_data = extractor.get_extracted_knowledge()
                entities = list(knowledge_data.get("entities", {}).keys())
                entity_texts = []
                for entity_id, entity_data in knowledge_data.get("entities", {}).items():
                    entity_text = entity_data.get("text", "")
                    if entity_text:
                        entity_texts.append(entity_text)
                return entity_texts[:10]
            else:
                return []
        except Exception as e:
            return []
    """Convert universal text knowledge to GNN-ready format for any domain"""

    def __init__(self, text_processor: AzureOpenAITextProcessor, domain: str = "general"):
        """Initialize with universal text processor"""
        self.text_processor = text_processor
        self.domain = domain
        self.gnn_data_dir = settings.processed_data_dir / "gnn" / domain
        self.gnn_data_dir.mkdir(parents=True, exist_ok=True)

        # Universal entity and relation mappings (no domain assumptions)
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}
        self.entity_type_to_idx: Dict[str, int] = {}
        self.relation_type_to_idx: Dict[str, int] = {}

        # Universal node features (dynamic based on discovered types)
        self.entity_features: Optional[torch.Tensor] = None
        self.entity_types: Optional[torch.Tensor] = None

        # Universal edge data (dynamic based on discovered relations)
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr: Optional[torch.Tensor] = None
        self.edge_types: Optional[torch.Tensor] = None

        # Check PyTorch Geometric availability
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available, GNN processing will be limited")

        logger.info(f"AzureMLGNNProcessor initialized for domain: {domain}")

    def prepare_universal_gnn_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """Prepare complete GNN dataset from universal text knowledge"""

        cache_path = self.gnn_data_dir / "universal_gnn_dataset.pkl"

        if use_cache and cache_path.exists():
            logger.info("Loading universal GNN data from cache...")
            return self._load_cached_data(cache_path)

        logger.info(f"Preparing universal GNN data for domain: {self.domain}")

        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("PyTorch Geometric not available for GNN processing")
            return {"error": "PyTorch Geometric not installed"}

        try:
            # Step 1: Build universal entity and relation mappings
            self._build_universal_mappings()

            # Step 2: Create universal node features from discovered entity types
            self._create_universal_node_features()

            # Step 3: Create universal edge data from discovered relations
            self._create_universal_edge_data()

            # Step 4: Create PyTorch Geometric data object
            gnn_data = self._create_universal_torch_geometric_data()

            # Step 5: Create training/validation splits
            train_data, val_data, test_data = self._create_universal_data_splits(gnn_data)

            # Step 6: Prepare universal task-specific labels
            node_labels, edge_labels = self._create_universal_task_labels()

            dataset = {
                'domain': self.domain,
                'processing_method': 'universal_text_based',
                'full_data': gnn_data,
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'node_labels': node_labels,
                'edge_labels': edge_labels,
                'entity_to_idx': self.entity_to_idx,
                'idx_to_entity': self.idx_to_entity,
                'relation_to_idx': self.relation_to_idx,
                'idx_to_relation': self.idx_to_relation,
                'entity_types': list(self.entity_type_to_idx.keys()),
                'relation_types': list(self.relation_type_to_idx.keys()),
                'stats': self._get_universal_dataset_stats()
            }

            # Cache the results
            self._save_cached_data(dataset, cache_path)

            logger.info(f"Universal GNN dataset prepared for {self.domain}: {dataset['stats']}")
            return dataset

        except Exception as e:
            logger.error(f"Universal GNN data preparation failed: {e}")
            return {"error": str(e), "domain": self.domain}

    def _build_universal_mappings(self):
        """Build entity and relation mappings from universal text processor"""

        # Build entity mappings from universal entities
        entity_idx = 0
        if hasattr(self.text_processor, 'entities') and self.text_processor.entities:
            for entity_id, entity in self.text_processor.entities.items():
                self.entity_to_idx[entity_id] = entity_idx
                self.idx_to_entity[entity_idx] = entity_id

                # Map entity types
                entity_type = entity.entity_type
                if entity_type not in self.entity_type_to_idx:
                    self.entity_type_to_idx[entity_type] = len(self.entity_type_to_idx)

                entity_idx += 1

        # Build relation mappings from universal relations
        relation_idx = 0
        if hasattr(self.text_processor, 'relations') and self.text_processor.relations:
            for relation in self.text_processor.relations:
                relation_id = relation.relation_id
                self.relation_to_idx[relation_id] = relation_idx
                self.idx_to_relation[relation_idx] = relation_id

                # Map relation types
                relation_type = relation.relation_type
                if relation_type not in self.relation_type_to_idx:
                    self.relation_type_to_idx[relation_type] = len(self.relation_type_to_idx)

                relation_idx += 1

        logger.info(f"Universal mappings built: {len(self.entity_to_idx)} entities, {len(self.relation_to_idx)} relations")

    def _create_universal_node_features(self):
        """Create universal node features from discovered entity types"""

        if not self.entity_to_idx:
            logger.warning("No entities found for node feature creation")
            return

        try:
            num_entities = len(self.entity_to_idx)
            num_entity_types = len(self.entity_type_to_idx)

            # Create simple one-hot encoding for entity types
            # In production, this could use more sophisticated embeddings
            feature_dim = max(num_entity_types, 8)  # Minimum feature dimension

            # Initialize features
            self.entity_features = torch.zeros(num_entities, feature_dim)
            self.entity_types = torch.zeros(num_entities, dtype=torch.long)

            # Fill features based on entity types
            for entity_id, entity_idx in self.entity_to_idx.items():
                if entity_id in self.text_processor.entities:
                    entity = self.text_processor.entities[entity_id]
                    entity_type = entity.entity_type

                    if entity_type in self.entity_type_to_idx:
                        type_idx = self.entity_type_to_idx[entity_type]
                        self.entity_types[entity_idx] = type_idx

                        # One-hot encoding for entity type
                        if type_idx < feature_dim:
                            self.entity_features[entity_idx, type_idx] = 1.0

                        # Add confidence as additional feature
                        if feature_dim > num_entity_types:
                            confidence_idx = min(num_entity_types, feature_dim - 1)
                            self.entity_features[entity_idx, confidence_idx] = entity.confidence

            logger.info(f"Universal node features created: {self.entity_features.shape}")

        except Exception as e:
            logger.error(f"Universal node feature creation failed: {e}")
            # ❌ REMOVED: Silent fallback - let the error propagate
            raise RuntimeError(f"Universal node feature creation failed: {e}")

    def _create_universal_edge_data(self):
        """Create universal edge data from discovered relations"""

        if not self.relation_to_idx:
            logger.warning("No relations found for edge data creation")
            return

        try:
            # Build edge index and attributes
            edge_list = []
            edge_attrs = []
            edge_types = []

            for relation in self.text_processor.relations:
                source_id = relation.source_entity_id
                target_id = relation.target_entity_id

                # Check if both entities exist in our mapping
                if source_id in self.entity_to_idx and target_id in self.entity_to_idx:
                    source_idx = self.entity_to_idx[source_id]
                    target_idx = self.entity_to_idx[target_id]

                    # Add edge
                    edge_list.append([source_idx, target_idx])

                    # Add edge attributes (confidence, relation type)
                    edge_attrs.append([relation.confidence])

                    # Add edge type
                    relation_type = relation.relation_type
                    if relation_type in self.relation_type_to_idx:
                        edge_types.append(self.relation_type_to_idx[relation_type])
                    else:
                        edge_types.append(0)  # Default type

            if edge_list:
                # Convert to tensors
                self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                self.edge_types = torch.tensor(edge_types, dtype=torch.long)

                logger.info(f"Universal edge data created: {self.edge_index.shape[1]} edges")
            else:
                logger.warning("No valid edges found")
                # Create empty tensors
                self.edge_index = torch.zeros((2, 0), dtype=torch.long)
                self.edge_attr = torch.zeros((0, 1), dtype=torch.float)
                self.edge_types = torch.zeros(0, dtype=torch.long)

        except Exception as e:
            logger.error(f"Universal edge data creation failed: {e}")
            # ❌ REMOVED: Silent fallback - let the error propagate
            raise RuntimeError(f"Universal edge data creation failed: {e}")

    def _create_universal_torch_geometric_data(self) -> Optional[Data]:
        """Create PyTorch Geometric data object from universal features"""

        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("PyTorch Geometric not available")
            return None

        try:
            data = Data(
                x=self.entity_features,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                edge_type=self.edge_types,
                node_type=self.entity_types
            )

            logger.info(f"Universal PyTorch Geometric data created: {data}")
            return data

        except Exception as e:
            logger.error(f"Universal PyTorch Geometric data creation failed: {e}")
            return None

    def _create_universal_data_splits(self, gnn_data) -> Tuple[Any, Any, Any]:
        """Create universal train/validation/test splits"""

        if gnn_data is None:
            return None, None, None

        try:
            num_nodes = gnn_data.num_nodes

            # Simple random split (80/10/10)
            indices = torch.randperm(num_nodes)
            train_size = int(0.8 * num_nodes)
            val_size = int(0.1 * num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True

            # Create data copies with masks
            train_data = gnn_data.clone()
            train_data.train_mask = train_mask

            val_data = gnn_data.clone()
            val_data.val_mask = val_mask

            test_data = gnn_data.clone()
            test_data.test_mask = test_mask

            return train_data, val_data, test_data

        except Exception as e:
            logger.error(f"Universal data split creation failed: {e}")
            return None, None, None

    def _create_universal_task_labels(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create universal labels for GNN tasks"""

        try:
            # Node classification labels: predict entity type
            node_labels = self.entity_types.clone() if self.entity_types is not None else None

            # Edge prediction labels: 1 for existing edges, 0 for non-existing
            edge_labels = torch.ones(self.edge_index.shape[1]) if self.edge_index is not None else None

            return node_labels, edge_labels

        except Exception as e:
            logger.error(f"Universal task label creation failed: {e}")
            return None, None

    def _get_universal_dataset_stats(self) -> Dict[str, Any]:
        """Get universal dataset statistics"""
        return {
            "domain": self.domain,
            "num_entities": len(self.entity_to_idx),
            "num_relations": len(self.relation_to_idx),
            "num_entity_types": len(self.entity_type_to_idx),
            "num_relation_types": len(self.relation_type_to_idx),
            "num_edges": self.edge_index.shape[1] if self.edge_index is not None else 0,
            "feature_dimension": self.entity_features.shape[1] if self.entity_features is not None else 0,
            "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE
        }

    def _save_cached_data(self, dataset: Dict[str, Any], cache_path: Path):
        """Save universal dataset to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            logger.info(f"Universal GNN dataset cached at {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache universal GNN dataset: {e}")

    def _load_cached_data(self, cache_path: Path) -> Dict[str, Any]:
        """Load universal dataset from cache"""
        try:
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f"Universal GNN dataset loaded from cache: {cache_path}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load cached universal GNN dataset: {e}")
            return {"error": f"Cache loading failed: {e}"}


# Legacy compatibility alias
MaintIEGNNDataProcessor = AzureMLGNNProcessor


def create_universal_gnn_processor(text_processor: AzureOpenAITextProcessor,
                                 domain: str = "general") -> AzureMLGNNProcessor:
    """Factory function to create universal GNN data processor"""
    return AzureMLGNNProcessor(text_processor, domain)


if __name__ == "__main__":
    # Example usage
    from ..azure_openai import AzureOpenAITextProcessor

    processor = AzureOpenAITextProcessor("maintenance")
    gnn_processor = AzureMLGNNProcessor(processor, "maintenance")

    # Process GNN data
    dataset = gnn_processor.prepare_universal_gnn_data()
    print(f"Universal GNN dataset: {dataset.get('stats', {})}")