"""
Semantic Feature Engineering for GNN Training.

This module replaces the simplistic one-hot encoding with semantic embeddings
using Azure OpenAI, addressing the feature engineering problems identified
in the design analysis.

Key Features:
- Semantic embeddings via Azure OpenAI
- Dynamic type encoding based on actual data
- Domain-adaptive feature generation
- Caching for performance optimization
- Fallback to simple features if OpenAI unavailable

Created as part of GNN Training Stage Design Analysis remediation plan.
Location: /docs/workflows/GNN_Training_Stage_Design_Analysis.md
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta

from core.models.gnn_data_models import (
    StandardizedEntity,
    StandardizedRelation, 
    StandardizedGraphData
)
from core.azure_openai.completion_service import AzureOpenAIService

logger = logging.getLogger(__name__)


class SemanticFeatureEngine:
    """
    Generate semantic features using Azure OpenAI embeddings.
    
    This class replaces the hardcoded 64-dimensional features with
    dynamic semantic embeddings, dramatically improving GNN performance.
    
    Features:
    - Azure OpenAI text-embedding-ada-002 integration
    - Intelligent caching for performance
    - Domain-specific feature adaptation
    - Fallback to simple features for offline scenarios
    """
    
    def __init__(
        self, 
        openai_service: Optional[AzureOpenAIService] = None,
        cache_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize semantic feature engine.
        
        Args:
            openai_service: Azure OpenAI service instance
            cache_dir: Directory for caching embeddings
            embedding_model: Azure OpenAI embedding model name
        """
        self.openai_service = openai_service
        self.embedding_model = embedding_model
        self.embedding_dim = 1536  # text-embedding-ada-002 dimension
        
        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("backend/data/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self._load_cache()
        
        # Dynamic type encoders
        self.entity_type_encoder = DynamicTypeEncoder("entity")
        self.relation_type_encoder = DynamicTypeEncoder("relation")
        
        logger.info(f"SemanticFeatureEngine initialized with model: {embedding_model}")
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / "embedding_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                self.cache = {k: np.array(v) for k, v in cache_data.items()}
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            cache_file = self.cache_dir / "embedding_cache.json"
            cache_data = {k: v.tolist() for k, v in self.cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        content = f"{text}||{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def generate_entity_embeddings(
        self, 
        entities: List[StandardizedEntity],
        batch_size: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Generate semantic embeddings for entities.
        
        Args:
            entities: List of entities to process
            batch_size: Batch size for API calls
            
        Returns:
            Dictionary mapping entity_id to embedding vector
            
        Example:
            >>> engine = SemanticFeatureEngine(openai_service)
            >>> embeddings = await engine.generate_entity_embeddings(entities)
            >>> print(f"Generated embeddings for {len(embeddings)} entities")
        """
        embeddings = {}
        
        # Prepare texts for embedding
        entity_texts = []
        entity_ids = []
        cache_hits = 0
        
        for entity in entities:
            # Create rich text representation
            entity_text = self._create_entity_text(entity)
            cache_key = self._get_cache_key(entity_text, self.embedding_model)
            
            # Check cache first
            if cache_key in self.cache:
                embeddings[entity.entity_id] = self.cache[cache_key]
                cache_hits += 1
            else:
                entity_texts.append(entity_text)
                entity_ids.append(entity.entity_id)
        
        logger.info(f"Cache hits: {cache_hits}/{len(entities)}, generating {len(entity_texts)} new embeddings")
        
        # Generate new embeddings in batches
        if entity_texts and self.openai_service:
            try:
                for i in range(0, len(entity_texts), batch_size):
                    batch_texts = entity_texts[i:i+batch_size]
                    batch_ids = entity_ids[i:i+batch_size]
                    
                    # Get embeddings from Azure OpenAI
                    batch_embeddings = await self._get_openai_embeddings(batch_texts)
                    
                    # Store results and cache
                    for entity_id, text, embedding in zip(batch_ids, batch_texts, batch_embeddings):
                        embeddings[entity_id] = embedding
                        cache_key = self._get_cache_key(text, self.embedding_model)
                        self.cache[cache_key] = embedding
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                
                # Save updated cache
                self._save_cache()
                
            except Exception as e:
                logger.error(f"Failed to generate OpenAI embeddings: {e}")
                # Fallback to simple embeddings
                for entity_id in entity_ids:
                    embeddings[entity_id] = self._create_fallback_embedding()
        
        elif entity_texts:
            logger.warning("OpenAI service not available, using fallback embeddings")
            # Generate fallback embeddings
            for entity_id in entity_ids:
                embeddings[entity_id] = self._create_fallback_embedding()
        
        logger.info(f"Generated embeddings for {len(embeddings)} entities")
        return embeddings
    
    async def generate_relation_embeddings(
        self, 
        relations: List[StandardizedRelation],
        entity_embeddings: Dict[str, np.ndarray],
        batch_size: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Generate semantic embeddings for relations.
        
        Args:
            relations: List of relations to process
            entity_embeddings: Pre-computed entity embeddings
            batch_size: Batch size for API calls
            
        Returns:
            Dictionary mapping relation_id to embedding vector
        """
        embeddings = {}
        
        # Prepare texts for embedding
        relation_texts = []
        relation_ids = []
        cache_hits = 0
        
        for relation in relations:
            # Create rich text representation
            relation_text = self._create_relation_text(relation, entity_embeddings)
            cache_key = self._get_cache_key(relation_text, self.embedding_model)
            
            # Check cache first
            if cache_key in self.cache:
                embeddings[relation.relation_id] = self.cache[cache_key]
                cache_hits += 1
            else:
                relation_texts.append(relation_text)
                relation_ids.append(relation.relation_id)
        
        logger.info(f"Relation cache hits: {cache_hits}/{len(relations)}, generating {len(relation_texts)} new embeddings")
        
        # Generate new embeddings
        if relation_texts and self.openai_service:
            try:
                for i in range(0, len(relation_texts), batch_size):
                    batch_texts = relation_texts[i:i+batch_size]
                    batch_ids = relation_ids[i:i+batch_size]
                    
                    batch_embeddings = await self._get_openai_embeddings(batch_texts)
                    
                    for relation_id, text, embedding in zip(batch_ids, batch_texts, batch_embeddings):
                        embeddings[relation_id] = embedding
                        cache_key = self._get_cache_key(text, self.embedding_model)
                        self.cache[cache_key] = embedding
                    
                    await asyncio.sleep(0.1)
                
                self._save_cache()
                
            except Exception as e:
                logger.error(f"Failed to generate relation embeddings: {e}")
                for relation_id in relation_ids:
                    embeddings[relation_id] = self._create_fallback_embedding()
        
        elif relation_texts:
            logger.warning("Using fallback relation embeddings")
            for relation_id in relation_ids:
                embeddings[relation_id] = self._create_fallback_embedding()
        
        return embeddings
    
    def _create_entity_text(self, entity: StandardizedEntity) -> str:
        """
        Create rich text representation for entity embedding.
        
        Args:
            entity: Entity to process
            
        Returns:
            Rich text representation
        """
        # Combine entity information for better embeddings
        text_parts = [
            f"Entity: {entity.text}",
            f"Type: {entity.entity_type}",
        ]
        
        if entity.context:
            text_parts.append(f"Context: {entity.context}")
        
        return " | ".join(text_parts)
    
    def _create_relation_text(
        self, 
        relation: StandardizedRelation,
        entity_embeddings: Dict[str, np.ndarray]
    ) -> str:
        """
        Create rich text representation for relation embedding.
        
        Args:
            relation: Relation to process
            entity_embeddings: Entity embeddings for context
            
        Returns:
            Rich text representation
        """
        text_parts = [
            f"Relation: {relation.relation_type}",
            f"From: {relation.source_entity}",
            f"To: {relation.target_entity}"
        ]
        
        if relation.context:
            text_parts.append(f"Context: {relation.context}")
        
        return " | ".join(text_parts)
    
    async def _get_openai_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings from Azure OpenAI service.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Call Azure OpenAI embedding API
            response = await self.openai_service.get_embeddings(
                texts=texts,
                model=self.embedding_model
            )
            
            # Convert to numpy arrays
            embeddings = [np.array(emb) for emb in response.embeddings]
            return embeddings
            
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            # Return fallback embeddings
            return [self._create_fallback_embedding() for _ in texts]
    
    def _create_fallback_embedding(self) -> np.ndarray:
        """
        Create fallback embedding when OpenAI is unavailable.
        
        Returns:
            Random normalized embedding vector
        """
        # Create random but normalized embedding
        embedding = np.random.normal(0, 1, self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def create_dynamic_node_features(
        self, 
        entity: StandardizedEntity,
        embedding: np.ndarray,
        domain: str
    ) -> np.ndarray:
        """
        Create dynamic node features combining embeddings with metadata.
        
        Args:
            entity: Entity to process
            embedding: Semantic embedding
            domain: Domain context
            
        Returns:
            Feature vector for GNN
        """
        features = []
        
        # Add semantic embedding (primary features)
        features.extend(embedding.tolist())
        
        # Add confidence feature
        features.append(entity.confidence)
        
        # Add domain-specific features
        domain_features = self._get_domain_features(entity, domain)
        features.extend(domain_features)
        
        # Add dynamic type encoding
        type_encoding = self.entity_type_encoder.encode(entity.entity_type)
        features.extend(type_encoding)
        
        return np.array(features, dtype=np.float32)
    
    def create_dynamic_edge_features(
        self, 
        relation: StandardizedRelation,
        embedding: np.ndarray,
        domain: str
    ) -> np.ndarray:
        """
        Create dynamic edge features combining embeddings with metadata.
        
        Args:
            relation: Relation to process
            embedding: Semantic embedding
            domain: Domain context
            
        Returns:
            Feature vector for GNN edges
        """
        features = []
        
        # Add semantic embedding (reduced dimension for edges)
        # Use first 256 dimensions to keep edge features manageable
        features.extend(embedding[:256].tolist())
        
        # Add confidence feature
        features.append(relation.confidence)
        
        # Add dynamic type encoding
        type_encoding = self.relation_type_encoder.encode(relation.relation_type)
        features.extend(type_encoding)
        
        return np.array(features, dtype=np.float32)
    
    def _get_domain_features(self, entity: StandardizedEntity, domain: str) -> List[float]:
        """
        Get domain-specific features for entity.
        
        Args:
            entity: Entity to process
            domain: Domain context
            
        Returns:
            List of domain-specific features
        """
        features = []
        
        # Text length features (normalized)
        text_length = len(entity.text)
        features.append(min(text_length / 100.0, 1.0))  # Capped at 1.0
        
        # Domain-specific patterns
        if domain == "maintenance":
            # Maintenance-specific features
            maintenance_keywords = ["repair", "fix", "maintenance", "service", "replace"]
            keyword_match = sum(1 for keyword in maintenance_keywords if keyword in entity.text.lower())
            features.append(keyword_match / len(maintenance_keywords))
        
        elif domain == "medical":
            # Medical-specific features  
            medical_keywords = ["treatment", "diagnosis", "patient", "symptom", "disease"]
            keyword_match = sum(1 for keyword in medical_keywords if keyword in entity.text.lower())
            features.append(keyword_match / len(medical_keywords))
        
        else:
            # General domain
            features.append(0.0)
        
        # Ensure consistent feature count
        while len(features) < 3:
            features.append(0.0)
        
        return features[:3]
    
    def fit_type_encoders(self, graph_data: StandardizedGraphData):
        """
        Fit dynamic type encoders on graph data.
        
        Args:
            graph_data: Graph data to learn types from
        """
        # Collect all entity and relation types
        entity_types = [entity.entity_type for entity in graph_data.entities]
        relation_types = [relation.relation_type for relation in graph_data.relations]
        
        # Fit encoders
        self.entity_type_encoder.fit(entity_types)
        self.relation_type_encoder.fit(relation_types)
        
        logger.info(f"Fitted type encoders: {len(self.entity_type_encoder.type_to_id)} entity types, "
                   f"{len(self.relation_type_encoder.type_to_id)} relation types")


class DynamicTypeEncoder:
    """
    Dynamic type encoder that learns from data instead of using hardcoded mappings.
    
    This replaces the fixed type mappings with adaptive encoding based on
    actual types found in the extraction data.
    """
    
    def __init__(self, encoder_type: str = "entity"):
        """
        Initialize dynamic type encoder.
        
        Args:
            encoder_type: Type of encoder ("entity" or "relation")
        """
        self.encoder_type = encoder_type
        self.type_to_id = {}
        self.id_to_type = {}
        self.unknown_id = None
        
    def fit(self, types: List[str]):
        """
        Learn type mappings from data.
        
        Args:
            types: List of types to learn from
        """
        # Get unique types
        unique_types = list(set(types))
        unique_types.sort()  # For reproducible ordering
        
        # Create mappings
        self.type_to_id = {type_name: i for i, type_name in enumerate(unique_types)}
        self.id_to_type = {i: type_name for type_name, i in self.type_to_id.items()}
        
        # Add unknown type
        unknown_type = "unknown"
        if unknown_type not in self.type_to_id:
            self.unknown_id = len(self.type_to_id)
            self.type_to_id[unknown_type] = self.unknown_id
            self.id_to_type[self.unknown_id] = unknown_type
        else:
            self.unknown_id = self.type_to_id[unknown_type]
        
        logger.debug(f"Fitted {self.encoder_type} encoder with {len(self.type_to_id)} types")
    
    def encode(self, type_name: str) -> List[float]:
        """
        Encode type as one-hot vector.
        
        Args:
            type_name: Type name to encode
            
        Returns:
            One-hot encoded vector
        """
        if not self.type_to_id:
            # Not fitted yet, return empty encoding
            return [0.0]
        
        type_id = self.type_to_id.get(type_name, self.unknown_id)
        
        # Create one-hot vector
        encoding = [0.0] * len(self.type_to_id)
        encoding[type_id] = 1.0
        
        return encoding
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension for this encoder."""
        return len(self.type_to_id) if self.type_to_id else 1


class FeaturePipeline:
    """
    Complete feature processing pipeline for GNN training.
    
    Orchestrates the entire feature engineering process from raw graph data
    to GNN-ready tensors.
    """
    
    def __init__(
        self, 
        semantic_engine: SemanticFeatureEngine,
        normalize_features: bool = True
    ):
        """
        Initialize feature pipeline.
        
        Args:
            semantic_engine: Semantic feature engine
            normalize_features: Whether to normalize features
        """
        self.semantic_engine = semantic_engine
        self.normalize_features = normalize_features
        
    async def process_graph_data(
        self, 
        graph_data: StandardizedGraphData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process complete graph data into GNN features.
        
        Args:
            graph_data: Standardized graph data
            
        Returns:
            Tuple of (node_features, edge_features, edge_indices)
        """
        logger.info(f"Processing graph data: {len(graph_data.entities)} entities, {len(graph_data.relations)} relations")
        
        # Fit type encoders
        self.semantic_engine.fit_type_encoders(graph_data)
        
        # Generate embeddings
        entity_embeddings = await self.semantic_engine.generate_entity_embeddings(graph_data.entities)
        relation_embeddings = await self.semantic_engine.generate_relation_embeddings(
            graph_data.relations, entity_embeddings
        )
        
        # Create node features
        node_features = []
        entity_id_to_index = {}
        
        for i, entity in enumerate(graph_data.entities):
            entity_id_to_index[entity.entity_id] = i
            
            embedding = entity_embeddings.get(entity.entity_id, self.semantic_engine._create_fallback_embedding())
            features = self.semantic_engine.create_dynamic_node_features(
                entity, embedding, graph_data.domain
            )
            node_features.append(features)
        
        # Create edge features and indices
        edge_features = []
        edge_indices = []
        
        for relation in graph_data.relations:
            # Check if both entities exist
            if (relation.source_entity in entity_id_to_index and 
                relation.target_entity in entity_id_to_index):
                
                source_idx = entity_id_to_index[relation.source_entity]
                target_idx = entity_id_to_index[relation.target_entity]
                
                edge_indices.append([source_idx, target_idx])
                
                embedding = relation_embeddings.get(relation.relation_id, self.semantic_engine._create_fallback_embedding())
                features = self.semantic_engine.create_dynamic_edge_features(
                    relation, embedding, graph_data.domain
                )
                edge_features.append(features)
        
        # Convert to numpy arrays
        node_features = np.array(node_features, dtype=np.float32)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, 256 + 1 + 5), dtype=np.float32)
        edge_indices = np.array(edge_indices, dtype=np.int64).T if edge_indices else np.empty((2, 0), dtype=np.int64)
        
        # Normalize features if requested
        if self.normalize_features:
            node_features = self._normalize_features(node_features)
            if edge_features.size > 0:
                edge_features = self._normalize_features(edge_features)
        
        logger.info(f"Generated features: nodes {node_features.shape}, edges {edge_features.shape}, indices {edge_indices.shape}")
        
        return node_features, edge_features, edge_indices
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to unit scale.
        
        Args:
            features: Features to normalize
            
        Returns:
            Normalized features
        """
        if features.size == 0:
            return features
            
        # L2 normalization per sample
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms