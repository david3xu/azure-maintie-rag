"""Universal GNN data loader for Azure Universal RAG system."""

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from core.models.universal_rag_models import UniversalEntity, UniversalRelation

logger = logging.getLogger(__name__)


def load_graph_data_from_cosmos(domain: str = "general") -> Tuple[List[Data], List[Data]]:
    """
    Load graph data from Azure Cosmos DB Gremlin

    Args:
        domain: Domain to load data for

    Returns:
        Tuple of (train_data, val_data) lists of PyTorch Geometric Data objects
    """
    try:
        # Use existing Gremlin client
        cosmos_client = AzureCosmosGremlinClient()

        # Query entities and relations
        entities = cosmos_client.get_all_entities(domain)
        relations = cosmos_client.get_all_relations(domain)

        logger.info(f"Loaded {len(entities)} entities and {len(relations)} relations from Cosmos DB")

        # Convert to PyTorch Geometric format
        graph_data = convert_to_pytorch_geometric(entities, relations)

        # Split into train/val
        train_data, val_data = split_graph_data(graph_data, train_ratio=0.8)

        return train_data, val_data

    except Exception as e:
        logger.error(f"Failed to load graph data from Cosmos DB: {e}")
        return [], []


def convert_to_pytorch_geometric(entities: List[Dict[str, Any]],
                                relations: List[Dict[str, Any]]) -> List[Data]:
    """
    Convert entities and relations to PyTorch Geometric Data format

    Args:
        entities: List of entity dictionaries
        relations: List of relation dictionaries

    Returns:
        List of PyTorch Geometric Data objects
    """
    try:
        # Create entity mapping
        entity_to_id = {}
        node_features = []
        node_labels = []

        # Process entities
        for i, entity in enumerate(entities):
            entity_to_id[entity["text"]] = i

            # Create node features (simple encoding for now)
            # In practice, you might use embeddings from a language model
            feature_vector = create_node_features(entity)
            node_features.append(feature_vector)

            # Create node labels (entity type classification)
            label = encode_entity_type(entity.get("entity_type", "unknown"))
            node_labels.append(label)

        # Create edge indices and features
        edge_indices = []
        edge_features = []

        for relation in relations:
            source_text = relation.get("source_entity", "")
            target_text = relation.get("target_entity", "")

            if source_text in entity_to_id and target_text in entity_to_id:
                source_id = entity_to_id[source_text]
                target_id = entity_to_id[target_text]

                # Add directed edge
                edge_indices.append([source_id, target_id])

                # Create edge features
                edge_feature = create_edge_features(relation)
                edge_features.append(edge_feature)

        # Convert to tensors
        if node_features:
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.long)
        else:
            # Create dummy data if no entities
            x = torch.zeros((1, 64), dtype=torch.float)
            y = torch.zeros(1, dtype=torch.long)

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Create dummy edge if no relations
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.zeros((1, 32), dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        logger.info(f"Created graph with {len(node_features)} nodes and {len(edge_indices)} edges")
        return [data]

    except Exception as e:
        logger.error(f"Failed to convert to PyTorch Geometric format: {e}")
        return []


def create_node_features(entity: Dict[str, Any]) -> List[float]:
    """
    Create node features from entity data

    Args:
        entity: Entity dictionary

    Returns:
        Feature vector
    """
    # Simple feature engineering - in practice, use embeddings
    features = []

    # Entity type encoding (one-hot like)
    entity_type = entity.get("entity_type", "unknown")
    type_features = encode_entity_type_features(entity_type)
    features.extend(type_features)

    # Text length features
    text = entity.get("text", "")
    features.append(len(text) / 100.0)  # Normalized text length

    # Confidence feature
    confidence = entity.get("confidence", 1.0)
    features.append(confidence)

    # Pad to fixed size (64 features)
    while len(features) < 64:
        features.append(0.0)

    return features[:64]


def create_edge_features(relation: Dict[str, Any]) -> List[float]:
    """
    Create edge features from relation data

    Args:
        relation: Relation dictionary

    Returns:
        Feature vector
    """
    # Simple feature engineering for edges
    features = []

    # Relation type encoding
    relation_type = relation.get("relation_type", "unknown")
    type_features = encode_relation_type_features(relation_type)
    features.extend(type_features)

    # Confidence feature
    confidence = relation.get("confidence", 1.0)
    features.append(confidence)

    # Pad to fixed size (32 features)
    while len(features) < 32:
        features.append(0.0)

    return features[:32]


def encode_entity_type(entity_type: str) -> int:
    """Encode entity type to integer label"""
    # Simple encoding - in practice, use a proper label encoder
    type_mapping = {
        "person": 0,
        "organization": 1,
        "location": 2,
        "concept": 3,
        "unknown": 4
    }
    return type_mapping.get(entity_type.lower(), 4)


def encode_entity_type_features(entity_type: str) -> List[float]:
    """Encode entity type to feature vector"""
    # One-hot like encoding
    types = ["person", "organization", "location", "concept", "unknown"]
    features = [1.0 if entity_type.lower() == t else 0.0 for t in types]
    return features


def encode_relation_type_features(relation_type: str) -> List[float]:
    """Encode relation type to feature vector"""
    # One-hot like encoding for relation types
    types = ["works_for", "located_in", "part_of", "related_to", "unknown"]
    features = [1.0 if relation_type.lower() == t else 0.0 for t in types]
    return features


def split_graph_data(graph_data: List[Data], train_ratio: float = 0.8) -> Tuple[List[Data], List[Data]]:
    """
    Split graph data into train and validation sets

    Args:
        graph_data: List of PyTorch Geometric Data objects
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_data, val_data)
    """
    if not graph_data:
        return [], []

    # For single graph, we can't split easily
    # In practice, you might want to use node-level splitting
    if len(graph_data) == 1:
        return graph_data, []

    # For multiple graphs, split the list
    split_idx = int(len(graph_data) * train_ratio)
    train_data = graph_data[:split_idx]
    val_data = graph_data[split_idx:]

    return train_data, val_data


def create_data_loaders(train_data: List[Data],
                       val_data: List[Data],
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch Geometric DataLoaders

    Args:
        train_data: Training data
        val_data: Validation data
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_graph_data(data_path: Optional[str] = None,
                   domain: str = "general",
                   batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Main function to load graph data

    Args:
        data_path: Path to data (if loading from file)
        domain: Domain to load from Cosmos DB
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if data_path and Path(data_path).exists():
        # Load from file (for future implementation)
        logger.info(f"Loading graph data from file: {data_path}")
        # TODO: Implement file-based loading
        return DataLoader([]), DataLoader([])
    else:
        # Load from Cosmos DB
        logger.info(f"Loading graph data from Cosmos DB for domain: {domain}")
        train_data, val_data = load_graph_data_from_cosmos(domain)

        if not train_data:
            logger.warning("No training data loaded")
            return DataLoader([]), DataLoader([])

        train_loader, val_loader = create_data_loaders(train_data, val_data, batch_size)

        logger.info(f"Created data loaders: train={len(train_data)} graphs, "
                   f"val={len(val_data)} graphs")

        return train_loader, val_loader


def get_graph_statistics(data_loader: DataLoader) -> Dict[str, Any]:
    """
    Get statistics about the graph data

    Args:
        data_loader: Data loader to analyze

    Returns:
        Dictionary with graph statistics
    """
    if not data_loader.dataset:
        return {"error": "No dataset available"}

    total_nodes = 0
    total_edges = 0
    num_graphs = len(data_loader.dataset)

    for data in data_loader.dataset:
        total_nodes += data.x.size(0)
        total_edges += data.edge_index.size(1)

    return {
        "num_graphs": num_graphs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_nodes_per_graph": total_nodes / num_graphs if num_graphs > 0 else 0,
        "avg_edges_per_graph": total_edges / num_graphs if num_graphs > 0 else 0
    }