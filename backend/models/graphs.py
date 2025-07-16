"""Universal graph models for any domain."""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import networkx as nx
from .entities import Entity
from .relations import Relation


@dataclass
class GraphNode:
    """Universal graph node that wraps entities."""

    entity: Entity
    neighbors: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'entity': self.entity.to_dict(),
            'neighbors': list(self.neighbors),
            'metadata': self.metadata
        }


@dataclass
class GraphEdge:
    """Universal graph edge that wraps relations."""

    relation: Relation
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            'relation': self.relation.to_dict(),
            'weight': self.weight,
            'metadata': self.metadata
        }


class KnowledgeGraph:
    """Universal knowledge graph for any domain.

    Domain-agnostic knowledge graph that can represent any domain's
    entities and relationships through configuration.
    """

    def __init__(self, domain: str = "universal"):
        """Initialize empty knowledge graph."""
        self.domain = domain
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.metadata: Dict[str, Any] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add entity to the graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, entity=entity)

    def add_relation(self, relation: Relation) -> None:
        """Add relation to the graph."""
        self.relations[relation.id] = relation

        # Ensure both entities exist
        if relation.source_entity_id not in self.entities:
            raise ValueError(f"Source entity {relation.source_entity_id} not found")
        if relation.target_entity_id not in self.entities:
            raise ValueError(f"Target entity {relation.target_entity_id} not found")

        # Add edge
        self.graph.add_edge(
            relation.source_entity_id,
            relation.target_entity_id,
            key=relation.id,
            relation=relation,
            weight=relation.strength
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self.relations.get(relation_id)

    def get_neighbors(self, entity_id: str, direction: str = "both") -> List[Entity]:
        """Get neighboring entities."""
        if entity_id not in self.graph:
            return []

        neighbor_ids = set()

        if direction in ["both", "outgoing"]:
            neighbor_ids.update(self.graph.successors(entity_id))

        if direction in ["both", "incoming"]:
            neighbor_ids.update(self.graph.predecessors(entity_id))

        return [self.entities[nid] for nid in neighbor_ids if nid in self.entities]

    def get_path(self, source_id: str, target_id: str, max_length: int = 5) -> List[str]:
        """Find shortest path between entities."""
        try:
            # Convert to undirected for path finding
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)

            if len(path) <= max_length + 1:  # +1 because path includes both endpoints
                return path
            return []
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def search_entities(self, query: str, limit: int = 10) -> List[Tuple[Entity, float]]:
        """Search entities by query with relevance scores."""
        results = []

        for entity in self.entities.values():
            score = entity.matches(query)
            if score > 0:
                results.append((entity, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_subgraph(self, entity_ids: List[str], depth: int = 1) -> 'KnowledgeGraph':
        """Extract subgraph around specified entities."""
        subgraph = KnowledgeGraph(domain=f"{self.domain}_subgraph")

        # Collect all nodes within depth
        nodes_to_include = set(entity_ids)
        current_nodes = set(entity_ids)

        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                if node in self.graph:
                    next_nodes.update(self.graph.neighbors(node))
            current_nodes = next_nodes - nodes_to_include
            nodes_to_include.update(current_nodes)

        # Add entities
        for node_id in nodes_to_include:
            if node_id in self.entities:
                subgraph.add_entity(self.entities[node_id])

        # Add relations
        for relation in self.relations.values():
            if (relation.source_entity_id in nodes_to_include and
                relation.target_entity_id in nodes_to_include):
                subgraph.add_relation(relation)

        return subgraph

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'domain': self.domain,
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected()),
            'metadata': self.metadata
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            'domain': self.domain,
            'entities': {eid: entity.to_dict() for eid, entity in self.entities.items()},
            'relations': {rid: relation.to_dict() for rid, relation in self.relations.items()},
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }