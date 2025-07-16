"""Graph store integration for Universal RAG system."""

from typing import Dict, List, Any, Optional
import networkx as nx


class GraphStoreClient:
    """Universal graph store client that works with any domain."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize graph store client."""
        self.config = config or {}
        self.store_type = self.config.get('type', 'networkx')
        self.graph = nx.MultiDiGraph()  # Simple NetworkX store for now

    def add_node(self, id: str, properties: Dict[str, Any] = None) -> None:
        """Add node to graph."""
        self.graph.add_node(id, **(properties or {}))

    def add_edge(self, source: str, target: str, relation_type: str, properties: Dict[str, Any] = None) -> None:
        """Add edge to graph."""
        self.graph.add_edge(source, target, relation_type=relation_type, **(properties or {}))

    def get_node(self, id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        if id in self.graph:
            return dict(self.graph.nodes[id])
        return None

    def get_neighbors(self, id: str) -> List[str]:
        """Get neighbors of a node."""
        return list(self.graph.neighbors(id))

    def find_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, node_ids: List[str]) -> nx.Graph:
        """Get subgraph containing specified nodes."""
        return self.graph.subgraph(node_ids)