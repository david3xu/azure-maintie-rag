"""
Schema processing utilities for MaintIE scheme.json
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SchemeNode:
    """Represents a node in the scheme hierarchy"""
    name: str
    fullname: str
    children: List['SchemeNode']
    metadata: Dict[str, Any]
    path: List[int]

class SchemeProcessor:
    """Process MaintIE scheme.json with hierarchy support"""

    def __init__(self, scheme_path: Path):
        self.scheme_path = scheme_path
        self.entity_hierarchy = {}
        self.relation_hierarchy = {}
        self.all_types = {"entity": set(), "relation": set()}

    def load_scheme(self) -> Dict[str, Any]:
        """Load and process complete scheme hierarchy"""
        if not self.scheme_path.exists():
            logger.warning(f"Scheme file not found: {self.scheme_path}")
            return {"entity": [], "relation": []}

        try:
            with open(self.scheme_path, 'r') as f:
                scheme = json.load(f)

            # Process hierarchies
            self.entity_hierarchy = self._build_hierarchy(scheme.get("entity", []))
            self.relation_hierarchy = self._build_hierarchy(scheme.get("relation", []))

            # Build flat type sets for quick lookup
            self._build_type_sets()

            logger.info(f"Loaded scheme: {len(self.all_types['entity'])} entity types, "
                       f"{len(self.all_types['relation'])} relation types")

            return scheme
        except Exception as e:
            logger.error(f"Error loading scheme: {e}")
            return {"entity": [], "relation": []}

    def _build_hierarchy(self, items: List[Dict]) -> Dict[str, SchemeNode]:
        """Build hierarchy tree from scheme items"""
        hierarchy = {}

        def collect_nodes(items_list, parent_path=""):
            for item in items_list:
                node = self._create_scheme_node(item)
                hierarchy[node.fullname] = node
                # Recursively collect children
                collect_nodes(item.get("children", []), node.fullname)

        collect_nodes(items)
        return hierarchy

    def _create_scheme_node(self, item: Dict) -> SchemeNode:
        """Create scheme node with children"""
        children = []
        for child_item in item.get("children", []):
            children.append(self._create_scheme_node(child_item))

        return SchemeNode(
            name=item.get("name", ""),
            fullname=item.get("fullname", ""),
            children=children,
            metadata={
                "color": item.get("color", ""),
                "active": item.get("active", True),
                "description": item.get("description", ""),
                "example_terms": item.get("example_terms", []),
                "id": item.get("id", ""),
                "path": item.get("path", [])
            },
            path=item.get("path", [])
        )

    def _build_type_sets(self):
        """Build flat sets of all available types"""
        def collect_types(hierarchy, category):
            for fullname, node in hierarchy.items():
                self.all_types[category].add(fullname)
                self._collect_children_types(node, category)

        collect_types(self.entity_hierarchy, "entity")
        collect_types(self.relation_hierarchy, "relation")

    def _collect_children_types(self, node: SchemeNode, category: str):
        """Recursively collect all child type names"""
        for child in node.children:
            self.all_types[category].add(child.fullname)
            self._collect_children_types(child, category)

    def get_all_types(self, category: str) -> Set[str]:
        """Get all available types for category"""
        return self.all_types.get(category, set())

    def find_node(self, fullname: str, category: str) -> SchemeNode:
        """Find node by fullname"""
        hierarchy = self.entity_hierarchy if category == "entity" else self.relation_hierarchy
        return hierarchy.get(fullname)