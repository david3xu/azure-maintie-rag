"""
Tri-Modal Search System - Vector + Graph + GNN Unified Search

This module implements the tri-modal unity principle by providing:
- TriModalOrchestrator: Main coordination and result synthesis
- VectorSearchEngine: Semantic similarity search
- GraphSearchEngine: Relational context search
- GNNSearchEngine: Pattern prediction search
- PydanticAI Tools: Enterprise integration for PydanticAI agents

Key competitive advantage: Simultaneous execution of all modalities
without heuristic selection, providing comprehensive unified results.
"""

from .gnn_search import GNNSearchEngine
from .graph_search import GraphSearchEngine

# Import main orchestrator
from .orchestrator import ModalityResult, SearchResult, TriModalOrchestrator

# PydanticAI tools moved to toolsets.py following target architecture
# from .pydantic_tools import (
#     execute_graph_search,
#     execute_tri_modal_search,
#     execute_vector_search,
#     search_with_tri_modal_tool,
# )

# Import individual search engines
from .vector_search import VectorSearchEngine

__all__ = [
    # Main orchestrator
    "TriModalOrchestrator",
    "SearchResult",
    "ModalityResult",
    # Individual search engines
    "VectorSearchEngine",
    "GraphSearchEngine",
    "GNNSearchEngine",
    # PydanticAI Tools moved to toolsets.py following target architecture
    # "execute_tri_modal_search",
    # "execute_vector_search", 
    # "execute_graph_search",
    # "search_with_tri_modal_tool",
]
