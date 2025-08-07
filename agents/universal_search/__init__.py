"""
Universal Search Agent - Consolidated Tri-Modal Search System

This module implements the consolidated tri-modal search system providing:
- ConsolidatedSearchOrchestrator: Unified search coordination and result synthesis
- Universal Search Agent: Simplified agent creation with consolidated orchestrator
- Tri-modal search capabilities: Vector + Graph + GNN in single orchestrator
- Centralized configuration management

Key advantage: Eliminates redundancy while preserving all tri-modal capabilities
through consolidated processing and centralized configuration.
"""

# Import consolidated components
from .orchestrators.consolidated_search_orchestrator import (
    ConsolidatedSearchOrchestrator,
    TriModalSearchResult,
    SearchResult,
    ModalityResult,
)

# Import consolidated agent
from .agent import (
    get_universal_search_agent,
    universal_search_agent,
    execute_universal_search,
    QueryRequest,
    SearchResponse,
)

# Import search workflow orchestrator (single source of truth)
try:
    from ..workflows.search_workflow_graph import SearchWorkflow

    WORKFLOW_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    WORKFLOW_ORCHESTRATOR_AVAILABLE = False
    SearchWorkflow = None

__all__ = [
    # Consolidated orchestrator
    "ConsolidatedSearchOrchestrator",
    "TriModalSearchResult",
    "SearchResult",
    "ModalityResult",
    # Consolidated agent
    "get_universal_search_agent",
    "universal_search_agent",
    "execute_universal_search",
    "QueryRequest",
    "SearchResponse",
    # Workflow orchestrator (if available)
    "SearchWorkflow",
]
