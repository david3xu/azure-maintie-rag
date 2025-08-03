"""
Infrastructure Search Layer

This module provides the single source of truth for search orchestration
in the Infrastructure layer, implementing proper layer separation.
"""

from .tri_modal_orchestrator import SearchExecutionResult, TriModalSearchOrchestrator

__all__ = ["TriModalSearchOrchestrator", "SearchExecutionResult"]
