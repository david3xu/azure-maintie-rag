"""
Supporting Infrastructure - Migration Notice

This module is being migrated to agents/shared/ for better organization.
Graph communication models are now available from agents.shared.graph_communication.
"""

# Import from new location for backward compatibility
from agents.shared.graph_communication import MessageType
# GraphMessage, GraphStatus deleted in Phase 1

__all__ = [
    "MessageType",
    # GraphMessage, GraphStatus deleted in Phase 1
]