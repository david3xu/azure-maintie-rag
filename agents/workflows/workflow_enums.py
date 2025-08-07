"""
Workflow Enums

Shared enum definitions for workflow system to avoid circular imports.
"""

from enum import Enum


class WorkflowState(Enum):
    """Workflow execution states"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class NodeState(Enum):
    """Individual node execution states"""

    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


__all__ = [
    "WorkflowState",
    "NodeState",
]
