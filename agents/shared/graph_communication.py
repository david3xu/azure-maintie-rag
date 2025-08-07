"""
Bidirectional Communication Hub Between Dual Graphs

This module provides access to centralized graph communication models.
All models are now centralized in agents.core.data_models to maintain the 
zero-hardcoded-values philosophy and single source of truth.
"""

# Import models from centralized data models
from agents.core.data_models import (
    MessageType,
)

# Export for backward compatibility
__all__ = [
    "MessageType",
]