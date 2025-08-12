"""
Query Generation Module for Universal RAG System
===============================================

This module provides query generation functionality through compatibility aliases
to the actual implementation in agents.shared.query_tools.

For backward compatibility with existing imports.
"""

# Import all query generation functions from the actual implementation
from agents.shared.query_tools import (
    generate_analysis_query,
    generate_gremlin_query,
)

# Re-export for backward compatibility
__all__ = [
    "generate_gremlin_query",
    "generate_analysis_query",
    "generate_query",  # Generic alias
]

# Generic query function alias
generate_query = generate_gremlin_query  # Default to gremlin query generation
