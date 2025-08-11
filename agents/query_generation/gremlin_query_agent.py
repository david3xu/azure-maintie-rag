"""
Gremlin Query Agent - Compatibility Module
==========================================

Backward compatibility wrapper for Gremlin query generation.
Actual implementation is in agents.shared.query_tools.
"""

from agents.shared.query_tools import generate_gremlin_query

# Re-export for backward compatibility
__all__ = ["generate_gremlin_query"]