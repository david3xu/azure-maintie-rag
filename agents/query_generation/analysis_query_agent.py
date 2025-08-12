"""
Analysis Query Agent - Compatibility Module
===========================================

Backward compatibility wrapper for analysis query generation.
Actual implementation is in agents.shared.query_tools.
"""

from agents.shared.query_tools import generate_analysis_query

# Re-export for backward compatibility
__all__ = ["generate_analysis_query"]
