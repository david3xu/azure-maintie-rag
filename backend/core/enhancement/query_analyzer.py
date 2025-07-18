"""
Query Analyzer - Universal Import
Imports universal components only
"""

from core.enhancement.universal_query_analyzer import (
    UniversalQueryAnalyzer,
    create_universal_analyzer
)

# Export only universal interface
__all__ = [
    'UniversalQueryAnalyzer',
    'create_universal_analyzer'
]