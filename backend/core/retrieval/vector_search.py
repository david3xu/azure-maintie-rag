"""
Vector Search - Universal Import
Imports universal components only
"""

from core.retrieval.universal_vector_search import (
    UniversalVectorSearch,
    create_universal_vector_search
)

# Export only universal interface
__all__ = [
    'UniversalVectorSearch',
    'create_universal_vector_search'
]