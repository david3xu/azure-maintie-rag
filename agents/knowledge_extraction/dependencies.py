"""
Knowledge Extraction Agent Dependencies

Agent-specific dependencies following target architecture.
Provides dependency injection pattern for Knowledge Extraction Agent.
"""

# Import dependencies from centralized data models
from agents.core.data_models import KnowledgeExtractionDeps


# Factory function for creating dependencies
def create_knowledge_extraction_deps() -> KnowledgeExtractionDeps:
    """
    Create Knowledge Extraction Agent dependencies with lazy initialization.

    Returns:
        KnowledgeExtractionDeps: Configured dependencies instance
    """
    return KnowledgeExtractionDeps()


# Export main components
__all__ = [
    "KnowledgeExtractionDeps",
    "create_knowledge_extraction_deps",
]