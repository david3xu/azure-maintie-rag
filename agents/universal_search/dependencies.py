"""
Universal Search Agent Dependencies

Agent-specific dependencies following target architecture.
Provides dependency injection pattern for Universal Search Agent.
"""

# Import dependencies from centralized data models
from agents.core.data_models import UniversalSearchDeps

# Factory function for creating dependencies
def create_universal_search_deps() -> UniversalSearchDeps:
    """
    Create Universal Search Agent dependencies with lazy initialization.
    
    Returns:
        UniversalSearchDeps: Configured dependencies instance
    """
    return UniversalSearchDeps()


# Export main components
__all__ = [
    "UniversalSearchDeps",
    "create_universal_search_deps",
]