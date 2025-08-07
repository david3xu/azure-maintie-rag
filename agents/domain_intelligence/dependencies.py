"""
Domain Intelligence Agent Dependencies

Agent-specific dependencies following target architecture.
Provides dependency injection pattern for Domain Intelligence Agent.
"""

# Import dependencies from centralized data models
from agents.core.data_models import DomainIntelligenceDeps


# Factory function for creating dependencies
def create_domain_intelligence_deps() -> DomainIntelligenceDeps:
    """
    Create Domain Intelligence Agent dependencies with lazy initialization.

    Returns:
        DomainIntelligenceDeps: Configured dependencies instance
    """
    return DomainIntelligenceDeps()


# Export main components
__all__ = [
    "DomainIntelligenceDeps",
    "create_domain_intelligence_deps",
]
