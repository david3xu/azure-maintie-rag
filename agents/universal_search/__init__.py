"""Universal Search Agent"""

# Import universal models from core
from agents.core.universal_models import SearchConfiguration, SearchResult

from .agent import (
    MultiModalSearchResult,
    create_universal_search_agent,
    run_universal_search,
    universal_search_agent,
)

__all__ = [
    "universal_search_agent",
    "create_universal_search_agent",
    "run_universal_search",
    "MultiModalSearchResult",
    "SearchResult",
    "SearchConfiguration",
]
