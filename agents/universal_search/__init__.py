"""Universal Search Agent"""

from .agent import (
    universal_search_agent,
    create_universal_search_agent,
    run_universal_search,
    MultiModalSearchResult,
)

# Import universal models from core
from agents.core.universal_models import SearchResult, SearchConfiguration

__all__ = [
    "universal_search_agent",
    "create_universal_search_agent",
    "run_universal_search",
    "MultiModalSearchResult",
    "SearchResult",
    "SearchConfiguration",
]
