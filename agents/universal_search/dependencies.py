"""
Universal Search Agent Dependencies

Agent-specific dependencies following target architecture.
Provides dependency injection pattern for Universal Search Agent.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field

from .azure_service_container import ConsolidatedAzureServices
from ..core.cache_manager import UnifiedCacheManager
from ..core.memory_manager import UnifiedMemoryManager
from .vector_search import VectorSearchEngine
from .graph_search import GraphSearchEngine
from .gnn_search import GNNSearchEngine


class UniversalSearchDeps(BaseModel):
    """
    Universal Search Agent dependencies following target architecture.
    
    Provides all necessary dependencies for tri-modal search operations
    with proper type validation and lazy initialization support.
    """
    
    # Core services
    azure_services: Optional[ConsolidatedAzureServices] = Field(
        default=None, 
        description="Azure services container"
    )
    cache_manager: Optional[UnifiedCacheManager] = Field(
        default=None, 
        description="Cache manager instance"
    )
    memory_manager: Optional[UnifiedMemoryManager] = Field(
        default=None, 
        description="Memory manager instance"
    )
    
    # Search engine components
    vector_search_engine: Optional[VectorSearchEngine] = Field(
        default=None, 
        description="Semantic vector search engine"
    )
    graph_search_engine: Optional[GraphSearchEngine] = Field(
        default=None, 
        description="Graph relationship search engine"
    )
    gnn_search_engine: Optional[GNNSearchEngine] = Field(
        default=None, 
        description="GNN pattern prediction search engine"
    )
    
    # Search configuration
    max_results_per_modality: int = Field(
        default=10, 
        description="Maximum results per search modality"
    )
    search_timeout_seconds: int = Field(
        default=30, 
        description="Timeout for search operations"
    )
    enable_parallel_search: bool = Field(
        default=True, 
        description="Enable parallel execution of search modalities"
    )
    result_synthesis_enabled: bool = Field(
        default=True, 
        description="Enable intelligent result synthesis"
    )
    
    # Performance settings
    cache_search_results: bool = Field(
        default=True, 
        description="Enable caching of search results"
    )
    performance_monitoring: bool = Field(
        default=True, 
        description="Enable performance monitoring"
    )
    
    class Config:
        arbitrary_types_allowed = True


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