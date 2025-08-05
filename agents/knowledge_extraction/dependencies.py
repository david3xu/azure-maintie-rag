"""
Knowledge Extraction Agent Dependencies

Agent-specific dependencies following target architecture.
Provides dependency injection pattern for Knowledge Extraction Agent.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field

from .azure_service_container import ConsolidatedAzureServices
from ..core.cache_manager import UnifiedCacheManager
from ..core.memory_manager import UnifiedMemoryManager


class KnowledgeExtractionDeps(BaseModel):
    """
    Knowledge Extraction Agent dependencies following target architecture.
    
    Provides all necessary dependencies for knowledge extraction operations
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
    
    # Extraction-specific components (lazy initialized)
    entity_processor: Optional[Any] = Field(
        default=None, 
        description="Multi-strategy entity processor"
    )
    relationship_processor: Optional[Any] = Field(
        default=None, 
        description="Relationship extraction processor"
    )
    validation_processor: Optional[Any] = Field(
        default=None, 
        description="Simple validation processor (replaced over-engineered system)"
    )
    
    # Extraction configuration
    extraction_timeout_seconds: int = Field(
        default=300, 
        description="Timeout for extraction operations"
    )
    max_concurrent_extractions: int = Field(
        default=5, 
        description="Maximum concurrent extraction operations"
    )
    enable_validation: bool = Field(
        default=True, 
        description="Enable extraction quality validation"
    )
    cache_extractions: bool = Field(
        default=True, 
        description="Enable caching of extraction results"
    )
    
    class Config:
        arbitrary_types_allowed = True


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