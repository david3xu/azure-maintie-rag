"""
Domain Intelligence Agent Dependencies

Agent-specific dependencies following target architecture.
Provides dependency injection pattern for Domain Intelligence Agent.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field

from ..core.azure_services import ConsolidatedAzureServices
from ..core.cache_manager import UnifiedCacheManager
from ..core.memory_manager import UnifiedMemoryManager
from .analyzers.unified_content_analyzer import UnifiedContentAnalyzer
from .pattern_engine import PatternEngine
from .config_generator import ConfigGenerator


class DomainIntelligenceDeps(BaseModel):
    """
    Domain Intelligence Agent dependencies following target architecture.
    
    Provides all necessary dependencies for domain intelligence operations
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
    
    # Domain-specific components
    domain_analyzer: Optional[UnifiedContentAnalyzer] = Field(
        default=None, 
        description="Unified content analysis component"
    )
    pattern_engine: Optional[PatternEngine] = Field(
        default=None, 
        description="Pattern extraction engine"
    )
    config_generator: Optional[ConfigGenerator] = Field(
        default=None, 
        description="Configuration generator"
    )
    
    # Performance optimization
    background_processing_enabled: bool = Field(
        default=True, 
        description="Enable background processing for optimization"
    )
    cache_enabled: bool = Field(
        default=True, 
        description="Enable caching for domain detection"
    )
    
    class Config:
        arbitrary_types_allowed = True


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