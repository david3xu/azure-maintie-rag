"""
Core Services - Unified foundational services for agent architecture

This module provides consolidated core services following clean architecture principles:
- ConsolidatedAzureServices: Unified Azure integration
- UnifiedCacheManager: High-performance caching with pattern indexing
- UnifiedMemoryManager: Standardized memory management
- UnifiedErrorHandler: Centralized error handling
- PydanticAI Provider: Enterprise PydanticAI integration with managed identity

All services follow dependency injection patterns and layer boundary rules.
"""

from .azure_services import ConsolidatedAzureServices, create_azure_service_container
from .cache_manager import UnifiedCacheManager, get_cache_manager, cached_operation
from .memory_manager import UnifiedMemoryManager, get_memory_manager
from .error_handler import UnifiedErrorHandler
from .pydantic_ai_provider import (
    create_azure_pydantic_provider,
    create_azure_pydantic_provider_async,
    create_pydantic_agent,
    create_pydantic_agent_async,
    test_azure_provider_connection,
    test_azure_provider_connection_async
)

# Backward compatibility aliases
AzureServiceContainer = ConsolidatedAzureServices
DomainCache = UnifiedCacheManager

__all__ = [
    "ConsolidatedAzureServices",
    "create_azure_service_container",
    "UnifiedCacheManager",
    "get_cache_manager",
    "cached_operation",
    "UnifiedMemoryManager",
    "get_memory_manager",
    "UnifiedErrorHandler",
    # PydanticAI Integration
    "create_azure_pydantic_provider",
    "create_azure_pydantic_provider_async",
    "create_pydantic_agent",
    "create_pydantic_agent_async",
    "test_azure_provider_connection",
    "test_azure_provider_connection_async",
    # Backward compatibility
    "AzureServiceContainer",
    "DomainCache"
]