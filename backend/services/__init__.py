"""
Consolidated Business Services Layer
High-level business logic services that orchestrate system resources and coordinate agents

This layer has been consolidated from 11 services to 6 clean services through
strategic merging of complementary functionality while maintaining backward compatibility.

Consolidation Results:
- workflow_service.py (combines legacy workflow tracking + modern orchestration)
- query_service.py (combines enhanced query processing + request orchestration)  
- cache_service.py (combines simple caching + multi-level orchestration)
- agent_service.py (combines PydanticAI service + agent coordination)
- infrastructure_service.py (existing service - no consolidation needed)
- ml_service.py (existing service - no consolidation needed)

Architecture:
- Consolidated services: Unified functionality with backward compatibility
- Enhanced capabilities: Modern orchestration patterns integrated with legacy services
- Clean boundaries: Proper separation between services and agents layers
- Performance optimized: Reduced complexity while maintaining full functionality
"""

# Consolidated services (merged from pairs)
from .workflow_service import ConsolidatedWorkflowService, WorkflowService  # Legacy alias
from .query_service import ConsolidatedQueryService, EnhancedQueryService, RequestOrchestrator  # Legacy aliases
from .cache_service import ConsolidatedCacheService, SimpleCacheService, CacheOrchestrator  # Legacy aliases
from .agent_service import ConsolidatedAgentService, PydanticAIAgentService, AgentCoordinator  # Legacy aliases

# Existing services (no consolidation needed)
from .infrastructure_service import AsyncInfrastructureService
from .ml_service import MLService

# Infrastructure support services moved to infra/support/
# from infra.support import DataService, CleanupService, PerformanceService

__all__ = [
    # Consolidated services (new unified implementations)
    'ConsolidatedWorkflowService',
    'ConsolidatedQueryService', 
    'ConsolidatedCacheService',
    'ConsolidatedAgentService',
    
    # Existing services (unchanged)
    'AsyncInfrastructureService',
    'MLService',
    
    # Backward compatibility aliases
    'WorkflowService',
    'EnhancedQueryService',
    'RequestOrchestrator', 
    'SimpleCacheService',
    'CacheOrchestrator',
    'PydanticAIAgentService',
    'AgentCoordinator'
]