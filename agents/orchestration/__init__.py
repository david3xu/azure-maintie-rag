"""
Orchestration Module - Unified Coordination Layer

This module provides centralized orchestration for all Azure Universal RAG
operations, coordinating the complete workflow from user queries to final responses.

Components:
- ConfigExtractionOrchestrator: Config-Extraction workflow coordination
- SearchOrchestrator: Unified search coordination across all modalities
- WorkflowOrchestrator: End-to-end workflow management and coordination
- PydanticAI Integration: Enterprise agent delegation and coordination

Key Features:
- Complete workflow orchestration from query to response
- Multi-agent coordination with proper boundary enforcement
- Performance optimization and monitoring across all stages
- Error handling and recovery mechanisms
- Azure service integration and resource management
- PydanticAI enterprise agent delegation
"""

# Import orchestration components
from .config_extraction_orchestrator import (
    ConfigExtractionOrchestrator,
    process_domain_with_config_extraction,
)
from .pydantic_integration import (
    AgentDelegationRequest,
    AgentDelegationResult,
    AgentDelegationStrategy,
    PydanticAgentConfig,
    PydanticAIWorkflowIntegration,
)
from .search_orchestrator import (
    ModalityResult,
    SearchOrchestrator,
    SearchRequest,
    SearchResults,
    SearchStrategy,
    execute_unified_search,
    get_search_orchestrator,
)
from .workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowProgress,
    WorkflowRequest,
    WorkflowResults,
    WorkflowStage,
    WorkflowStatus,
    execute_complete_workflow,
    get_workflow_orchestrator,
)

__all__ = [
    # Config-Extraction Orchestrator
    "ConfigExtractionOrchestrator",
    "process_domain_with_config_extraction",
    # Search Orchestrator
    "SearchOrchestrator",
    "SearchRequest",
    "SearchResults",
    "ModalityResult",
    "SearchStrategy",
    "execute_unified_search",
    "get_search_orchestrator",
    # Workflow Orchestrator
    "WorkflowOrchestrator",
    "WorkflowRequest",
    "WorkflowResults",
    "WorkflowProgress",
    "WorkflowStage",
    "WorkflowStatus",
    "execute_complete_workflow",
    "get_workflow_orchestrator",
    # PydanticAI Integration
    "PydanticAIWorkflowIntegration",
    "AgentDelegationRequest",
    "AgentDelegationResult",
    "AgentDelegationStrategy",
    "PydanticAgentConfig",
]
