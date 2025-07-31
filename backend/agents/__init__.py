"""
Universal RAG Agent System - PydanticAI Implementation

This module contains the modern PydanticAI-based agent system for Universal RAG,
implementing intelligent agents with advanced capabilities:

Architecture:
- universal_agent.py: Main PydanticAI agent with 8 registered tools
- tools/: PydanticAI tools (search, discovery, dynamic)
- capabilities/: Domain intelligence capabilities (GNN, vector, graph, knowledge, prompt)
- workflows/: Agent-specific workflow processing
- memory/: Agent memory management
- base/: Essential types and constants (legacy compatibility)
- discovery/: Core discovery components (used by tools)
- search/: Search orchestration components (used by tools)

Features:
- Tri-modal search (Vector + Graph + GNN simultaneously)
- Zero-configuration domain detection and adaptation
- Dynamic tool generation and performance optimization
- Advanced pattern learning and evolution tracking
- Azure-native service integration
- Domain intelligence capabilities
"""

# Import the main PydanticAI agent (requires environment setup)
try:
    from .universal_agent import agent, health_check
except Exception as e:
    # Agent initialization failed - likely missing API keys
    import logging
    logging.warning(f"Universal agent initialization failed: {e}")
    agent = None
    
    def health_check():
        return {"status": "agent_not_initialized", "error": str(e)}

# Import essential types for compatibility
from .base import (
    AgentInterface,
    AgentContext,
    AgentResponse,
    AgentCapability,
    ReasoningStep,
    ReasoningTrace
)

# Import domain intelligence capabilities
from .capabilities import (
    GNNIntelligence,
    VectorIntelligence,
    GraphIntelligence,
    KnowledgeIntelligence,
    PromptIntelligence
)

# Import Azure integration
from .azure_integration import (
    AzureServiceContainer,
    create_azure_service_container
)

__all__ = [
    # Main PydanticAI agent
    'agent',
    'health_check',
    
    # Domain intelligence capabilities
    'GNNIntelligence',
    'VectorIntelligence',
    'GraphIntelligence',
    'KnowledgeIntelligence',
    'PromptIntelligence',
    
    # Azure integration
    'AzureServiceContainer',
    'create_azure_service_container',
    
    # Essential types (legacy compatibility)
    'AgentInterface',
    'AgentContext',
    'AgentResponse', 
    'AgentCapability',
    'ReasoningStep',
    'ReasoningTrace'
]