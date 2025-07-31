"""
Agent Capabilities

This module contains domain-specific intelligence capabilities that were moved
from the services layer to properly align with agent-centric architecture.

Domain intelligence capabilities:
- GNN Intelligence: Graph Neural Network reasoning
- Vector Intelligence: Vector search and similarity  
- Graph Intelligence: Knowledge graph operations
- Knowledge Intelligence: Knowledge extraction and processing
- Prompt Intelligence: Intelligent prompt processing
"""

# Import domain intelligence capabilities
# Note: These are the actual service classes moved from services layer
try:
    from .gnn_intelligence import GNNService as GNNIntelligence
except ImportError:
    GNNIntelligence = None

try:
    from .vector_intelligence import VectorService as VectorIntelligence  
except ImportError:
    VectorIntelligence = None

try:
    from .graph_intelligence import GraphService as GraphIntelligence
except ImportError:
    GraphIntelligence = None

try:
    from .knowledge_intelligence import KnowledgeService as KnowledgeIntelligence
except ImportError:
    KnowledgeIntelligence = None

try:
    from .prompt_intelligence import PromptService as PromptIntelligence
except ImportError:
    PromptIntelligence = None

__all__ = [
    # Domain intelligence capabilities (former services, now properly placed)
    'GNNIntelligence',
    'VectorIntelligence', 
    'GraphIntelligence',
    'KnowledgeIntelligence',
    'PromptIntelligence'
]