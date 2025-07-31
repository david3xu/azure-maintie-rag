"""
Essential Agent Types - Preserved from legacy agent interface

This module contains the essential types and classes that are still referenced
by the discovery system and other components. These will gradually be migrated
to PydanticAI equivalents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AgentCapability(Enum):
    """Agent capabilities for dynamic agent selection"""
    SEARCH_ORCHESTRATION = "search_orchestration"
    REASONING_SYNTHESIS = "reasoning_synthesis"
    CONTEXT_MANAGEMENT = "context_management"
    TOOL_DISCOVERY = "tool_discovery"
    DOMAIN_ADAPTATION = "domain_adaptation"


class ReasoningStep(Enum):
    """Reasoning process steps for transparency"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


@dataclass
class AgentContext:
    """Context container for agent reasoning - data-driven design"""
    query: str
    domain: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    search_constraints: Dict[str, Any] = None
    performance_targets: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.search_constraints is None:
            self.search_constraints = {}
        if self.performance_targets is None:
            self.performance_targets = {"max_response_time": 3.0}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReasoningTrace:
    """Trace of agent reasoning process for transparency and learning"""
    step: ReasoningStep
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: float
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResponse:
    """Agent response with reasoning transparency"""
    result: Dict[str, Any]
    reasoning_trace: List[ReasoningTrace]
    confidence: float
    sources: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    suggested_follow_up: List[str] = None
    
    def __post_init__(self):
        if self.suggested_follow_up is None:
            self.suggested_follow_up = []


# Minimal interface for compatibility
class AgentInterface(ABC):
    """Minimal agent interface for compatibility - being replaced by PydanticAI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def process_query(self, context: AgentContext) -> AgentResponse:
        """Process query and return response"""
        pass


# Export essential types
__all__ = [
    'AgentCapability',
    'AgentContext', 
    'AgentResponse',
    'AgentInterface',
    'ReasoningStep',
    'ReasoningTrace'
]