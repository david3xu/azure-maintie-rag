"""
Agent Interface - Abstract base class for all intelligent agents in Universal RAG system.
Follows Rule 1 (Data-Driven) and Rule 2 (Clean Architecture) from CODING_STANDARDS.md
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


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


class AgentInterface(ABC):
    """
    Abstract base class for all intelligent agents in the Universal RAG system.
    
    Design Principles:
    - Data-driven: All behavior controlled by configuration and context
    - Clean Architecture: Clear separation of concerns and dependencies
    - Async-first: Non-blocking operations for scalability
    - Transparent: Full reasoning trace for explainability
    - Adaptive: Self-configuring based on domain and performance data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent with data-driven configuration.
        
        Args:
            config: Agent configuration dictionary (no hardcoded values)
        """
        self.config = config
        self.capabilities = self._extract_capabilities(config)
        self.performance_targets = config.get("performance_targets", {"max_response_time": 3.0})
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize agent resources asynchronously"""
        if self._is_initialized:
            return
            
        await self._initialize_resources()
        self._is_initialized = True
        self.logger.info(f"{self.__class__.__name__} initialized with capabilities: {[c.value for c in self.capabilities]}")
    
    @abstractmethod
    async def _initialize_resources(self) -> None:
        """Initialize agent-specific resources - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _extract_capabilities(self, config: Dict[str, Any]) -> List[AgentCapability]:
        """Extract agent capabilities from configuration - implemented by subclasses"""
        pass
    
    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Main processing method - core agent intelligence.
        
        Args:
            context: Agent context with query and reasoning state
            
        Returns:
            AgentResponse with results and reasoning trace
        """
        pass
    
    async def can_handle(self, context: AgentContext) -> Dict[str, Any]:
        """
        Determine if agent can handle the given context.
        
        Args:
            context: Agent context to evaluate
            
        Returns:
            Dictionary with capability assessment:
            - can_handle: bool
            - confidence: float (0-1)
            - estimated_performance: Dict[str, Any]
            - required_resources: List[str]
        """
        # Default implementation - subclasses should override for specific logic
        domain_match = self._assess_domain_match(context)
        complexity_fit = self._assess_complexity_fit(context)
        resource_availability = await self._assess_resource_availability()
        
        confidence = (domain_match + complexity_fit + resource_availability) / 3
        
        return {
            "can_handle": confidence > 0.5,
            "confidence": confidence,
            "estimated_performance": {
                "response_time": self._estimate_response_time(context),
                "quality_score": confidence * 0.9  # Conservative estimate
            },
            "required_resources": self._get_required_resources(context)
        }
    
    async def stream_reasoning(self, context: AgentContext) -> AsyncIterator[ReasoningTrace]:
        """
        Stream reasoning process in real-time for transparency.
        
        Args:
            context: Agent context for reasoning
            
        Yields:
            ReasoningTrace objects as reasoning progresses
        """
        # Default implementation - can be overridden for streaming
        response = await self.process(context)
        for trace in response.reasoning_trace:
            yield trace
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return self.capabilities
    
    def get_performance_targets(self) -> Dict[str, Any]:
        """Get performance targets from configuration"""
        return self.performance_targets
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform agent health check.
        
        Returns:
            Health status dictionary
        """
        if not self._is_initialized:
            return {
                "healthy": False,
                "status": "not_initialized",
                "message": "Agent not initialized"
            }
        
        try:
            resource_health = await self._check_resource_health()
            return {
                "healthy": resource_health.get("healthy", True),
                "status": "operational",
                "capabilities": [c.value for c in self.capabilities],
                "resource_status": resource_health,
                "performance_targets": self.performance_targets
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "message": str(e)
            }
    
    # Protected helper methods for subclasses
    
    def _assess_domain_match(self, context: AgentContext) -> float:
        """Assess how well agent domain matches context domain"""
        agent_domains = self.config.get("supported_domains", ["general"])
        context_domain = context.domain or "general"
        
        if context_domain in agent_domains or "general" in agent_domains:
            return 0.9
        return 0.3  # Can attempt but not optimal
    
    def _assess_complexity_fit(self, context: AgentContext) -> float:
        """Assess if agent can handle query complexity"""
        query_length = len(context.query.split())
        max_complexity = self.config.get("max_query_complexity", 100)
        
        if query_length <= max_complexity:
            return 1.0
        return max(0.2, max_complexity / query_length)
    
    async def _assess_resource_availability(self) -> float:
        """Assess resource availability for processing"""
        # Default implementation - subclasses can override
        return 0.8  # Assume resources generally available
    
    def _estimate_response_time(self, context: AgentContext) -> float:
        """Estimate response time based on context complexity"""
        base_time = self.config.get("base_response_time", 1.0)
        complexity_factor = len(context.query.split()) / 20.0
        return base_time * (1 + complexity_factor)
    
    def _get_required_resources(self, context: AgentContext) -> List[str]:
        """Get required resources for processing context"""
        return self.config.get("required_resources", ["cpu", "memory"])
    
    async def _check_resource_health(self) -> Dict[str, Any]:
        """Check health of agent resources - can be overridden"""
        return {"healthy": True, "details": "Default health check"}
    
    def _create_reasoning_trace(
        self, 
        step: ReasoningStep, 
        description: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_ms: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Helper to create reasoning trace entries"""
        return ReasoningTrace(
            step=step,
            description=description,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms,
            confidence=confidence,
            metadata=metadata or {}
        )


__all__ = [
    'AgentInterface', 
    'AgentContext', 
    'AgentResponse', 
    'ReasoningTrace', 
    'AgentCapability', 
    'ReasoningStep'
]