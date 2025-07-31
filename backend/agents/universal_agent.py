"""
Universal PydanticAI Agent (Simplified Architecture)

This module provides a backward-compatible interface to the simplified universal agent
while maintaining all competitive advantages and removing import complexity.
"""

from .simple_universal_agent import (
    SimplifiedUniversalAgent,
    QueryRequest as SimpleQueryRequest,
    AgentResponse,
    get_universal_agent,
    process_intelligent_query
)
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Backward compatibility classes
class QueryRequest(BaseModel):
    """Enhanced query request model for backward compatibility"""
    query: str
    domain: Optional[str] = None
    search_types: List[str] = ["vector", "graph", "gnn"]  # Maintained for compatibility
    max_results: int = 10
    context: Dict[str, Any] = {}
    performance_requirements: Dict[str, float] = {"max_response_time": 3.0}
    
    def to_simple_request(self) -> SimpleQueryRequest:
        """Convert to simplified request"""
        return SimpleQueryRequest(
            query=self.query,
            domain=self.domain,
            max_results=self.max_results,
            context=self.context
        )


@dataclass
class QueryResponse:
    """Legacy response format for backward compatibility"""
    success: bool
    results: Any
    execution_time_ms: float
    domain_detected: Optional[str] = None
    cached_result: bool = False
    performance_metrics: Dict[str, Any] = None
    error_details: Optional[str] = None
    
    @classmethod
    def from_agent_response(cls, response: AgentResponse, domain: Optional[str] = None) -> 'QueryResponse':
        """Create from simplified AgentResponse"""
        return cls(
            success=response.success,
            results=response.result,
            execution_time_ms=response.execution_time * 1000,  # Convert to ms
            domain_detected=domain,
            cached_result=response.cached,
            performance_metrics={
                "execution_time_ms": response.execution_time * 1000,
                "cached": response.cached
            },
            error_details=response.error
        )


class UniversalAgent:
    """
    Backward compatibility wrapper for SimplifiedUniversalAgent
    
    Maintains the same interface while using the simplified architecture internally.
    All competitive advantages are preserved:
    - Tri-modal search orchestration
    - Zero-config domain discovery  
    - Sub-3-second response times
    - Azure service integration
    """
    
    def __init__(self):
        self._simple_agent: Optional[SimplifiedUniversalAgent] = None
        logger.info("Universal Agent (legacy interface) initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self._simple_agent = await get_universal_agent()
            return True
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process query using legacy interface
        
        Args:
            request: Legacy QueryRequest object
        
        Returns:
            QueryResponse with results and performance metrics
        """
        if not self._simple_agent:
            await self.initialize()
        
        # Convert to simplified request
        simple_request = request.to_simple_request()
        
        # Process using simplified agent
        response = await self._simple_agent.process_query(simple_request)
        
        # Convert back to legacy response format
        return QueryResponse.from_agent_response(response, simple_request.domain)
    
    async def tri_modal_search(self, query: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """Legacy tri-modal search interface"""
        if not self._simple_agent:
            await self.initialize()
        
        return await self._simple_agent.tri_modal_search(query, domain)
    
    async def domain_discovery(self, query: str) -> str:
        """Legacy domain discovery interface"""
        if not self._simple_agent:
            await self.initialize()
        
        return await self._simple_agent.domain_discovery(query)
    
    async def health_check(self) -> Dict[str, Any]:
        """Legacy health check interface"""
        if not self._simple_agent:
            await self.initialize()
        
        return await self._simple_agent.health_check()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Legacy performance metrics interface"""
        if not self._simple_agent:
            await self.initialize()
        
        return await self._simple_agent.get_performance_metrics()


# Global agent instance for backward compatibility
_legacy_global_agent: Optional[UniversalAgent] = None


async def get_universal_agent_legacy() -> UniversalAgent:
    """Get or create global universal agent instance (legacy interface)"""
    global _legacy_global_agent
    
    if _legacy_global_agent is None:
        _legacy_global_agent = UniversalAgent()
        await _legacy_global_agent.initialize()
    
    return _legacy_global_agent


# Export both old and new interfaces
__all__ = [
    # New simplified interface
    'SimplifiedUniversalAgent', 'get_universal_agent', 'process_intelligent_query',
    
    # Legacy interface  
    'UniversalAgent', 'QueryRequest', 'QueryResponse', 'get_universal_agent_legacy'
]