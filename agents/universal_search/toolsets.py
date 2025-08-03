"""
🎯 Universal Search Agent Toolset - Following Target Architecture

This implements the PydanticAI-compliant toolset pattern as specified in:
/docs/implementation/AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md

Target Structure:
agents/universal_search/toolsets.py  # Search-specific Toolset classes

Replaces tools/search_tools.py with proper agent co-location.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

# Import models from agent module
from .agent import (
    QueryRequest,
    TriModalSearchRequest, 
    TriModalSearchResult,
    AgentResponse,
    DomainDetectionResult,
)


class UniversalSearchDeps(BaseModel):
    """Universal Search Agent dependencies following target architecture"""
    azure_services: Optional[any] = None
    cache_manager: Optional[any] = None
    search_orchestrator: Optional[any] = None
    
    class Config:
        arbitrary_types_allowed = True


class UniversalSearchToolset(FunctionToolset):
    """
    🎯 PydanticAI-Compliant Universal Search Toolset
    
    Following AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md target architecture:
    - Search-specific Toolset class in universal_search/toolsets.py
    - Replaces scattered @universal_agent.tool decorators
    - Self-contained with agent co-location
    """

    def __init__(self):
        super().__init__()
        
        # Register core universal search tools
        self.add_function(self.detect_domain, name='detect_domain')
        self.add_function(self.execute_tri_modal_search, name='execute_tri_modal_search')
        self.add_function(self.process_intelligent_query, name='process_intelligent_query')

    async def detect_domain(
        self, ctx: RunContext[UniversalSearchDeps], query: str
    ) -> DomainDetectionResult:
        """Detect domain from query using Domain Intelligence Agent delegation"""
        try:
            # Import here to avoid circular imports during lazy initialization
            from ..domain_intelligence.agent import get_domain_intelligence_agent
            
            domain_agent = get_domain_intelligence_agent()
            
            # Delegate to Domain Intelligence Agent
            result = await domain_agent.run(
                "detect_domain_from_query",
                message_history=[
                    {"role": "user", "content": f"Detect domain from this query: {query}"}
                ]
            )

            # Extract result data
            if hasattr(result, "data") and isinstance(result.data, DomainDetectionResult):
                return result.data
            else:
                # Fallback result
                return DomainDetectionResult(
                    domain="general",
                    confidence=0.3,
                    matched_patterns=[],
                    reasoning="Fallback domain detection",
                    discovered_entities=[],
                )

        except Exception as e:
            # Error fallback
            return DomainDetectionResult(
                domain="general",
                confidence=0.1,
                matched_patterns=[],
                reasoning=f"Error in domain detection: {str(e)}",
                discovered_entities=[],
            )

    async def execute_tri_modal_search(
        self, ctx: RunContext[UniversalSearchDeps], request: TriModalSearchRequest
    ) -> TriModalSearchResult:
        """Execute tri-modal search (Vector + Graph + GNN) with optimal performance"""
        try:
            import time
            start_time = time.time()
            
            # TODO: Implement actual tri-modal search orchestration
            # For now, return mock results to establish the pattern
            
            vector_results = [
                {"score": 0.95, "content": "Vector search result 1", "source": "vector_db"},
                {"score": 0.87, "content": "Vector search result 2", "source": "vector_db"},
            ]
            
            graph_results = [
                {"score": 0.92, "content": "Graph relationship result 1", "source": "graph_db"},
                {"score": 0.84, "content": "Graph relationship result 2", "source": "graph_db"},
            ]
            
            gnn_results = [
                {"score": 0.89, "content": "GNN prediction result 1", "source": "gnn_model"},
                {"score": 0.81, "content": "GNN prediction result 2", "source": "gnn_model"},
            ]
            
            execution_time = time.time() - start_time
            
            return TriModalSearchResult(
                query=request.query,
                domain=request.domain,
                vector_results=vector_results,
                graph_results=graph_results,
                gnn_results=gnn_results,
                synthesis_score=0.91,
                execution_time=execution_time,
            )

        except Exception as e:
            # Return error result
            return TriModalSearchResult(
                query=request.query,
                domain=request.domain or "general",
                vector_results=[],
                graph_results=[],
                gnn_results=[],
                synthesis_score=0.0,
                execution_time=0.0,
            )

    async def process_intelligent_query(
        self, ctx: RunContext[UniversalSearchDeps], query: str, domain: Optional[str] = None
    ) -> AgentResponse:
        """Process query with intelligent domain detection and tri-modal search"""
        try:
            import time
            start_time = time.time()
            
            # Step 1: Detect domain if not provided
            if not domain:
                domain_result = await self.detect_domain(ctx, query)
                domain = domain_result.domain
            
            # Step 2: Execute tri-modal search
            search_request = TriModalSearchRequest(
                query=query,
                domain=domain,
                search_types=["vector", "graph", "gnn"],
                max_results=10
            )
            
            search_result = await self.execute_tri_modal_search(ctx, search_request)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                result=search_result,
                execution_time=execution_time,
                cached=False,
                error=None,
            )

        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0.0
            
            return AgentResponse(
                success=False,
                result=None,
                execution_time=execution_time,
                cached=False,
                error=str(e),
            )


# Create the main toolset instance following target architecture
universal_search_toolset = UniversalSearchToolset()