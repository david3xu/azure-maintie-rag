"""
🎯 Universal Search Agent Toolset - Following Target Architecture

This implements the PydanticAI-compliant toolset pattern as specified in:
/docs/implementation/AGENT_BOUNDARY_FIXES_IMPLEMENTATION.md

Target Structure:
agents/universal_search/toolsets.py  # Search-specific Toolset classes

Replaces tools/search_tools.py with proper agent co-location.
"""

from typing import Dict, List, Optional
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

# Import models from centralized data models
from agents.core.data_models import (
    QueryRequest,
    TriModalSearchRequest,
    TriModalSearchResult,
    SearchResponse as AgentResponse,
    DomainDetectionResult,
    UniversalSearchDeps
)
from agents.core.constants import UniversalSearchConstants, UniversalSearchToolsetConstants


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
                # Use fallback values from centralized configuration
                return DomainDetectionResult(
                    domain=UniversalSearchToolsetConstants.FALLBACK_DOMAIN,
                    confidence=UniversalSearchToolsetConstants.FALLBACK_CONFIDENCE_HIGH,
                    matched_patterns=[],
                    reasoning=UniversalSearchToolsetConstants.FALLBACK_REASONING_TEMPLATE,
                    discovered_entities=[],
                )

        except Exception as e:
            # Use error fallback values from centralized configuration
            return DomainDetectionResult(
                domain=UniversalSearchToolsetConstants.FALLBACK_DOMAIN,
                confidence=UniversalSearchToolsetConstants.FALLBACK_CONFIDENCE_ERROR,
                matched_patterns=[],
                reasoning=UniversalSearchToolsetConstants.ERROR_REASONING_TEMPLATE.format(error=str(e)),
                discovered_entities=[],
            )

    async def execute_tri_modal_search(
        self, ctx: RunContext[UniversalSearchDeps], request: TriModalSearchRequest
    ) -> TriModalSearchResult:
        """Execute tri-modal search (Vector + Graph + GNN) with optimal performance"""
        try:
            import time
            start_time = time.time()
            
            # Real tri-modal search orchestration using actual Azure services
            from agents.universal_search.orchestrators.consolidated_search_orchestrator import ConsolidatedSearchOrchestrator
            from config.centralized_config import get_search_config
            
            # Get dynamic search configuration for domain
            search_config = get_search_config(request.domain or "general", request.query)
            
            # Initialize real search orchestrator
            orchestrator = ConsolidatedSearchOrchestrator()
            
            # Execute actual tri-modal search with learned parameters
            search_result = await orchestrator.execute_tri_modal_search(
                query=request.query,
                domain=request.domain or "general",
                search_types=request.search_types or ["vector", "graph", "gnn"],
                max_results=request.max_results or 10
            )
            
            execution_time = time.time() - start_time
            
            return TriModalSearchResult(
                query=request.query,
                domain=request.domain,
                vector_results=search_result.vector_results,
                graph_results=search_result.graph_results,
                gnn_results=search_result.gnn_results,
                synthesis_score=search_result.synthesis_score,
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
            
            # Step 2: Execute tri-modal search with domain-optimized parameters
            # Load search configuration from centralized system
            from config.centralized_config import get_search_config
            search_config = get_search_config(domain, query)
            
            search_request = TriModalSearchRequest(
                query=query,
                domain=domain,
                search_types=search_config.enabled_search_types,
                max_results=search_config.max_results_per_query
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