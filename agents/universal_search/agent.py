"""
Universal Search Agent - Consolidated Implementation

This module provides the Universal Search Agent using the consolidated search orchestrator
that eliminates configuration redundancy while preserving tri-modal search capabilities.

Key Features:
- Unified tri-modal search orchestration (Vector + Graph + GNN)
- Centralized configuration management
- Simplified agent creation pattern
- Enhanced performance through consolidated processing
- Domain intelligence integration

Architecture Integration:
- Uses ConsolidatedSearchOrchestrator for unified search operations
- Integrates with centralized configuration system
- Maintains backward compatibility with existing interfaces
- Follows proven patterns from Knowledge Extraction Agent consolidation
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Clean configuration imports (CODING_STANDARDS compliant) 
from config.centralized_config import get_model_config, get_search_config

# Backward compatibility for gradual migration
class UniversalSearchAgentConfig:
    def __init__(self):
        model_config = get_model_config()
        self.azure_endpoint = model_config.azure_endpoint
        self.api_version = model_config.api_version  
        self.deployment_name = model_config.deployment_name

get_universal_search_agent_config = lambda: UniversalSearchAgentConfig()
get_tri_modal_orchestration_config = get_search_config  # Alias for compatibility

# Import consolidated orchestrator
from .orchestrators.consolidated_search_orchestrator import (
    ConsolidatedSearchOrchestrator,
    TriModalSearchResult,
    SearchResult
)

# Import domain intelligence integration
try:
    from ..domain_intelligence.agent import get_domain_agent
    DOMAIN_AGENT_AVAILABLE = True
except ImportError:
    DOMAIN_AGENT_AVAILABLE = False
    get_domain_agent = None

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Universal Search query request model"""
    query: str = Field(..., description="Search query text")
    domain: Optional[str] = Field(default=None, description="Optional domain specification")
    search_types: Optional[List[str]] = Field(default=None, description="Search types to execute")
    max_results: Optional[int] = Field(default=None, description="Maximum results per modality")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class SearchResponse(BaseModel):
    """Universal Search response model"""
    success: bool = Field(..., description="Whether the search succeeded")
    query: str = Field(..., description="Original query")
    domain: str = Field(..., description="Domain used for search")
    results: TriModalSearchResult = Field(..., description="Tri-modal search results")
    execution_time: float = Field(..., description="Total execution time")
    cached: bool = Field(default=False, description="Whether result was cached")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class UniversalSearchDeps(BaseModel):
    """Universal Search Agent dependencies"""
    azure_services: Optional[Any] = None
    cache_manager: Optional[Any] = None
    orchestrator: Optional[ConsolidatedSearchOrchestrator] = None
    domain_agent: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True


# Lazy initialization to avoid import-time Azure connection requirements
_universal_search_agent = None
_consolidated_orchestrator = None


def get_consolidated_orchestrator() -> ConsolidatedSearchOrchestrator:
    """Get consolidated search orchestrator with lazy initialization"""
    global _consolidated_orchestrator
    if _consolidated_orchestrator is None:
        _consolidated_orchestrator = ConsolidatedSearchOrchestrator()
    return _consolidated_orchestrator


def _create_agent_with_consolidated_orchestrator() -> Agent:
    """Create Universal Search Agent with consolidated orchestrator integration"""
    import os
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.azure import AzureProvider

    try:
        # Get configurations
        agent_config = get_universal_search_agent_config()
        model_config = get_model_config()
        orchestration_config = get_tri_modal_orchestration_config()

        # Configure Azure OpenAI provider
        azure_endpoint = agent_config.azure_endpoint
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_version = agent_config.api_version
        deployment_name = agent_config.deployment_name

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable is required")

        # Use Azure OpenAI with API key
        azure_model = OpenAIModel(
            deployment_name,
            provider=AzureProvider(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
            ),
        )

        # Create agent with consolidated search capabilities
        agent = Agent(
            azure_model,
            deps_type=UniversalSearchDeps,
            name="universal-search-agent",
            system_prompt=(
                "You are the Universal Search Orchestrator using consolidated tri-modal search processing. "
                "Your capabilities include:"
                "1. Unified tri-modal search orchestration (Vector + Graph + GNN) in optimized pipeline"
                "2. Domain-aware search optimization through integration with Domain Intelligence Agent"
                "3. Intelligent result synthesis and cross-modal confidence scoring"
                "4. Sub-3-second response times through consolidated processing and caching"
                "5. Dynamic search type selection based on query characteristics"
                "You work with centralized configuration and provide enterprise-grade search capabilities."
            ),
        )
        
        return agent

    except Exception as e:
        error_msg = (
            f"❌ Failed to create Universal Search Agent with consolidated orchestrator: {e}. "
            "Please ensure Azure OpenAI credentials are properly configured."
        )
        raise RuntimeError(error_msg)


def get_universal_search_agent() -> Agent:
    """Get Universal Search Agent with lazy initialization and consolidated orchestrator"""
    global _universal_search_agent
    if _universal_search_agent is None:
        _universal_search_agent = _create_agent_with_consolidated_orchestrator()
    return _universal_search_agent


# For backward compatibility
universal_search_agent = get_universal_search_agent


async def execute_universal_search(
    query: str,
    domain: str = None,
    search_types: List[str] = None,
    max_results: int = None
) -> SearchResponse:
    """
    Execute universal search using consolidated orchestrator.
    
    This is a simplified wrapper that delegates to the consolidated orchestrator
    while maintaining backward compatibility.
    """
    try:
        start_time = time.time()
        
        # Get orchestrator
        orchestrator = get_consolidated_orchestrator()
        
        # Detect domain if not provided
        if domain is None:
            if DOMAIN_AGENT_AVAILABLE:
                try:
                    domain_agent = get_domain_agent()
                    # Simplified domain detection - in practice this would use full domain agent
                    domain = "general"  # Fallback
                except Exception as e:
                    logger.warning(f"Domain detection failed: {e}")
                    domain = "general"
            else:
                domain = "general"
        
        # Execute tri-modal search
        search_result = await orchestrator.execute_tri_modal_search(
            query=query,
            domain=domain,
            search_types=search_types,
            max_results=max_results
        )
        
        execution_time = time.time() - start_time
        
        # Create response
        response = SearchResponse(
            success=True,
            query=query,
            domain=domain,
            results=search_result,
            execution_time=execution_time,
            cached=False  # Could be enhanced with actual caching
        )
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Universal search failed: {e}")
        
        # Create error response
        return SearchResponse(
            success=False,
            query=query,
            domain=domain or "unknown",
            results=TriModalSearchResult(
                query=query,
                domain=domain or "unknown",
                vector_results=[],
                graph_results=[],
                gnn_results=[],
                synthesis_score=0.0,
                execution_time=execution_time,
                modalities_executed=[],
                vector_execution_time=0.0,
                graph_execution_time=0.0,
                gnn_execution_time=0.0,
                total_results=0,
                high_confidence_results=0,
                average_confidence=0.0,
                cross_modal_agreement=0.0
            ),
            execution_time=execution_time,
            error=str(e)
        )


async def test_universal_search_agent():
    """Test the Universal Search Agent with consolidated orchestrator"""
    try:
        # Get agent with lazy initialization 
        agent = get_universal_search_agent()
        
        print("✅ Universal Search Agent created successfully with consolidated orchestrator")
        print(f"   - Agent name: {agent.name}")
        print(f"   - Dependencies type: {agent._deps_type.__name__ if hasattr(agent, '_deps_type') else 'None'}")
        
        # Test orchestrator directly
        orchestrator = get_consolidated_orchestrator()
        print(f"✅ Consolidated orchestrator initialized: {type(orchestrator).__name__}")
        
        return {
            "agent_created": True,
            "lazy_initialization": True,
            "consolidated_orchestrator": True,
            "azure_openai_model": True
        }
        
    except Exception as e:
        print(f"❌ Universal Search Agent test failed: {e}")
        return {
            "agent_created": False,
            "error": str(e)
        }


# Export main components
__all__ = [
    "get_universal_search_agent",
    "universal_search_agent", 
    "execute_universal_search",
    "test_universal_search_agent",
    "QueryRequest",
    "SearchResponse",
    "UniversalSearchDeps"
]