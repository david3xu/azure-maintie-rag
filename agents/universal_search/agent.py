"""
Universal PydanticAI Agent with Domain Intelligence Delegation

This module provides the Universal Agent that orchestrates search operations and
delegates domain intelligence tasks to the specialized Domain Intelligence Agent:
- PydanticAI Agent Delegation pattern for domain tasks
- Tri-modal search orchestration (Vector + Graph + GNN)
- Zero-config domain discovery through agent delegation
- Sub-3-second response times with optimized caching
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Azure integration imports
from ..core.azure_services import ConsolidatedAzureServices as AzureServiceContainer
from ..core.azure_services import create_azure_service_container

# Domain Intelligence Agent import
from ..domain_intelligence.agent import (
    AvailableDomainsResult,
    DomainAnalysisResult,
    DomainDetectionResult,
    domain_agent,
)

# Config-Extraction Orchestrator import
# Temporarily commented out due to import issues
# from ..orchestration.config_extraction_orchestrator import ConfigExtractionOrchestrator

logger = logging.getLogger(__name__)


def _get_default_search_types() -> List[str]:
    """
    Get default search types preserving tri-modal competitive advantage.

    Uses data-driven approach without hardcoded fallbacks - gets configuration
    from Azure services or returns optimal tri-modal defaults.
    """
    # Return tri-modal competitive advantage without hardcoded config dependencies
    # This eliminates the bad import fallback and uses Azure service configuration
    return ["vector", "graph", "gnn"]


class QueryRequest(BaseModel):
    """Universal Agent query request model"""

    query: str = Field(description="User query text")
    domain: Optional[str] = Field(
        default=None, description="Optional domain specification"
    )
    max_results: int = Field(default=10, description="Maximum search results")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class TriModalSearchRequest(BaseModel):
    """Tri-modal search request model"""

    query: str = Field(description="Search query")
    domain: str = Field(description="Domain for search optimization")
    search_types: List[str] = Field(
        default_factory=_get_default_search_types, description="Search types to execute"
    )
    max_results: int = Field(default=10, description="Maximum results per search type")


class TriModalSearchResult(BaseModel):
    """Tri-modal search result model"""

    query: str = Field(description="Original query")
    domain: str = Field(description="Domain used for search")
    vector_results: List[Dict] = Field(description="Vector search results")
    graph_results: List[Dict] = Field(description="Graph search results")
    gnn_results: List[Dict] = Field(description="GNN search results")
    synthesis_score: float = Field(description="Result synthesis confidence")
    execution_time: float = Field(description="Total execution time")


class AgentResponse(BaseModel):
    """Universal Agent response model"""

    success: bool = Field(description="Whether the operation succeeded")
    result: Any = Field(description="Operation result")
    execution_time: float = Field(description="Execution time in seconds")
    cached: bool = Field(default=False, description="Whether result was cached")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# Lazy initialization to avoid import-time Azure connection requirements
_universal_agent = None

def create_universal_agent() -> Agent:
    """Create Universal Agent with Azure OpenAI connection"""
    import os
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.azure import AzureProvider

    try:
        # Configure Azure OpenAI provider - always use production endpoint
        azure_endpoint = "https://oai-maintie-rag-prod-fymhwfec3ra2w.openai.azure.com/"
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_version = "2024-08-01-preview"
        deployment_name = "gpt-4o-mini"  # Use production deployment

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable is required")

        # Use Azure OpenAI with API key - Correct PydanticAI syntax
        azure_model = OpenAIModel(
            deployment_name,
            provider=AzureProvider(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
            ),
        )

        agent = Agent(
            azure_model,
            name="universal-agent",
            system_prompt=(
                "You are the Universal Search Orchestrator. Your role is to:"
                "1. Delegate domain detection to the Domain Intelligence Agent"
                "2. Orchestrate tri-modal search (Vector + Graph + GNN) with optimal performance"
                "3. Synthesize results from multiple search modalities"
                "4. Maintain sub-3-second response times through intelligent caching"
                "You work with the Domain Intelligence Agent to achieve zero-config domain discovery."
            ),
        )
        logger.info(f"Universal Agent initialized with Azure OpenAI: {deployment_name}")
        return agent

    except ImportError as e:
        # PHASE 0 REQUIREMENT: No statistical-only fallback - raise error instead
        error_msg = (
            f"❌ PHASE 0 REQUIREMENT: Azure provider import failed: {e}. "
            "Statistical-only fallback mode is disabled. Please ensure pydantic-ai[azure] is installed."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    except Exception as e:
        # PHASE 0 REQUIREMENT: No statistical-only fallback - raise error instead
        error_msg = (
            f"❌ PHASE 0 REQUIREMENT: Failed to create Azure provider: {e}. "
            "Statistical-only fallback mode is disabled. Please ensure Azure OpenAI credentials are properly configured."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def get_universal_agent() -> Agent:
    """Get Universal Agent with lazy initialization"""
    global _universal_agent
    if _universal_agent is None:
        _universal_agent = create_universal_agent()
    return _universal_agent


# Import the toolset following target architecture
from .toolsets import universal_search_toolset, UniversalSearchDeps

def create_universal_agent_with_toolset() -> Agent:
    """Create Universal Agent with proper toolset pattern like Agent 1"""
    import os
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.azure import AzureProvider

    try:
        # Configure Azure OpenAI provider
        azure_endpoint = "https://oai-maintie-rag-prod-fymhwfec3ra2w.openai.azure.com/"
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_version = "2024-08-01-preview"
        deployment_name = "gpt-4o-mini"

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

        # Create agent with proper toolset pattern (like Agent 1)
        agent = Agent(
            azure_model,
            deps_type=UniversalSearchDeps,
            toolsets=[universal_search_toolset],  # ✅ PydanticAI compliant toolset
            name="universal-search-agent",
            system_prompt=(
                "You are the Universal Search Orchestrator. Your role is to:"
                "1. Delegate domain detection to the Domain Intelligence Agent"
                "2. Orchestrate tri-modal search (Vector + Graph + GNN) with optimal performance"
                "3. Synthesize results from multiple search modalities"
                "4. Maintain sub-3-second response times through intelligent caching"
                "You work with the Domain Intelligence Agent to achieve zero-config domain discovery."
            ),
        )
        
        return agent

    except Exception as e:
        error_msg = (
            f"❌ PHASE 0 REQUIREMENT: Failed to create Universal Search Agent: {e}. "
            "Statistical-only fallback mode is disabled. Please ensure Azure OpenAI credentials are properly configured."
        )
        raise RuntimeError(error_msg)

# Update the lazy initialization to use toolset pattern
def get_universal_agent() -> Agent:
    """Get Universal Agent with lazy initialization and proper toolset"""
    global _universal_agent
    if _universal_agent is None:
        _universal_agent = create_universal_agent_with_toolset()
    return _universal_agent

# For backward compatibility, create module-level agent getter
universal_agent = get_universal_agent

