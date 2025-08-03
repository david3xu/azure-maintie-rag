"""
PydanticAI Integration for Azure RAG System

This module registers our consolidated tri-modal search with PydanticAI
while preserving all competitive advantages and maintaining proper layer separation.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Use existing settings instead of non-existent v2_config_models
from config.settings import azure_settings

from .core.azure_services import ConsolidatedAzureServices

# Import our V2 models and configurations
from .models.responses import AgentResponse, TriModalSearchResult
from .tools.search_tools import (
    TriModalSearchRequest,
    TriModalSearchResponse,
    execute_tri_modal_search,
)

logger = logging.getLogger(__name__)


class PydanticAIQueryRequest(BaseModel):
    """Query request model for PydanticAI integration"""

    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    domain: Optional[str] = Field(None, description="Domain context")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")
    include_metadata: bool = Field(default=True, description="Include search metadata")


class PydanticAIQueryResponse(BaseModel):
    """Query response model for PydanticAI integration"""

    query: str = Field(..., description="Original query")
    results: str = Field(..., description="Formatted search results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    competitive_advantages_used: Dict[str, bool] = Field(
        default_factory=dict, description="Competitive advantages utilized"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Performance and quality metrics"
    )


@dataclass
class PydanticAIDependencies:
    """Dependencies for PydanticAI agent"""

    azure_services: ConsolidatedAzureServices
    app_settings: Any  # Using Any instead of non-existent Any


# Create PydanticAI Agent with our consolidated search - temporarily commented for import fix
# azure_rag_agent = Agent[PydanticAIDependencies, PydanticAIQueryResponse](
#     'azure-universal-rag',
#     deps_type=PydanticAIDependencies,
#     result_type=PydanticAIQueryResponse,
#     system_prompt="""
#     You are an intelligent Azure RAG system with unique competitive advantages:
#
#     1. **Tri-Modal Search Unity**: Simultaneously execute Vector + Graph + GNN search
#     2. **Sub-3-Second Response**: Guaranteed response time under 3 seconds
#     3. **Zero-Config Discovery**: Automatic domain detection without configuration
#     4. **Azure-Native Integration**: Deep integration with Azure AI services
#
#     Your role is to:
#     - Execute intelligent search using all available modalities
#     - Provide comprehensive, accurate responses
#     - Maintain performance SLAs while maximizing quality
#     - Leverage Azure services for enhanced capabilities
#
#     Always prioritize accuracy and relevance while maintaining performance guarantees.
#     """
# )


from .domain_intelligence.pydantic_tools import discover_domain_tool
from .knowledge_extraction.pydantic_tools import (
    build_knowledge_graph_tool,
    extract_entities_tool,
    extract_relationships_tool,
)

# Import distributed PydanticAI tools
from .universal_search.pydantic_tools import search_with_tri_modal_tool


# Tool registration for tri-modal search
# @azure_rag_agent.tool
async def search_with_tri_modal(
    ctx: RunContext[PydanticAIDependencies],
    query: str,
    domain: Optional[str] = None,
    max_results: int = 10,
) -> TriModalSearchResponse:
    """
    Execute tri-modal search with competitive advantage preservation.

    This tool provides access to our core competitive advantage:
    simultaneous Vector + Graph + GNN search with intelligent result synthesis.
    """
    # Delegate to the specialized tool in universal_search
    return await search_with_tri_modal_tool(
        ctx, query, domain, max_results, ctx.deps.azure_services
    )


# Tool registration for domain discovery
# @azure_rag_agent.tool
async def discover_domain(ctx: RunContext[PydanticAIDependencies], text: str) -> str:
    """
    Zero-configuration domain discovery using statistical pattern learning.

    This tool provides our competitive advantage of automatic domain detection
    without requiring any predefined configurations or rules.
    """
    # Delegate to the specialized tool in domain_intelligence
    return await discover_domain_tool(ctx, text, ctx.deps.azure_services)


# Tool registration for entity extraction
# @azure_rag_agent.tool
async def extract_entities(
    ctx: RunContext[PydanticAIDependencies],
    text: str,
    domain: Optional[str] = None,
    extraction_strategy: str = "hybrid",
) -> dict:
    """
    Extract entities from text using advanced extraction strategies.

    This tool provides enterprise-grade entity extraction capabilities.
    """
    # Delegate to the specialized tool in knowledge_extraction
    result = await extract_entities_tool(
        ctx, text, domain, extraction_strategy, ctx.deps.azure_services
    )

    # Convert to dict for PydanticAI compatibility
    return {
        "entities": result.entities,
        "confidence": result.average_confidence,
        "metadata": result.extraction_metadata,
    }


# Tool registration for relationship extraction
# @azure_rag_agent.tool
async def extract_relationships(
    ctx: RunContext[PydanticAIDependencies],
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    domain: Optional[str] = None,
) -> dict:
    """
    Extract relationships from text using advanced relationship detection.

    This tool provides enterprise-grade relationship extraction capabilities.
    """
    # Delegate to the specialized tool in knowledge_extraction
    result = await extract_relationships_tool(
        ctx, text, entities, domain, "semantic", ctx.deps.azure_services
    )

    # Convert to dict for PydanticAI compatibility
    return {
        "relationships": result.relationships,
        "confidence": result.average_confidence,
        "metadata": result.extraction_metadata,
    }


# Main query processing function
# @azure_rag_agent.run
async def process_query(
    ctx: RunContext[PydanticAIDependencies], user_prompt: str
) -> PydanticAIQueryResponse:
    """
    Main query processing with full competitive advantage utilization.

    This function orchestrates our complete intelligent RAG workflow:
    1. Zero-config domain discovery (if needed)
    2. Tri-modal search execution (Vector + Graph + GNN)
    3. Intelligent result synthesis
    4. Performance SLA compliance validation
    """
    import time

    start_time = time.time()

    logger.info(f"PydanticAI processing query: {user_prompt[:100]}...")

    try:
        # Step 1: Domain Discovery (competitive advantage)
        discovered_domain = await discover_domain(ctx, user_prompt)

        # Step 2: Tri-Modal Search (core competitive advantage)
        search_result = await search_with_tri_modal(
            ctx, query=user_prompt, domain=discovered_domain, max_results=10
        )

        # Step 3: Calculate execution time and validate SLA
        execution_time = time.time() - start_time

        # Step 4: Track competitive advantages used
        competitive_advantages_used = {
            "tri_modal_search": True,
            "zero_config_discovery": True,
            "azure_native_integration": True,
            "sub_3s_response_sla": execution_time < 3.0,
        }

        # Step 5: Compile performance metrics
        performance_metrics = {
            "execution_time": execution_time,
            "search_confidence": search_result.confidence,
            "domain_discovered": discovered_domain,
            "modalities_used": search_result.modality_contributions.keys()
            if search_result.modality_contributions
            else [],
            "sla_compliance": search_result.performance_met,
            "competitive_advantage_score": ctx.deps.app_settings.overall_competitive_score,
        }

        # Step 6: Format response
        response = PydanticAIQueryResponse(
            query=user_prompt,
            results=search_result.results,
            confidence=search_result.confidence,
            execution_time=execution_time,
            competitive_advantages_used=competitive_advantages_used,
            performance_metrics=performance_metrics,
        )

        logger.info(
            f"PydanticAI query completed successfully: {execution_time:.2f}s, "
            f"confidence: {search_result.confidence:.2f}"
        )

        return response

    except Exception as e:
        execution_time = time.time() - start_time

        logger.error(f"PydanticAI query processing failed: {e}")

        # Return error response with metrics
        return PydanticAIQueryResponse(
            query=user_prompt,
            results=f"Error processing query: {str(e)}",
            confidence=0.0,
            execution_time=execution_time,
            competitive_advantages_used={
                "tri_modal_search": False,
                "zero_config_discovery": False,
                "azure_native_integration": False,
                "sub_3s_response_sla": False,
            },
            performance_metrics={"execution_time": execution_time, "error": str(e)},
        )


async def create_pydantic_ai_agent(
    azure_services: ConsolidatedAzureServices, app_settings: Optional[Any] = None
) -> Agent:
    """
    Create and configure PydanticAI agent with Azure dependencies.

    Args:
        azure_services: Consolidated Azure services container
        app_settings: V2 application settings (optional, will create default if None)

    Returns:
        Configured PydanticAI agent ready for query processing
    """
    if app_settings is None:
        app_settings = Any()

    # Create dependencies
    dependencies = PydanticAIDependencies(
        azure_services=azure_services, app_settings=app_settings
    )

    logger.info(
        f"Created PydanticAI agent with competitive advantage score: "
        f"{app_settings.overall_competitive_score:.2f}"
    )

    return azure_rag_agent


async def process_intelligent_query(
    query: str,
    azure_services: ConsolidatedAzureServices,
    app_settings: Optional[Any] = None,
) -> PydanticAIQueryResponse:
    """
    Convenience function for processing queries through PydanticAI integration.

    Args:
        query: User query string
        azure_services: Azure services container
        app_settings: Application settings (optional)

    Returns:
        Query response with competitive advantage metrics
    """
    # Create agent
    agent = await create_pydantic_ai_agent(azure_services, app_settings)

    # Create dependencies
    dependencies = PydanticAIDependencies(
        azure_services=azure_services, app_settings=app_settings or Any()
    )

    # Process query
    result = await agent.run(query, deps=dependencies)

    return result


# Export functions
__all__ = [
    "azure_rag_agent",
    "create_pydantic_ai_agent",
    "process_intelligent_query",
    "PydanticAIQueryRequest",
    "PydanticAIQueryResponse",
    "PydanticAIDependencies",
]


# Test function for development
async def test_pydantic_ai_integration():
    """Test PydanticAI integration functionality"""
    from ..config.v2_config_models import Any
    from .core.azure_services import create_azure_service_container

    print("Testing PydanticAI Integration...")

    try:
        # Create mock Azure services (for testing)
        azure_services = await create_azure_service_container()

        # Create V2 settings
        settings = Any()

        # Test query processing
        result = await process_intelligent_query(
            "What are the best practices for machine learning in Azure?",
            azure_services,
            settings,
        )

        print(f"✅ Query processed successfully!")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        print(f"   Competitive advantages: {result.competitive_advantages_used}")
        print(f"   Performance metrics: {result.performance_metrics}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_pydantic_ai_integration())
