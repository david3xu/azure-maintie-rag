"""
Universal Search Agent - Multi-Modal Search Orchestration
========================================================

This agent orchestrates tri-modal search (Vector + Graph + GNN) WITHOUT domain assumptions.
Follows PydanticAI best practices with proper agent delegation and centralized dependencies.

Key Principles:
- Universal search patterns that work for ANY domain
- Proper agent delegation to Domain Intelligence and Knowledge Extraction agents
- Uses centralized dependencies (no duplicate Azure clients)
- Atomic tools with single responsibilities
- Real Azure service integration (Cognitive Search, Cosmos DB, Azure ML)
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.core.universal_models import (
    SearchConfiguration,
    SearchResult,
    UniversalDomainAnalysis,
)
from agents.domain_intelligence.agent import domain_intelligence_agent
from agents.knowledge_extraction.agent import knowledge_extraction_agent
from agents.shared.query_tools import generate_gremlin_query, generate_search_query


class MultiModalSearchResult(BaseModel):
    """Results from multi-modal search across Vector + Graph + GNN."""

    vector_results: List[SearchResult] = Field(default_factory=list)
    graph_results: List[Dict[str, Any]] = Field(default_factory=list)
    gnn_results: List[Dict[str, Any]] = Field(default_factory=list)
    unified_results: List[SearchResult] = Field(default_factory=list)
    search_confidence: float = Field(ge=0.0, le=1.0)
    total_results_found: int = Field(ge=0)
    search_strategy_used: str
    processing_time_seconds: float = Field(ge=0.0)


class SearchMetrics(BaseModel):
    """Metrics from search operations."""

    vector_search_time: float = Field(ge=0.0)
    graph_search_time: float = Field(ge=0.0)
    gnn_search_time: float = Field(ge=0.0)
    total_search_time: float = Field(ge=0.0)
    results_merged: int = Field(ge=0)
    duplicate_results_removed: int = Field(ge=0)


# Use centralized Azure PydanticAI provider
from agents.core.azure_pydantic_provider import get_azure_openai_model

# Create the Universal Search Agent with proper PydanticAI patterns
universal_search_agent = Agent[UniversalDeps, MultiModalSearchResult](
    get_azure_openai_model(),
    deps_type=UniversalDeps,
    output_type=MultiModalSearchResult,
    system_prompt="""You are the Universal Search Agent.

Your role is to orchestrate multi-modal search (Vector + Graph + GNN) for ANY type of content using universal patterns.

CRITICAL RULES:
- Use UNIVERSAL search patterns that work for any domain
- DO NOT assume content types or domain categories
- DELEGATE to Domain Intelligence Agent for query analysis
- ORCHESTRATE vector search, graph traversal, and GNN inference
- UNIFY results using universal ranking algorithms (not domain-specific)
- ADAPT search strategy based on discovered content characteristics

You orchestrate:
1. Vector similarity search (Azure Cognitive Search)
2. Graph relationship traversal (Azure Cosmos DB Gremlin)  
3. GNN pattern inference (Azure ML)
4. Result unification and ranking (universal algorithms)

Always use the Domain Intelligence Agent first to understand query characteristics,
then adapt your search strategy accordingly using proper agent delegation.""",
)


@universal_search_agent.tool
async def execute_multi_modal_search(
    ctx: RunContext[UniversalDeps],
    user_query: str,
    max_results: int = 10,
    use_domain_analysis: bool = True,
) -> MultiModalSearchResult:
    """
    Execute multi-modal search with proper agent delegation.

    This tool orchestrates multiple search modalities and demonstrates
    proper PydanticAI agent delegation patterns.
    """
    import time

    start_time = time.time()

    domain_analysis = None
    search_strategy = "universal_default"

    # Delegate to Domain Intelligence Agent for query analysis (proper PydanticAI pattern)
    if use_domain_analysis:
        try:
            domain_result = await domain_intelligence_agent.run(
                f"Analyze the following search query characteristics:\n\n{user_query}",
                deps=ctx.deps,  # Pass dependencies properly
                usage=ctx.usage,  # Pass usage for tracking
            )
            domain_analysis = domain_result.output
            search_strategy = f"adaptive_{domain_analysis.domain_signature}"

        except Exception as e:
            print(
                f"Warning: Domain analysis failed, using default search strategy: {e}"
            )

    # Generate search configuration using tools
    search_config = await generate_search_query(
        ctx, user_query, domain_characteristics=domain_analysis
    )

    # Execute parallel search across modalities
    vector_results = []
    graph_results = []
    gnn_results = []
    search_metrics = SearchMetrics(
        vector_search_time=0.0,
        graph_search_time=0.0,
        gnn_search_time=0.0,
        total_search_time=0.0,
        results_merged=0,
        duplicate_results_removed=0,
    )

    # 1. Vector similarity search (Azure Cognitive Search)
    if ctx.deps.is_service_available("search"):
        vector_start = time.time()
        try:
            vector_results = await _execute_vector_search(
                ctx, user_query, search_config, max_results
            )
        except Exception as e:
            print(f"Vector search failed: {e}")
        search_metrics.vector_search_time = time.time() - vector_start

    # 2. Graph relationship traversal (Azure Cosmos DB)
    if ctx.deps.is_service_available("cosmos"):
        graph_start = time.time()
        try:
            graph_results = await _execute_graph_search(
                ctx, user_query, domain_analysis, max_results
            )
        except Exception as e:
            print(f"Graph search failed: {e}")
        search_metrics.graph_search_time = time.time() - graph_start

    # 3. GNN pattern inference (Azure ML)
    if ctx.deps.is_service_available("gnn"):
        gnn_start = time.time()
        try:
            gnn_results = await _execute_gnn_search(
                ctx, user_query, vector_results, graph_results, max_results
            )
        except Exception as e:
            print(f"GNN search failed: {e}")
        search_metrics.gnn_search_time = time.time() - gnn_start

    # Unify results using universal ranking
    unified_results, metrics_update = await _unify_search_results(
        vector_results,
        graph_results,
        gnn_results,
        user_query,
        domain_analysis,
        max_results,
    )

    search_metrics.results_merged = metrics_update["merged"]
    search_metrics.duplicate_results_removed = metrics_update["duplicates_removed"]
    search_metrics.total_search_time = time.time() - start_time

    # Calculate search confidence based on result quality
    search_confidence = _calculate_search_confidence(
        vector_results, graph_results, gnn_results, unified_results
    )

    return MultiModalSearchResult(
        vector_results=vector_results,
        graph_results=graph_results,
        gnn_results=gnn_results,
        unified_results=unified_results,
        search_confidence=search_confidence,
        total_results_found=len(unified_results),
        search_strategy_used=search_strategy,
        processing_time_seconds=search_metrics.total_search_time,
    )


async def _execute_vector_search(
    ctx: RunContext[UniversalDeps],
    query: str,
    search_config: Dict[str, Any],
    max_results: int,
) -> List[SearchResult]:
    """Execute vector similarity search using Azure Cognitive Search."""
    search_client = ctx.deps.search_client

    try:
        # Execute search with generated configuration
        search_response = await search_client.search(
            search_text=search_config["search_text"],
            top=min(search_config.get("top", max_results), max_results),
            highlight_fields=search_config.get("highlight_fields", []),
            search_mode=search_config.get("search_mode", "any"),
            query_type=search_config.get("query_type", "simple"),
        )

        # Convert to universal search results
        results = []
        for result in search_response.get("value", []):
            search_result = SearchResult(
                title=result.get("title", ""),
                content=result.get("content", "")[:500],  # Truncate for display
                score=result.get("@search.score", 0.0),
                source="vector_search",
                metadata={
                    "search_highlights": result.get("@search.highlights", {}),
                    "azure_search_score": result.get("@search.score", 0.0),
                },
            )
            results.append(search_result)

        return results

    except Exception as e:
        print(f"Vector search execution failed: {e}")
        return []


async def _execute_graph_search(
    ctx: RunContext[UniversalDeps],
    query: str,
    domain_analysis: Optional[UniversalDomainAnalysis],
    max_results: int,
) -> List[Dict[str, Any]]:
    """Execute graph traversal search using Azure Cosmos DB Gremlin."""
    cosmos_client = ctx.deps.cosmos_client

    try:
        # Generate Gremlin query for relationship traversal
        entity_types = []
        relationship_types = []

        # Use discovered characteristics (not predetermined assumptions)
        if domain_analysis:
            # Extract potential entity types from analysis
            entity_indicators = getattr(domain_analysis, "entity_indicators", [])
            if entity_indicators:
                entity_types = entity_indicators

        gremlin_query = await generate_gremlin_query(
            ctx,
            f"Find entities and relationships related to: {query}",
            entity_types=entity_types,
            relationship_types=relationship_types,
        )

        # Execute graph query
        graph_response = await cosmos_client.execute_query(gremlin_query)

        # Convert to universal format
        results = []
        for item in graph_response[:max_results]:
            if isinstance(item, dict):
                results.append(
                    {
                        "entity": item.get("label", "unknown"),
                        "properties": item.get("properties", {}),
                        "relationships": item.get("relationships", []),
                        "confidence": item.get("confidence", 0.5),
                        "source": "graph_search",
                    }
                )

        return results

    except Exception as e:
        print(f"Graph search execution failed: {e}")
        return []


async def _execute_gnn_search(
    ctx: RunContext[UniversalDeps],
    query: str,
    vector_results: List[SearchResult],
    graph_results: List[Dict[str, Any]],
    max_results: int,
) -> List[Dict[str, Any]]:
    """Execute GNN pattern inference using Azure ML."""
    gnn_client = ctx.deps.gnn_client

    try:
        # Prepare input for GNN based on vector and graph results
        gnn_input = {
            "query_embedding": query,  # Would be actual embedding in real implementation
            "vector_context": [r.content for r in vector_results[:5]],
            "graph_context": [g.get("entity", "") for g in graph_results[:5]],
            "max_results": max_results,
        }

        # Execute GNN inference
        gnn_response = await gnn_client.predict(gnn_input)

        # Convert to universal format
        results = []
        for prediction in gnn_response.get("predictions", [])[:max_results]:
            results.append(
                {
                    "predicted_entity": prediction.get("entity", ""),
                    "confidence": prediction.get("confidence", 0.0),
                    "reasoning": prediction.get("reasoning", ""),
                    "source": "gnn_inference",
                }
            )

        return results

    except Exception as e:
        print(f"GNN search execution failed: {e}")
        return []


async def _unify_search_results(
    vector_results: List[SearchResult],
    graph_results: List[Dict[str, Any]],
    gnn_results: List[Dict[str, Any]],
    query: str,
    domain_analysis: Optional[UniversalDomainAnalysis],
    max_results: int,
) -> tuple[List[SearchResult], Dict[str, int]]:
    """Unify results from multiple search modalities using universal ranking."""
    unified_results = []
    seen_content = set()
    duplicates_removed = 0

    # Convert all results to SearchResult format for unification
    all_results = []

    # Add vector results
    all_results.extend(vector_results)

    # Convert graph results
    for graph_result in graph_results:
        search_result = SearchResult(
            title=graph_result.get("entity", "Graph Entity"),
            content=str(graph_result.get("properties", {}))[:200],
            score=graph_result.get("confidence", 0.5),
            source="graph_search",
            metadata=graph_result,
        )
        all_results.append(search_result)

    # Convert GNN results
    for gnn_result in gnn_results:
        search_result = SearchResult(
            title=gnn_result.get("predicted_entity", "GNN Prediction"),
            content=gnn_result.get("reasoning", "")[:200],
            score=gnn_result.get("confidence", 0.5),
            source="gnn_inference",
            metadata=gnn_result,
        )
        all_results.append(search_result)

    # Universal ranking algorithm (domain-agnostic)
    for result in all_results:
        content_hash = hash(result.content[:100])  # Simple deduplication

        if content_hash in seen_content:
            duplicates_removed += 1
            continue

        seen_content.add(content_hash)

        # Universal relevance scoring (not domain-specific)
        relevance_score = result.score

        # Boost based on source diversity
        if result.source == "vector_search":
            relevance_score *= 1.0  # Base score
        elif result.source == "graph_search":
            relevance_score *= 1.1  # Slight boost for relationship info
        elif result.source == "gnn_inference":
            relevance_score *= 1.2  # Boost for ML insights

        # Boost based on content quality indicators
        if len(result.content) > 100:  # More substantial content
            relevance_score *= 1.05

        # Update score
        result.score = relevance_score
        unified_results.append(result)

    # Sort by unified relevance score
    unified_results.sort(key=lambda x: x.score, reverse=True)

    return unified_results[:max_results], {
        "merged": len(unified_results),
        "duplicates_removed": duplicates_removed,
    }


def _calculate_search_confidence(
    vector_results: List[SearchResult],
    graph_results: List[Dict[str, Any]],
    gnn_results: List[Dict[str, Any]],
    unified_results: List[SearchResult],
) -> float:
    """Calculate overall search confidence based on result quality."""
    if not unified_results:
        return 0.0

    # Base confidence from result availability
    modalities_used = 0
    if vector_results:
        modalities_used += 1
    if graph_results:
        modalities_used += 1
    if gnn_results:
        modalities_used += 1

    base_confidence = modalities_used / 3.0  # Up to 3 modalities

    # Boost based on result quality
    avg_score = sum(r.score for r in unified_results) / len(unified_results)
    quality_boost = min(avg_score, 1.0)

    # Final confidence
    overall_confidence = (base_confidence + quality_boost) / 2.0

    return min(overall_confidence, 1.0)


@universal_search_agent.tool
async def validate_search_requirements(
    ctx: RunContext[UniversalDeps],
) -> Dict[str, Any]:
    """
    Validate that required services are available for universal search.
    """
    required_services = ["openai"]  # Required for search orchestration
    optional_services = ["search", "cosmos", "gnn", "monitoring"]

    validation_result = {
        "required_services_available": all(
            ctx.deps.is_service_available(service) for service in required_services
        ),
        "search_modalities_available": {
            "vector_search": ctx.deps.is_service_available("search"),
            "graph_search": ctx.deps.is_service_available("cosmos"),
            "gnn_inference": ctx.deps.is_service_available("gnn"),
        },
        "total_modalities": sum(
            1
            for service in ["search", "cosmos", "gnn"]
            if ctx.deps.is_service_available(service)
        ),
        "can_perform_basic_search": ctx.deps.is_service_available("search"),
        "can_perform_multi_modal": sum(
            1
            for service in ["search", "cosmos", "gnn"]
            if ctx.deps.is_service_available(service)
        )
        >= 2,
        "available_services": ctx.deps.get_available_services(),
    }

    return validation_result


# Factory function for proper agent initialization
async def create_universal_search_agent() -> (
    Agent[UniversalDeps, MultiModalSearchResult]
):
    """
    Create Universal Search Agent with initialized dependencies.

    Follows PydanticAI best practices for agent creation.
    """
    deps = await get_universal_deps()

    # Validate required services
    if not deps.is_service_available("openai"):
        raise RuntimeError("Universal Search Agent requires Azure OpenAI service")

    # Check for at least one search modality
    search_modalities = sum(
        1
        for service in ["search", "cosmos", "gnn"]
        if deps.is_service_available(service)
    )

    if search_modalities == 0:
        print(
            "Warning: No search modalities available. Search functionality will be limited."
        )

    return universal_search_agent


# Main execution function for testing
async def run_universal_search(
    query: str, max_results: int = 10, use_domain_analysis: bool = True
) -> MultiModalSearchResult:
    """
    Run universal search with proper PydanticAI patterns.
    """
    deps = await get_universal_deps()
    agent = await create_universal_search_agent()

    result = await agent.run(
        f"Execute multi-modal search for the following query:\n\nQuery: {query}\nMax Results: {max_results}",
        deps=deps,
    )

    return result.output
