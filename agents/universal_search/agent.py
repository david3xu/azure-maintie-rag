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


from agents.core.agent_toolsets import get_universal_search_toolset

# Use centralized Azure PydanticAI provider and toolsets
from agents.core.azure_pydantic_provider import get_azure_openai_model

# Create the Universal Search Agent with proper PydanticAI patterns using toolsets
universal_search_agent = Agent[UniversalDeps, MultiModalSearchResult](
    get_azure_openai_model(),
    output_type=MultiModalSearchResult,
    toolsets=[get_universal_search_toolset()],
    retries=3,  # Add retry configuration for Azure OpenAI reliability
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


async def _execute_multi_modal_search_internal(
    ctx: RunContext[UniversalDeps],
    user_query: str,
    max_results: int = 10,
    use_domain_analysis: bool = True,
) -> MultiModalSearchResult:
    """
    Internal multi-modal search implementation (not a tool to avoid conflicts).

    Uses centralized tools from agent_toolsets for actual search operations.
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
            # FAIL FAST - Domain analysis is required
            raise RuntimeError(
                f"Domain analysis failed: {e}. Cannot proceed without domain characteristics."
            ) from e

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

    # MANDATORY TRI-MODAL SEARCH: Vector + Graph + GNN ALL REQUIRED
    print("🎯 Executing MANDATORY tri-modal search (Vector + Graph + GNN)...")

    # 1. Vector similarity search (REQUIRED)
    print("   🔍 Vector search (MANDATORY)...")
    vector_start = time.time()
    vector_results = await _execute_vector_search(
        ctx, user_query, search_config, max_results
    )
    search_metrics.vector_search_time = time.time() - vector_start
    print(f"   ✅ Vector search completed: {len(vector_results)} results")

    # 2. Graph relationship traversal (REQUIRED)
    print("   🕸️  Graph search (MANDATORY)...")
    graph_start = time.time()
    graph_results = await _execute_graph_search(
        ctx, user_query, domain_analysis, max_results
    )
    search_metrics.graph_search_time = time.time() - graph_start
    print(f"   ✅ Graph search completed: {len(graph_results)} results")

    # 3. GNN pattern inference (REQUIRED)
    print("   🧠 GNN search (MANDATORY)...")
    gnn_start = time.time()
    gnn_results = await _execute_gnn_search(
        ctx, user_query, vector_results, graph_results, max_results
    )
    search_metrics.gnn_search_time = time.time() - gnn_start
    print(f"   ✅ GNN search completed: {len(gnn_results)} results")

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
        search_response = await search_client.search_documents(
            query=search_config["search_text"],
            top=min(search_config.get("top", max_results), max_results),
            filters=None,  # Add filters parameter
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
        # FAIL FAST - Don't return empty results on failure
        raise RuntimeError(
            f"Vector search execution failed: {e}. Check Azure Cognitive Search service."
        ) from e


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

        # Execute graph query using available public method
        # Extract key terms from query as potential entities
        query_terms = [term.strip() for term in query.split() if len(term.strip()) > 3]
        graph_response = []

        # Search for entities related to query terms
        for term in query_terms[:3]:  # Limit to first 3 terms
            try:
                related_entities = await cosmos_client.find_related_entities(
                    entity_text=term,
                    domain="universal",  # Use universal domain
                    limit=max_results // len(query_terms[:3]),
                )
                graph_response.extend(related_entities)
            except Exception as e:
                # FAIL FAST - Don't continue with partial failures
                raise RuntimeError(
                    f"Failed to find entities for term '{term}': {e}. Check Azure Cosmos DB Gremlin query execution."
                ) from e

        # Convert to universal format
        results = []
        for item in graph_response[:max_results]:
            if isinstance(item, dict):
                results.append(
                    {
                        "entity": item.get(
                            "target_entity", item.get("source_entity", "unknown")
                        ),
                        "properties": {
                            "relation_type": item.get("relation_type", "related"),
                            "source_entity": item.get("source_entity", ""),
                            "target_entity": item.get("target_entity", ""),
                        },
                        "relationships": [item.get("relation_type", "related")],
                        "confidence": 0.7,  # Default confidence for graph results
                        "source": "graph_search",
                    }
                )

        return results

    except Exception as e:
        # FAIL FAST - Don't return empty results on failure
        raise RuntimeError(
            f"Graph search execution failed: {e}. Check Azure Cosmos DB Gremlin connection."
        ) from e


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
        # FAIL FAST - Don't return empty results on failure
        raise RuntimeError(
            f"GNN search execution failed: {e}. Check Azure ML GNN inference endpoint."
        ) from e


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

        # FAIL FAST - Require domain analysis for search weights
        if not domain_analysis or not domain_analysis.processing_config:
            raise RuntimeError(
                "Domain analysis required for search weight calculation. Cannot proceed without proper configuration."
            )

        # Use Agent 1's centralized search weights (normalized 0.0-1.0)
        vector_weight = max(
            0.1, domain_analysis.processing_config.vector_search_weight * 2.0
        )  # Scale to 0.2-2.0 range for meaningful impact
        graph_weight = max(
            0.1, domain_analysis.processing_config.graph_search_weight * 2.0
        )  # Scale to 0.2-2.0 range for meaningful impact

        # Calculate GNN weight as complement (when vector + graph weights are lower, GNN gets higher weight)
        gnn_weight = max(
            0.5, 2.0 - (vector_weight + graph_weight) / 2.0
        )  # Adaptive GNN weight based on other modalities

        # Apply Agent 1's centralized search weights (from domain analysis)
        if result.source == "vector_search":
            relevance_score *= (
                vector_weight  # Agent 1 centralized vector weight (scaled)
            )
        elif result.source == "graph_search":
            relevance_score *= graph_weight  # Agent 1 centralized graph weight (scaled)
        elif result.source == "gnn_inference":
            relevance_score *= (
                gnn_weight  # Adaptive GNN weight (computed from Agent 1 weights)
            )

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


# Tool already defined in agent_toolsets.py - removed duplicate to fix conflict


# Factory function for proper agent initialization
async def create_universal_search_agent() -> (
    Agent[UniversalDeps, MultiModalSearchResult]
):
    """
    Create Universal Search Agent with initialized dependencies.

    Follows PydanticAI best practices for agent creation.
    """
    deps = await get_universal_deps()

    # Validate ALL REQUIRED services for mandatory tri-modal search
    print("🚨 Validating MANDATORY tri-modal search requirements...")

    required_services = {
        "openai": "Azure OpenAI service",
        "search": "Azure Cognitive Search (Vector modality)",
        "cosmos": "Azure Cosmos DB Gremlin (Graph modality)",
        "gnn": "Azure ML GNN inference (GNN modality)",
    }

    missing_services = []
    for service_key, service_name in required_services.items():
        if not deps.is_service_available(service_key):
            missing_services.append(service_name)

    if missing_services:
        raise RuntimeError(
            f"MANDATORY tri-modal search requirements not met. Missing services: {missing_services}. "
            f"All three modalities (Vector + Graph + GNN) require their respective Azure services."
        )

    return universal_search_agent


# Alias for backward compatibility
UniversalSearchAgent = universal_search_agent


# Main execution function for testing
async def run_universal_search(
    query: str, max_results: int = 10, use_domain_analysis: bool = True
) -> MultiModalSearchResult:
    """
    Run universal search using centralized toolsets directly.
    """
    deps = await get_universal_deps()

    # Use centralized orchestration tool directly
    from agents.core.agent_toolsets import orchestrate_universal_search

    class SearchRunContext:
        def __init__(self, deps):
            self.deps = deps
            self.usage = None

    ctx = SearchRunContext(deps)

    # Call centralized orchestration
    search_results = await orchestrate_universal_search(
        ctx, query, max_results, use_domain_analysis
    )

    # Convert to MultiModalSearchResult format
    unified_results = []
    for result in search_results["unified_results"]:
        search_result = SearchResult(
            title=result["title"],
            content=result["content"],
            score=result["score"],
            source=result["source"],
            metadata=result.get("metadata", {}),
        )
        unified_results.append(search_result)

    return MultiModalSearchResult(
        vector_results=search_results.get(
            "vector_results", []
        ),  # Include vector results
        graph_results=search_results["graph_results"],
        gnn_results=search_results.get(
            "gnn_results", []
        ),  # NOW IMPLEMENTED - Include actual GNN results
        unified_results=unified_results,
        search_confidence=search_results["search_confidence"],
        total_results_found=search_results["total_results_found"],
        search_strategy_used=search_results["search_strategy_used"],
        processing_time_seconds=search_results["processing_time_seconds"],
    )


# Main execution for testing
async def main():
    """Test Universal Search Agent with real Azure services."""
    print("🔍 Testing Universal Search Agent with Azure services")

    test_queries = [
        "Azure AI services",
        "language processing",
        "machine learning models",
        "document analysis",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print("=" * 60)

        try:
            result = await run_universal_search(query, max_results=5)

            print(f"✅ Search Results:")
            print(f"   - Total results: {result.total_results_found}")
            print(f"   - Search confidence: {result.search_confidence:.3f}")
            print(f"   - Processing time: {result.processing_time_seconds:.3f}s")
            print(f"   - Strategy: {result.search_strategy_used}")

            if result.unified_results:
                print(f"   - Top result: {result.unified_results[0].title}")
                print(f"     Score: {result.unified_results[0].score:.3f}")
                print(f"     Source: {result.unified_results[0].source}")
            else:
                print("   ❌ No results returned - investigating...")

            # Break after first query for detailed analysis
            break

        except Exception as e:
            # FAIL FAST - Don't mask search failures
            raise RuntimeError(
                f"Universal search orchestration failed: {e}. Check all Azure services are operational."
            ) from e
            break


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
