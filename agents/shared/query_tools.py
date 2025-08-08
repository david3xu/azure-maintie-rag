"""
Query Generation Utilities for Universal RAG System
===================================================

Utility functions for query generation that can be called by agent tools.
These replace the old pseudo-agents with proper utility functions.
"""

from typing import Any, Dict, List, Optional

from pydantic_ai.tools import RunContext

from agents.core.universal_deps import UniversalDeps
from agents.core.universal_models import UniversalDomainAnalysis


async def generate_gremlin_query(
    ctx: RunContext[UniversalDeps],
    query_intent: str,
    entity_types: Optional[List[str]] = None,
    relationship_types: Optional[List[str]] = None,
) -> str:
    """
    Generate Gremlin graph query from intent and discovered characteristics.

    This is a proper tool (not a pseudo-agent) that performs atomic query generation.
    """
    # Use universal configuration instead of hardcoded patterns
    config = ctx.deps.config_manager.get_extraction_config()

    # Build query based on discovered characteristics, not predetermined assumptions
    base_query = "g.V()"

    # Add filters based on discovered entity types (domain-agnostic)
    if entity_types:
        entity_filter = " or ".join(
            [f"has('type', '{etype}')" for etype in entity_types]
        )
        base_query += f".where({entity_filter})"

    # Add relationship traversal based on discovered patterns
    if relationship_types:
        for rel_type in relationship_types:
            base_query += f".out('{rel_type}')"

    # Apply universal search patterns
    if "similarity" in query_intent.lower():
        base_query += f".has('confidence', gt({config.confidence_threshold}))"

    if "limit" not in base_query:
        base_query += f".limit({config.max_results})"

    return base_query


async def generate_search_query(
    ctx: RunContext[UniversalDeps],
    user_query: str,
    domain_characteristics: Optional[UniversalDomainAnalysis] = None,
) -> Dict[str, Any]:
    """
    Generate Azure Cognitive Search query with universal patterns.

    Adapts query based on discovered content characteristics, not domain assumptions.
    """
    config = ctx.deps.config_manager.get_search_config()

    # Base search configuration
    search_config = {
        "search_text": user_query,
        "top": config.max_results,
        "highlight_fields": ["content", "title"],
        "search_mode": "any",
        "query_type": (
            "semantic" if ctx.deps.is_service_available("search") else "simple"
        ),
    }

    # Adapt based on discovered characteristics (not hardcoded domain types)
    if domain_characteristics:
        # Adjust search parameters based on measured content properties
        if domain_characteristics.vocabulary_complexity > 0.7:
            search_config["search_mode"] = "all"  # More precise for complex content

        if domain_characteristics.concept_density > 0.8:
            search_config["top"] = min(
                config.max_results * 2, 50
            )  # More results for concept-rich content

        # Use discovered patterns for field weighting
        search_fields = []
        if "code_patterns" in domain_characteristics.discovered_patterns:
            search_fields.append("code_content^2.0")
        if "structured_content" in domain_characteristics.discovered_patterns:
            search_fields.append("structured_fields^1.5")

        if search_fields:
            search_config["search_fields"] = search_fields

    return search_config


async def generate_analysis_query(
    ctx: RunContext[UniversalDeps], content: str, analysis_type: str = "characteristics"
) -> Dict[str, Any]:
    """
    Generate analysis configuration for content characteristic discovery.

    This replaces the analysis_query_agent pseudo-agent with proper tool functionality.
    """
    config = await ctx.deps.config_manager.get_universal_config(
        "/workspace/azure-maintie-rag/data/raw"
    )

    analysis_config = {
        "content": content,
        "analysis_type": analysis_type,
        "min_confidence": config.get("entity_confidence_threshold", 0.8),
        "max_entities": config.get("max_entities_per_chunk", 15),
        "max_relationships": config.get("min_relationship_strength", 0.7),
    }

    # Universal analysis patterns (no hardcoded domain assumptions)
    if analysis_type == "characteristics":
        analysis_config.update(
            {
                "analyze_vocabulary": True,
                "analyze_structure": True,
                "analyze_patterns": True,
                "discover_entity_types": True,  # Discover, don't assume
                "measure_complexity": True,
            }
        )

    elif analysis_type == "extraction":
        analysis_config.update(
            {
                "extract_entities": True,
                "extract_relationships": True,
                "validate_quality": True,
                "adaptive_thresholds": True,  # Adapt based on content
            }
        )

    elif analysis_type == "search":
        analysis_config.update(
            {
                "identify_key_concepts": True,
                "measure_relevance": True,
                "adaptive_ranking": True,
            }
        )

    return analysis_config


async def orchestrate_query_workflow(
    ctx: RunContext[UniversalDeps],
    user_query: str,
    workflow_type: str = "universal_search",
) -> Dict[str, Any]:
    """
    Orchestrate multi-step query workflow with proper agent delegation.

    This replaces the universal_query_orchestrator pseudo-agent.
    """
    workflow_config = {
        "user_query": user_query,
        "workflow_type": workflow_type,
        "steps": [],
        "agent_sequence": [],
    }

    # Define workflow based on query characteristics (discovered, not assumed)
    if workflow_type == "universal_search":
        workflow_config["steps"] = [
            "analyze_query_characteristics",  # Domain Intelligence
            "extract_key_concepts",  # Knowledge Extraction
            "execute_multi_modal_search",  # Universal Search
        ]
        workflow_config["agent_sequence"] = [
            "domain_intelligence",
            "knowledge_extraction",
            "universal_search",
        ]

    elif workflow_type == "knowledge_discovery":
        workflow_config["steps"] = [
            "discover_content_patterns",  # Domain Intelligence
            "extract_entities_relationships",  # Knowledge Extraction
            "build_knowledge_connections",  # Graph construction
        ]
        workflow_config["agent_sequence"] = [
            "domain_intelligence",
            "knowledge_extraction",
        ]

    # Add universal processing parameters
    config = ctx.deps.config_manager.get_workflow_config()
    workflow_config.update(
        {
            "max_iterations": config.max_workflow_steps,
            "timeout_seconds": config.workflow_timeout,
            "enable_monitoring": ctx.deps.is_service_available("monitoring"),
            "enable_caching": True,
        }
    )

    return workflow_config


# Utility function registry for easy access
QUERY_UTILITIES = [
    generate_gremlin_query,
    generate_search_query,
    generate_analysis_query,
    orchestrate_query_workflow,
]
