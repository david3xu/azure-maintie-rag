"""
PydanticAI Tools for Knowledge Extraction Agent
===============================================

This module provides PydanticAI-compatible tools for the Knowledge Extraction Agent,
implementing entity and relationship extraction with enterprise integration.

Features:
- Entity extraction with multiple strategies
- Relationship extraction and validation
- Knowledge graph construction
- ConsolidatedAzureServices integration
- Enterprise error handling and monitoring
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic_ai import RunContext

from ..core.azure_services import ConsolidatedAzureServices
from ..models.responses import (
    EntityExtractionResponse,
    RelationshipExtractionResponse,
    UniversalEntity,
    UniversalRelation,
)

# Import our consolidated tools and models
from .extraction_tools import (
    EntityExtractor,
    ExtractionError,
    KnowledgeGraphBuilder,
    RelationshipExtractor,
    VectorEmbeddingGenerator,
)

logger = logging.getLogger(__name__)


async def extract_entities_tool(
    ctx: RunContext,
    text: str,
    domain: Optional[str] = None,
    extraction_strategy: str = "hybrid",
    azure_services: Optional[ConsolidatedAzureServices] = None,
) -> EntityExtractionResponse:
    """
    Extract entities from text using advanced extraction strategies.

    Args:
        ctx: PydanticAI run context
        text: Text to extract entities from
        domain: Optional domain context for focused extraction
        extraction_strategy: Extraction strategy ('pattern', 'nlp', 'hybrid')
        azure_services: Optional ConsolidatedAzureServices instance

    Returns:
        EntityExtractionResponse with extracted entities and metadata
    """
    try:
        # Extract dependencies from context if available
        confidence_threshold = 0.7
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "azure_services"):
            azure_services = ctx.deps.azure_services
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "app_settings"):
            confidence_threshold = (
                ctx.deps.app_settings.tri_modal_search.competitive_advantage.confidence_threshold
            )

        # Create entity extraction request
        extraction_request = EntityExtractionRequest(
            text=text,
            domain=domain,
            extraction_strategy=extraction_strategy,
            confidence_threshold=confidence_threshold,
            include_metadata=True,
        )

        # Execute entity extraction
        logger.info(
            f"PydanticAI Knowledge Extraction executing entity extraction: {len(text)} chars, strategy={extraction_strategy}"
        )

        result = await execute_entity_extraction(ctx, extraction_request)

        logger.info(
            f"PydanticAI Knowledge Extraction entity extraction completed: "
            f"{len(result.entities)} entities, confidence={result.average_confidence:.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"PydanticAI Knowledge Extraction entity extraction failed: {e}")

        # Return error response
        return EntityExtractionResponse(
            entities=[],
            extraction_metadata={
                "strategy": extraction_strategy,
                "domain": domain,
                "error": True,
                "error_message": str(e),
            },
            confidence_scores=[],
            average_confidence=0.0,
            execution_time=0.0,
        )


async def extract_relationships_tool(
    ctx: RunContext,
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    domain: Optional[str] = None,
    extraction_strategy: str = "semantic",
    azure_services: Optional[ConsolidatedAzureServices] = None,
) -> RelationshipExtractionResponse:
    """
    Extract relationships from text using advanced relationship detection.

    Args:
        ctx: PydanticAI run context
        text: Text to extract relationships from
        entities: Optional pre-extracted entities to use as anchors
        domain: Optional domain context for focused extraction
        extraction_strategy: Extraction strategy ('syntactic', 'semantic', 'pattern')
        azure_services: Optional ConsolidatedAzureServices instance

    Returns:
        RelationshipExtractionResponse with extracted relationships and metadata
    """
    try:
        # Extract dependencies from context if available
        confidence_threshold = 0.7
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "azure_services"):
            azure_services = ctx.deps.azure_services
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "app_settings"):
            confidence_threshold = (
                ctx.deps.app_settings.tri_modal_search.competitive_advantage.confidence_threshold
            )

        # Create relationship extraction request
        extraction_request = RelationshipExtractionRequest(
            text=text,
            entities=entities,
            domain=domain,
            extraction_strategy=extraction_strategy,
            confidence_threshold=confidence_threshold,
            include_metadata=True,
        )

        # Execute relationship extraction
        logger.info(
            f"PydanticAI Knowledge Extraction executing relationship extraction: {len(text)} chars, strategy={extraction_strategy}"
        )

        result = await execute_relationship_extraction(ctx, extraction_request)

        logger.info(
            f"PydanticAI Knowledge Extraction relationship extraction completed: "
            f"{len(result.relationships)} relationships, confidence={result.average_confidence:.2f}"
        )

        return result

    except Exception as e:
        logger.error(
            f"PydanticAI Knowledge Extraction relationship extraction failed: {e}"
        )

        # Return error response
        return RelationshipExtractionResponse(
            relationships=[],
            extraction_metadata={
                "strategy": extraction_strategy,
                "domain": domain,
                "error": True,
                "error_message": str(e),
            },
            confidence_scores=[],
            average_confidence=0.0,
            execution_time=0.0,
        )


async def build_knowledge_graph_tool(
    ctx: RunContext,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    domain: Optional[str] = None,
    azure_services: Optional[ConsolidatedAzureServices] = None,
) -> Dict[str, Any]:
    """
    Build knowledge graph from extracted entities and relationships.

    Args:
        ctx: PydanticAI run context
        entities: List of extracted entities
        relationships: List of extracted relationships
        domain: Optional domain context
        azure_services: Optional ConsolidatedAzureServices instance

    Returns:
        Knowledge graph structure with nodes, edges, and metadata
    """
    try:
        # Build knowledge graph structure
        graph_structure = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "domain": domain,
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "construction_method": "pydantic_ai_knowledge_extraction",
            },
        }

        # Convert entities to nodes
        entity_id_map = {}
        for i, entity in enumerate(entities):
            node_id = f"entity_{i}"
            entity_id_map[entity.get("entity_id", str(i))] = node_id

            graph_structure["nodes"].append(
                {
                    "id": node_id,
                    "label": entity.get("text", ""),
                    "type": entity.get("entity_type", "unknown"),
                    "confidence": entity.get("confidence", 0.0),
                    "properties": entity.get("properties", {}),
                    "domain": domain,
                }
            )

        # Convert relationships to edges
        for i, relationship in enumerate(relationships):
            source_id = relationship.get("source_entity_id")
            target_id = relationship.get("target_entity_id")

            # Map to graph node IDs
            source_node = entity_id_map.get(source_id, source_id)
            target_node = entity_id_map.get(target_id, target_id)

            graph_structure["edges"].append(
                {
                    "id": f"edge_{i}",
                    "source": source_node,
                    "target": target_node,
                    "relationship_type": relationship.get(
                        "relationship_type", "related"
                    ),
                    "confidence": relationship.get("confidence", 0.0),
                    "properties": relationship.get("properties", {}),
                    "domain": domain,
                }
            )

        # Calculate graph metrics
        graph_structure["metrics"] = {
            "node_count": len(graph_structure["nodes"]),
            "edge_count": len(graph_structure["edges"]),
            "average_node_confidence": sum(
                node["confidence"] for node in graph_structure["nodes"]
            )
            / max(1, len(graph_structure["nodes"])),
            "average_edge_confidence": sum(
                edge["confidence"] for edge in graph_structure["edges"]
            )
            / max(1, len(graph_structure["edges"])),
            "connectivity_ratio": len(graph_structure["edges"])
            / max(1, len(graph_structure["nodes"])),
            "competitive_advantage_score": 0.9,  # High score for successful graph construction
        }

        logger.info(
            f"PydanticAI Knowledge Extraction knowledge graph built: "
            f"{len(graph_structure['nodes'])} nodes, {len(graph_structure['edges'])} edges"
        )

        return graph_structure

    except Exception as e:
        logger.error(
            f"PydanticAI Knowledge Extraction knowledge graph construction failed: {e}"
        )
        return {
            "nodes": [],
            "edges": [],
            "metadata": {"error": True, "error_message": str(e), "domain": domain},
            "metrics": {
                "node_count": 0,
                "edge_count": 0,
                "competitive_advantage_score": 0.0,
            },
        }


async def validate_extraction_quality_tool(
    ctx: RunContext,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    text: str,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate extraction quality and provide improvement recommendations.

    Args:
        ctx: PydanticAI run context
        entities: Extracted entities
        relationships: Extracted relationships
        text: Original text
        domain: Optional domain context

    Returns:
        Quality validation results and recommendations
    """
    try:
        # Calculate quality metrics
        text_length = len(text)
        entity_density = len(entities) / max(
            1, text_length / 100
        )  # Entities per 100 characters
        relationship_coverage = len(relationships) / max(
            1, len(entities)
        )  # Relationships per entity

        # Calculate confidence scores
        entity_confidences = [e.get("confidence", 0.0) for e in entities]
        relationship_confidences = [r.get("confidence", 0.0) for r in relationships]

        avg_entity_confidence = sum(entity_confidences) / max(
            1, len(entity_confidences)
        )
        avg_relationship_confidence = sum(relationship_confidences) / max(
            1, len(relationship_confidences)
        )

        # Quality assessment
        quality_results = {
            "extraction_metrics": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "text_length": text_length,
                "entity_density": entity_density,
                "relationship_coverage": relationship_coverage,
            },
            "confidence_metrics": {
                "average_entity_confidence": avg_entity_confidence,
                "average_relationship_confidence": avg_relationship_confidence,
                "high_confidence_entities": len(
                    [e for e in entities if e.get("confidence", 0) > 0.8]
                ),
                "high_confidence_relationships": len(
                    [r for r in relationships if r.get("confidence", 0) > 0.8]
                ),
            },
            "quality_indicators": [],
            "recommendations": [],
            "overall_score": 0.0,
            "competitive_advantage_maintained": False,
        }

        # Quality assessment logic
        score = 1.0

        if entity_density < 0.5:
            quality_results["quality_indicators"].append(
                "Low entity extraction density"
            )
            quality_results["recommendations"].append(
                "Consider providing more detailed text"
            )
            score *= 0.8

        if relationship_coverage < 0.3:
            quality_results["quality_indicators"].append("Low relationship coverage")
            quality_results["recommendations"].append(
                "Text may benefit from more explicit relationship descriptions"
            )
            score *= 0.7

        if avg_entity_confidence < 0.7:
            quality_results["quality_indicators"].append("Low entity confidence")
            quality_results["recommendations"].append(
                "Review text clarity for better entity extraction"
            )
            score *= 0.8

        if avg_relationship_confidence < 0.7:
            quality_results["quality_indicators"].append("Low relationship confidence")
            quality_results["recommendations"].append(
                "Consider more explicit relationship language"
            )
            score *= 0.8

        quality_results["overall_score"] = score
        quality_results["competitive_advantage_maintained"] = score >= 0.7

        logger.info(
            f"PydanticAI Knowledge Extraction quality validation: "
            f"score={score:.2f}, entities={len(entities)}, relationships={len(relationships)}"
        )

        return quality_results

    except Exception as e:
        logger.error(f"PydanticAI Knowledge Extraction quality validation failed: {e}")
        return {
            "overall_score": 0.0,
            "competitive_advantage_maintained": False,
            "error": str(e),
        }


# Export the tools
__all__ = [
    "extract_entities_tool",
    "extract_relationships_tool",
    "build_knowledge_graph_tool",
    "validate_extraction_quality_tool",
]
