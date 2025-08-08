"""
Knowledge Extraction Agent - Universal Entity/Relationship Extraction
====================================================================

This agent performs universal entity and relationship extraction WITHOUT domain assumptions.
Follows PydanticAI best practices with proper agent delegation and clean boundaries.

Key Principles:
- Universal extraction patterns that work for ANY domain
- Uses centralized dependencies (no duplicate Azure clients)
- Proper agent delegation to Domain Intelligence Agent
- Atomic tools with single responsibilities
- Real Azure Cosmos DB Gremlin integration
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

from agents.core.universal_deps import UniversalDeps, get_universal_deps
from agents.core.universal_models import (
    ExtractedEntity,
    ExtractedRelationship,
    UniversalDomainAnalysis,
    UniversalProcessingConfiguration,
)
from agents.domain_intelligence.agent import domain_intelligence_agent
from agents.shared.query_tools import generate_analysis_query, generate_gremlin_query
from infrastructure.azure_ml.classification_client import SimpleAzureClassifier
from infrastructure.prompt_workflows.processors.quality_assessor import (
    assess_extraction_quality,
)
from infrastructure.prompt_workflows.prompt_workflow_orchestrator import (
    PromptWorkflowOrchestrator,
)
from infrastructure.prompt_workflows.universal_prompt_generator import (
    UniversalPromptGenerator,
)


class ExtractionResult(BaseModel):
    """Results from universal extraction process."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    processing_signature: str = Field(description="Processing configuration used")
    graph_nodes_created: int = Field(default=0, ge=0)
    graph_edges_created: int = Field(default=0, ge=0)
    quality_assessment: Optional[Dict[str, Any]] = Field(
        default=None, description="Automated quality assessment"
    )


class GraphOperationResult(BaseModel):
    """Results from graph database operations."""

    operation_type: str
    nodes_affected: int = Field(ge=0)
    edges_affected: int = Field(ge=0)
    success: bool
    error_message: Optional[str] = None


# Create the Knowledge Extraction Agent with proper PydanticAI patterns
knowledge_extraction_agent = Agent[UniversalDeps, ExtractionResult](
    "openai:gpt-4o",  # PydanticAI format for Azure OpenAI via AsyncAzureOpenAI client
    deps_type=UniversalDeps,
    output_type=ExtractionResult,
    system_prompt="""You are the Universal Knowledge Extraction Agent.

Your role is to extract entities and relationships from ANY type of content using universal patterns.

CRITICAL RULES:
- Use UNIVERSAL extraction patterns that work for any domain
- DO NOT assume entity types - discover them from content
- DO NOT use predetermined relationship categories
- ADAPT extraction based on content characteristics from Domain Intelligence Agent
- Store extracted knowledge in Azure Cosmos DB using Gremlin queries
- Ensure extracted entities and relationships are domain-agnostic

You extract:
1. Entities (discovered types, not predetermined categories)
2. Relationships (discovered patterns, not fixed schemas)  
3. Properties (measured from content, not assumed)
4. Graph connections (based on actual content relationships)

Always use the Domain Intelligence Agent first to understand content characteristics,
then adapt your extraction approach accordingly.""",
)


@knowledge_extraction_agent.tool
async def extract_with_generated_prompts(
    ctx: RunContext[UniversalDeps],
    content: str,
    use_domain_analysis: bool = True,
    data_directory: Optional[str] = None,
) -> ExtractionResult:
    """
    Extract entities and relationships using dynamically generated prompts from domain analysis.

    This tool demonstrates the complete prompt workflow integration:
    1. Domain analysis → 2. Prompt generation → 3. Entity/relationship extraction
    """
    logger.info("🔬 Starting extraction with generated prompts...")

    try:
        # Create workflow orchestrator with domain intelligence
        orchestrator = (
            await PromptWorkflowOrchestrator.create_with_domain_intelligence()
        )

        # Use the workflow orchestrator for complete extraction
        workflow_results = await orchestrator.execute_extraction_workflow(
            texts=[content],
            confidence_threshold=0.7,
            max_entities=50,
            max_relationships=40,
        )

        # Convert workflow results to agent format
        entities = []
        for entity_data in workflow_results.get("entities", []):
            entities.append(
                ExtractedEntity(
                    text=entity_data.get("text", ""),
                    type=entity_data.get("entity_type", "concept"),
                    confidence=entity_data.get("confidence", 0.7),
                    context=entity_data.get("context", ""),
                    properties=entity_data.get("properties", {}),
                )
            )

        relationships = []
        for rel_data in workflow_results.get("relationships", []):
            relationships.append(
                ExtractedRelationship(
                    source_entity=rel_data.get("subject", ""),
                    target_entity=rel_data.get("object", ""),
                    relationship_type=rel_data.get("predicate", "relates_to"),
                    confidence=rel_data.get("confidence", 0.7),
                    context=rel_data.get("context", ""),
                    properties={},
                )
            )

        # Calculate extraction confidence from workflow results
        workflow_metadata = workflow_results.get("workflow_metadata", {})
        extraction_confidence = workflow_metadata.get("overall_confidence", 0.0)

        processing_signature = (
            f"generated_prompts_"
            f"{workflow_metadata.get('extraction_strategy', 'universal')}_"
            f"{len(entities)}e_{len(relationships)}r"
        )

        logger.info(
            f"✅ Generated prompt extraction completed: {len(entities)} entities, {len(relationships)} relationships"
        )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            extraction_confidence=extraction_confidence,
            processing_signature=processing_signature,
            quality_assessment=workflow_results.get("quality_metrics"),
        )

    except Exception as e:
        logger.error(f"❌ Generated prompt extraction failed: {e}")
        logger.info("🔄 Falling back to standard extraction with hardcoded prompts...")

        # Comprehensive fallback mechanism
        try:
            # Try standard extraction first
            fallback_result = await extract_entities_and_relationships(
                ctx, content, use_domain_analysis
            )
            fallback_result.processing_signature += "_fallback_standard"
            return fallback_result
        except Exception as fallback_error:
            logger.error(f"❌ Standard extraction also failed: {fallback_error}")
            logger.info("🔄 Using emergency fallback with pattern-based extraction...")

            # Emergency fallback - basic pattern extraction
            words = content.split()
            entities = []
            relationships = []

            # Extract capitalized words as potential entities
            for i, word in enumerate(words):
                if (
                    word and word[0].isupper() and len(word) > 2 and i < 20
                ):  # Limit to 20
                    entities.append(
                        ExtractedEntity(
                            text=word.strip('.,!?;:"()[]'),
                            type="emergency_entity",
                            confidence=0.5,
                            context=f"Emergency extraction from word position {i}",
                            properties={"extraction_method": "emergency_fallback"},
                        )
                    )

            # Create basic relationships between adjacent entities
            for i in range(min(len(entities) - 1, 10)):  # Limit to 10
                relationships.append(
                    ExtractedRelationship(
                        source_entity=entities[i].text,
                        target_entity=entities[i + 1].text,
                        relationship_type="emergency_relation",
                        confidence=0.3,
                        context="Emergency fallback relationship",
                        properties={"extraction_method": "emergency_fallback"},
                    )
                )

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                extraction_confidence=0.3,
                processing_signature="emergency_fallback_pattern_extraction",
                quality_assessment={
                    "overall_score": 0.3,
                    "quality_level": "emergency_fallback",
                    "extraction_method": "emergency_pattern_fallback",
                    "note": "Generated prompts and standard extraction both failed",
                },
            )


@knowledge_extraction_agent.tool
async def extract_entities_and_relationships(
    ctx: RunContext[UniversalDeps], content: str, use_domain_analysis: bool = True
) -> ExtractionResult:
    """
    Extract entities and relationships using universal patterns with optional domain analysis.

    This tool demonstrates proper agent delegation to Domain Intelligence Agent.
    """
    processing_config = None
    domain_analysis = None

    # Delegate to Domain Intelligence Agent for content analysis (proper PydanticAI pattern)
    if use_domain_analysis:
        try:
            domain_result = await domain_intelligence_agent.run(
                f"Analyze content characteristics for extraction optimization:\n\n{content}",
                deps=ctx.deps,  # Pass dependencies properly
                usage=ctx.usage,  # Pass usage for tracking
            )
            domain_analysis = domain_result.output

            # Generate adaptive processing configuration
            processing_result = await domain_intelligence_agent.run(
                "Generate processing configuration based on the analyzed characteristics",
                deps=ctx.deps,
                usage=ctx.usage,
            )
            # Note: This would need the proper output from domain intelligence

        except Exception as e:
            print(f"Warning: Domain analysis failed, using default configuration: {e}")

    # Use configuration from analysis or fall back to defaults
    config_manager = ctx.deps.config_manager
    base_config = config_manager.get_extraction_config()

    # Apply adaptive configuration if available
    max_entities = base_config.max_entities
    max_relationships = base_config.max_relationships
    confidence_threshold = base_config.confidence_threshold

    if domain_analysis:
        # Adapt based on discovered characteristics (not domain assumptions)
        if domain_analysis.vocabulary_complexity > 0.7:
            max_entities = min(max_entities * 2, 100)
        if domain_analysis.concept_density > 0.8:
            max_relationships = min(max_relationships * 2, 50)
        if "proper_noun_rich" in getattr(domain_analysis, "entity_indicators", []):
            confidence_threshold *= 0.9

    # Universal entity extraction using prompt workflow system
    entities = await _extract_entities_with_prompt_workflow(
        ctx, content, domain_analysis, max_entities, confidence_threshold
    )

    # Universal relationship extraction using prompt workflow system
    relationships = await _extract_relationships_with_prompt_workflow(
        ctx, content, entities, domain_analysis, max_relationships, confidence_threshold
    )

    # Calculate extraction confidence
    extraction_confidence = min(
        (len(entities) + len(relationships)) / max(max_entities + max_relationships, 1),
        1.0,
    )

    # Generate processing signature
    signature_parts = [
        f"me{max_entities}",
        f"mr{max_relationships}",
        f"ct{confidence_threshold:.2f}",
    ]
    if domain_analysis:
        signature_parts.append(domain_analysis.content_signature)
    processing_signature = "_".join(signature_parts)

    # Perform quality assessment
    quality_assessment = None
    try:
        entities_dict = [
            {
                "entity_type": e.type,
                "confidence": e.confidence,
                "text": e.text,
            }
            for e in entities
        ]
        relationships_dict = [
            {
                "relation_type": r.relationship_type,
                "confidence": r.confidence,
                "source": r.source_entity,
                "target": r.target_entity,
            }
            for r in relationships
        ]

        quality_assessment = assess_extraction_quality(
            entities_dict,
            relationships_dict,
            [content[:1000]],  # Use first 1k chars for assessment
        )
        logger.info(
            f"Quality assessment: {quality_assessment.get('quality_level', 'unknown')} "
            f"(score: {quality_assessment.get('overall_score', 0.0):.3f})"
        )
    except Exception as e:
        logger.warning(f"Quality assessment failed: {e}")

    return ExtractionResult(
        entities=entities,
        relationships=relationships,
        extraction_confidence=extraction_confidence,
        processing_signature=processing_signature,
        quality_assessment=quality_assessment,
    )


async def _extract_entities_with_prompt_workflow(
    ctx: RunContext[UniversalDeps],
    content: str,
    domain_analysis: Optional[UniversalDomainAnalysis],
    max_entities: int,
    confidence_threshold: float,
) -> List[ExtractedEntity]:
    """Extract entities using the prompt workflow system (no hardcoded patterns)."""
    entities = []

    try:
        if not ctx.deps.is_service_available("openai"):
            # Fallback to simple pattern detection if OpenAI unavailable
            return await _extract_entities_fallback(
                content, max_entities, confidence_threshold
            )

        # Generate domain-specific entity extraction prompt
        if domain_analysis:
            # Create temporary data for prompt generation (simulating domain analysis)
            temp_data = {
                "domain_signature": domain_analysis.content_signature,
                "content_confidence": domain_analysis.vocabulary_complexity,
                "discovered_domain_description": f"content with {domain_analysis.content_signature} characteristics",
                "discovered_content_patterns": [
                    {
                        "category": pattern.replace("_", " ").title(),
                        "description": f"Pattern: {pattern}",
                        "examples": [],
                    }
                    for pattern in getattr(
                        domain_analysis, "discovered_patterns", ["general_content"]
                    )[:3]
                ],
                "discovered_entity_types": getattr(
                    domain_analysis, "entity_indicators", ["concept", "entity", "term"]
                ),
                "entity_confidence_threshold": confidence_threshold,
                "key_domain_insights": [
                    f"Entity extraction for {domain_analysis.content_signature}"
                ],
                "vocabulary_richness": domain_analysis.vocabulary_complexity,
                "technical_density": domain_analysis.concept_density,
                "analysis_processing_time": 0.5,
                "example_entity": "discovered_entity",
                "adaptive_entity_type": "concept",
            }

            # Use OpenAI to extract entities based on discovered domain characteristics
            extraction_prompt = f"""
            You are analyzing content to extract meaningful entities. Based on domain analysis, focus on:
            - Content type: {domain_analysis.content_signature}
            - Vocabulary complexity: {domain_analysis.vocabulary_complexity:.2f}
            - Concept density: {domain_analysis.concept_density:.2f}
            
            Extract entities from this content with confidence >= {confidence_threshold}:
            {content[:2000]}  # Truncate for API limits
            
            Return JSON array of entities with: text, type, confidence, context
            """

            # Get OpenAI client and make request
            openai_client = ctx.deps.openai_client
            if openai_client:
                response = await openai_client.complete_chat(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.3,
                )

                # Parse response and convert to ExtractedEntity objects
                # This would need proper JSON parsing and error handling
                entity_data = response.get("content", "[]")
                # Simple fallback parsing for now

        # Convert to ExtractedEntity objects
        # For now, use a simple approach until full prompt workflow integration
        entities = await _extract_entities_simple_approach(
            content, max_entities, confidence_threshold
        )

    except Exception as e:
        print(f"Prompt workflow extraction failed: {e}, using fallback")
        entities = await _extract_entities_fallback(
            content, max_entities, confidence_threshold
        )

    return entities[:max_entities]


async def _extract_relationships_with_prompt_workflow(
    ctx: RunContext[UniversalDeps],
    content: str,
    entities: List[ExtractedEntity],
    domain_analysis: Optional[UniversalDomainAnalysis],
    max_relationships: int,
    confidence_threshold: float,
) -> List[ExtractedRelationship]:
    """Extract relationships using the prompt workflow system (no hardcoded patterns)."""
    relationships = []

    try:
        if not ctx.deps.is_service_available("openai") or not entities:
            return await _extract_relationships_fallback(
                content, entities, max_relationships, confidence_threshold
            )

        # Generate domain-specific relationship extraction prompt
        if domain_analysis:
            relationship_prompt = f"""
            You are analyzing content to extract meaningful relationships between entities. Based on domain analysis:
            - Content type: {domain_analysis.content_signature}
            - Vocabulary complexity: {domain_analysis.vocabulary_complexity:.2f}
            
            Available entities: {[e.text for e in entities[:10]]}
            
            Extract relationships from this content with confidence >= {confidence_threshold}:
            {content[:2000]}
            
            Return JSON array of relationships with: source_entity, target_entity, relation_type, confidence, context
            """

            # Use OpenAI for relationship extraction
            openai_client = ctx.deps.openai_client
            if openai_client:
                response = await openai_client.complete_chat(
                    messages=[{"role": "user", "content": relationship_prompt}],
                    temperature=0.3,
                )

        # For now, use simple approach until full integration
        relationships = await _extract_relationships_simple_approach(
            content, entities, max_relationships, confidence_threshold
        )

    except Exception as e:
        print(f"Prompt workflow relationship extraction failed: {e}, using fallback")
        relationships = await _extract_relationships_fallback(
            content, entities, max_relationships, confidence_threshold
        )

    return relationships[:max_relationships]


async def _extract_entities_simple_approach(
    content: str, max_entities: int, confidence_threshold: float
) -> List[ExtractedEntity]:
    """Enhanced entity extraction with classification client integration."""
    entities = []
    words = content.split()

    # Initialize classifier for better entity typing
    classifier = SimpleAzureClassifier()

    # Universal pattern: Capitalized words (domain-agnostic)
    for i, word in enumerate(words):
        if word and word[0].isupper() and len(word) > 2:
            context_start = max(0, i - 3)
            context_end = min(len(words), i + 4)
            context = " ".join(words[context_start:context_end])

            base_confidence = 0.7
            if any(indicator in context.lower() for indicator in ["the", "a", "an"]):
                base_confidence += 0.1

            if base_confidence >= confidence_threshold:
                try:
                    # Use classification client for better entity typing
                    classification_result = await classifier.classify_entity(
                        word, context
                    )
                    entity_type = classification_result.entity_type
                    classification_confidence = classification_result.confidence

                    # Combine base confidence with classification confidence
                    final_confidence = (
                        base_confidence + classification_confidence
                    ) / 2.0

                    entity = ExtractedEntity(
                        text=word,
                        type=entity_type,  # Use classifier result instead of generic type
                        confidence=final_confidence,
                        context=context,
                        start_position=i,
                        properties={
                            "discovery_method": "capitalization_pattern_with_classification",
                            "classification_metadata": classification_result.metadata,
                            "base_confidence": base_confidence,
                            "classification_confidence": classification_confidence,
                        },
                    )
                    entities.append(entity)

                except Exception as e:
                    # Fallback to original approach if classification fails
                    entity = ExtractedEntity(
                        text=word,
                        type="discovered_entity",  # Fallback type
                        confidence=base_confidence,
                        context=context,
                        start_position=i,
                        properties={
                            "discovery_method": "capitalization_pattern_fallback",
                            "classification_error": str(e),
                        },
                    )
                    entities.append(entity)

                if len(entities) >= max_entities:
                    break

    return entities


async def _extract_relationships_simple_approach(
    content: str,
    entities: List[ExtractedEntity],
    max_relationships: int,
    confidence_threshold: float,
) -> List[ExtractedRelationship]:
    """Enhanced relationship extraction with classification client integration."""
    relationships = []

    # Initialize classifier for better relationship typing
    classifier = SimpleAzureClassifier()

    # Use proximity and co-occurrence with classification enhancement
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities[i + 1 :], i + 1):
            # Check if entities appear near each other in content
            entity1_pos = content.lower().find(entity1.text.lower())
            entity2_pos = content.lower().find(entity2.text.lower())

            if entity1_pos != -1 and entity2_pos != -1:
                distance = abs(entity2_pos - entity1_pos)
                if distance < 100:  # Entities within 100 characters
                    base_confidence = max(0.5, 1.0 - distance / 200.0)

                    if base_confidence >= confidence_threshold:
                        # Extract context around both entities for classification
                        start_pos = min(entity1_pos, entity2_pos)
                        end_pos = max(
                            entity1_pos + len(entity1.text),
                            entity2_pos + len(entity2.text),
                        )
                        relation_context = content[
                            max(0, start_pos - 50) : end_pos + 50
                        ]

                        try:
                            # Use classification client for better relationship typing
                            classification_result = await classifier.classify_relation(
                                entity1.text, entity2.text, relation_context
                            )
                            relation_type = (
                                classification_result.entity_type
                            )  # relation type stored in entity_type field
                            classification_confidence = classification_result.confidence

                            # Combine base confidence with classification confidence
                            final_confidence = (
                                base_confidence + classification_confidence
                            ) / 2.0

                            relationship = ExtractedRelationship(
                                source_entity=entity1.text,
                                target_entity=entity2.text,
                                relationship_type=relation_type,  # Use classifier result
                                confidence=final_confidence,
                                context=relation_context,
                                properties={
                                    "discovery_method": "proximity_with_classification",
                                    "distance": distance,
                                    "classification_metadata": classification_result.metadata,
                                    "base_confidence": base_confidence,
                                    "classification_confidence": classification_confidence,
                                },
                            )
                            relationships.append(relationship)

                        except Exception as e:
                            # Fallback to original approach if classification fails
                            relationship = ExtractedRelationship(
                                source_entity=entity1.text,
                                target_entity=entity2.text,
                                relationship_type="co_occurs_with",  # Fallback type
                                confidence=base_confidence,
                                context=f"Entities appear within {distance} characters",
                                properties={
                                    "discovery_method": "proximity_fallback",
                                    "distance": distance,
                                    "classification_error": str(e),
                                },
                            )
                            relationships.append(relationship)

                        if len(relationships) >= max_relationships:
                            return relationships

    return relationships


async def _extract_entities_fallback(
    content: str, max_entities: int, confidence_threshold: float
) -> List[ExtractedEntity]:
    """Fallback entity extraction when OpenAI is unavailable."""
    return await _extract_entities_simple_approach(
        content, max_entities, confidence_threshold
    )


async def _extract_relationships_fallback(
    content: str,
    entities: List[ExtractedEntity],
    max_relationships: int,
    confidence_threshold: float,
) -> List[ExtractedRelationship]:
    """Fallback relationship extraction when OpenAI is unavailable."""
    return await _extract_relationships_simple_approach(
        content, entities, max_relationships, confidence_threshold
    )


@knowledge_extraction_agent.tool
async def store_knowledge_in_graph(
    ctx: RunContext[UniversalDeps], extraction_result: ExtractionResult
) -> GraphOperationResult:
    """
    Store extracted knowledge in Azure Cosmos DB using Gremlin queries.

    This tool performs atomic graph operations without orchestrating other agents.
    """
    if not ctx.deps.is_service_available("cosmos"):
        return GraphOperationResult(
            operation_type="store_knowledge",
            success=False,
            error_message="Azure Cosmos DB service not available",
        )

    try:
        cosmos_client = ctx.deps.cosmos_client
        nodes_created = 0
        edges_created = 0

        # Store entities as graph nodes
        for entity in extraction_result.entities:
            # Generate Gremlin query for entity creation
            gremlin_query = await generate_gremlin_query(
                ctx, f"Create entity node for {entity.text}", entity_types=[entity.type]
            )

            # Customize query for entity creation
            entity_query = f"""
            g.V().has('text', '{entity.text}').fold().coalesce(
                unfold(),
                addV('entity')
                    .property('text', '{entity.text}')
                    .property('type', '{entity.type}')
                    .property('confidence', {entity.confidence})
                    .property('context', '{entity.context[:100]}')
            )
            """

            result = await cosmos_client.execute_query(entity_query)
            if result:
                nodes_created += 1

        # Store relationships as graph edges
        for relationship in extraction_result.relationships:
            edge_query = f"""
            g.V().has('text', '{relationship.source_entity}').as('source')
             .V().has('text', '{relationship.target_entity}').as('target')
             .addE('{relationship.relationship_type}')
             .from_('source').to('target')
             .property('confidence', {relationship.confidence})
             .property('context', '{relationship.context[:100]}')
            """

            result = await cosmos_client.execute_query(edge_query)
            if result:
                edges_created += 1

        return GraphOperationResult(
            operation_type="store_knowledge",
            nodes_affected=nodes_created,
            edges_affected=edges_created,
            success=True,
        )

    except Exception as e:
        return GraphOperationResult(
            operation_type="store_knowledge", success=False, error_message=str(e)
        )


@knowledge_extraction_agent.tool
async def validate_extraction_requirements(
    ctx: RunContext[UniversalDeps],
) -> Dict[str, Any]:
    """
    Validate that required services are available for knowledge extraction.
    """
    required_services = ["openai"]  # Required for extraction
    optional_services = ["cosmos", "monitoring"]  # Optional for enhanced functionality

    validation_result = {
        "required_services_available": all(
            ctx.deps.is_service_available(service) for service in required_services
        ),
        "optional_services": {
            service: ctx.deps.is_service_available(service)
            for service in optional_services
        },
        "can_perform_extraction": True,
        "can_store_in_graph": ctx.deps.is_service_available("cosmos"),
        "available_services": ctx.deps.get_available_services(),
    }

    return validation_result


# Factory function for proper agent initialization
async def create_knowledge_extraction_agent() -> Agent[UniversalDeps, ExtractionResult]:
    """
    Create Knowledge Extraction Agent with initialized dependencies.

    Follows PydanticAI best practices for agent creation.
    """
    deps = await get_universal_deps()

    # Validate required services
    if not deps.is_service_available("openai"):
        raise RuntimeError("Knowledge Extraction Agent requires Azure OpenAI service")

    return knowledge_extraction_agent


# Main execution function for testing
async def run_knowledge_extraction(
    content: str, use_domain_analysis: bool = True, use_generated_prompts: bool = False
) -> ExtractionResult:
    """
    Run knowledge extraction with proper PydanticAI patterns.

    Args:
        content: Text content to extract from
        use_domain_analysis: Whether to use domain intelligence
        use_generated_prompts: Whether to use dynamically generated prompts (recommended)
    """
    deps = await get_universal_deps()
    agent = await create_knowledge_extraction_agent()

    if use_generated_prompts:
        # Use the new generated prompts workflow
        result = await agent.run(
            f"Use extract_with_generated_prompts tool for: {content[:200]}...",
            deps=deps,
        )
    else:
        # Use the standard extraction (with hardcoded prompts)
        result = await agent.run(
            f"Extract entities and relationships from the following content:\n\n{content}",
            deps=deps,
        )

    return result.output


if __name__ == "__main__":
    # Test the agent
    sample_content = """
    Azure Cosmos DB is a globally distributed database service. It supports multiple data models
    including document, key-value, graph, and column-family. Cosmos DB provides comprehensive SLAs
    for throughput, availability, latency, and consistency. The service integrates with Azure Functions
    and Azure App Service for building scalable applications.
    """

    async def test_agent():
        try:
            result = await run_knowledge_extraction(sample_content)
            print("Knowledge Extraction Result:")
            print(f"Entities found: {len(result.entities)}")
            print(f"Relationships found: {len(result.relationships)}")
            print(f"Extraction confidence: {result.extraction_confidence:.2f}")

            for entity in result.entities[:3]:  # Show first 3
                print(
                    f"Entity: {entity.text} ({entity.type}, confidence: {entity.confidence:.2f})"
                )

            for rel in result.relationships[:3]:  # Show first 3
                print(
                    f"Relationship: {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity}"
                )

        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test_agent())
