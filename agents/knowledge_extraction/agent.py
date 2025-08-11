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
    
    # Required fields for schema compliance
    processing_time: float = Field(default=0.0, ge=0.0, description="Processing duration in seconds")
    extracted_concepts: List[str] = Field(default_factory=list, description="Key concepts extracted from content")
    
    # Graph operation fields
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


# Use centralized Azure PydanticAI provider
from agents.core.azure_pydantic_provider import get_azure_openai_model

# Use centralized toolset management
from agents.core.agent_toolsets import knowledge_extraction_toolset

# Create the Knowledge Extraction Agent with proper PydanticAI patterns using toolsets
knowledge_extraction_agent = Agent[UniversalDeps, ExtractionResult](
    get_azure_openai_model(),
    output_type=ExtractionResult,
    toolsets=[knowledge_extraction_toolset],
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



async def _extract_with_enhanced_agent_guidance_optimized_internal(
    ctx: RunContext[UniversalDeps], content: str, use_domain_analysis: bool = True
) -> ExtractionResult:
    """
    OPTIMIZED ENHANCED EXTRACTION: Fast version with Agent 1 guidance.
    
    Optimizations:
    - Simplified Agent 1 calls (fewer steps)
    - Limited word processing (top candidates only)  
    - Faster relationship extraction
    """
    import time
    start_time = time.time()
    
    # Quick domain analysis if requested
    if use_domain_analysis:
        try:
            domain_result = await domain_intelligence_agent.run(
                f"Quick analysis for extraction: {content[:1000]}",  # Increased from 300 to 1000 chars
                deps=ctx.deps,
                usage=ctx.usage,
            )
            domain_analysis = domain_result.output
            
            # Use domain analysis for adaptive parameters
            complexity_factor = domain_analysis.characteristics.vocabulary_complexity
            max_entities = min(int(20 + complexity_factor * 15), 50)  # Increased cap to 50 for larger content
            max_relationships = min(int(15 + complexity_factor * 10), 30)  # Increased cap to 30 for larger content
            confidence_threshold = max(0.6, 0.8 - complexity_factor * 0.2)
            
        except Exception:
            # Fallback to defaults if Agent 1 fails
            max_entities = 25  # Increased fallback values for larger content processing
            max_relationships = 20  
            confidence_threshold = 0.7
            domain_analysis = None
    else:
        max_entities = 25  # Increased fallback values for larger content processing
        max_relationships = 20
        confidence_threshold = 0.7
        domain_analysis = None

    # ✅ TRUE LLM-BASED ENTITY EXTRACTION (Replaces primitive word-splitting)
    entities = []
    logger.info("🧠 Using LLM for semantic entity extraction (no word-splitting)")
    
    try:
        # Get LLM client from dependencies
        openai_client = ctx.deps.openai_client
        
        # Use Agent 1's entity type predictions if available (REAL data from Agent 1)
        predicted_types = []
        if domain_analysis and hasattr(domain_analysis, 'characteristics'):
            # Get entity types directly from Agent 1's LLM analysis using REAL dependencies
            try:
                from agents.core.agent_toolsets import predict_entity_types
                
                # Create proper RunContext with REAL Azure dependencies (not mock)
                class RealRunContext:
                    def __init__(self, real_deps):
                        self.deps = real_deps  # These are the REAL Azure service dependencies
                        self.usage = None
                
                real_ctx = RealRunContext(ctx.deps)  # Pass through REAL Azure dependencies
                type_predictions = await predict_entity_types(
                    real_ctx, content, domain_analysis.characteristics
                )
                predicted_types = type_predictions.get('predicted_entity_types', [])
                logger.info(f"🎯 Agent 1 LLM predicted entity types (REAL): {predicted_types}")
            except Exception as e:
                logger.warning(f"Could not get Agent 1 predictions: {e}")
        
        # Create LLM extraction prompt
        extraction_prompt = f"""Extract entities from the following content using semantic understanding.

Content to analyze:
{content[:2000]}

Entity Types to Focus On: {', '.join(predicted_types) if predicted_types else 'Any meaningful concepts'}

Task: Extract the most important entities (concepts, objects, processes) from this content.

Requirements:
1. Extract complete meaningful phrases, not individual words
2. Focus on significant concepts that represent important entities
3. Each entity should be 2-10 words describing a complete concept
4. Provide confidence scores based on semantic importance
5. Include surrounding context for each entity

Return results in this JSON format:
{{
  "entities": [
    {{
      "text": "complete entity phrase",
      "type": "entity_category",
      "confidence": 0.85,
      "context": "surrounding context"
    }}
  ]
}}

Maximum entities: {max_entities}"""

        # Get LLM semantic extraction
        response = await openai_client.get_completion(
            extraction_prompt,
            max_tokens=1200,  # Increased from 800 to handle larger content processing
            temperature=0.3  # Balanced creativity/consistency
        )
        
        # Parse LLM response (supports markdown JSON blocks)
        import json
        import re
        
        # First try to extract JSON from markdown code blocks
        markdown_json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if markdown_json_match:
            json_content = markdown_json_match.group(1)
        else:
            # Fallback to raw JSON extraction
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            json_content = json_match.group() if json_match else None
            
        if json_content:
            llm_result = json.loads(json_content)
            extracted_entities = llm_result.get('entities', [])
            
            # Convert to ExtractedEntity objects
            for i, entity_data in enumerate(extracted_entities[:max_entities]):
                entity = ExtractedEntity(
                    text=entity_data.get('text', ''),
                    type=entity_data.get('type', 'concept'),  # Fixed field name
                    confidence=float(entity_data.get('confidence', 0.7)),
                    context=entity_data.get('context', ''),
                    positions=[],  # Add positions field
                    metadata={
                        "extraction_method": "llm_semantic",
                        "agent1_guided": bool(predicted_types),
                        "llm_generated": True
                    }
                )
                entities.append(entity)
                
            logger.info(f"✅ LLM extracted {len(entities)} semantic entities")
        else:
            logger.error("❌ Could not parse LLM entity extraction response")
            
    except Exception as e:
        logger.error(f"❌ LLM entity extraction failed: {e}")
        # No fallback to primitive logic - LLM must work

    # ✅ TRUE LLM-BASED RELATIONSHIP EXTRACTION (Replaces primitive proximity logic)
    relationships = []
    if entities:
        logger.info("🔗 Using LLM for semantic relationship extraction")
        
        try:
            # Create entity pairs for relationship analysis
            entity_texts = [e.text for e in entities[:8]]  # Limit to first 8 entities
            
            # Create LLM relationship prompt
            relationship_prompt = f"""Analyze the relationships between entities in this content using semantic understanding.

Content:
{content[:2000]}

Entities found:
{chr(10).join([f"- {text}" for text in entity_texts])}

Task: Identify the most important semantic relationships between these entities.

Requirements:
1. Only identify relationships that are explicitly or implicitly described in the content
2. Use meaningful relationship types that describe the actual connection
3. Provide confidence based on how clearly the relationship is expressed
4. Include context showing where the relationship is indicated

Return results in this JSON format:
{{
  "relationships": [
    {{
      "source": "source entity text",
      "target": "target entity text", 
      "relation": "meaningful_relationship_type",
      "confidence": 0.85,
      "context": "text evidence for relationship"
    }}
  ]
}}

Maximum relationships: {max_relationships}"""

            # Get LLM relationship extraction
            response = await openai_client.get_completion(
                relationship_prompt,
                max_tokens=800,  # Increased from 600 to handle larger content processing
                temperature=0.2  # Lower temperature for more consistent relationships
            )
            
            # Parse LLM response with robust error handling (supports markdown JSON blocks)
            # First try to extract JSON from markdown code blocks
            markdown_json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if markdown_json_match:
                json_content = markdown_json_match.group(1)
            else:
                # Fallback to raw JSON extraction
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_content = json_match.group() if json_match else None
            
            if json_content:
                try:
                    llm_result = json.loads(json_content)
                    extracted_relationships = llm_result.get('relationships', [])
                except json.JSONDecodeError as json_error:
                    logger.warning(f"JSON parsing failed: {json_error}")
                    # Try to find just the relationships array
                    relationships_match = re.search(r'"relationships"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    if relationships_match:
                        try:
                            # Try to parse just the relationships array
                            relationships_json = f'{{"relationships":[{relationships_match.group(1)}]}}'
                            llm_result = json.loads(relationships_json)
                            extracted_relationships = llm_result.get('relationships', [])
                            logger.info("✅ Recovered relationships from partial JSON")
                        except json.JSONDecodeError:
                            logger.error("❌ Could not recover relationships from malformed JSON")
                            extracted_relationships = []
                    else:
                        extracted_relationships = []
            else:
                logger.error("❌ No JSON structure found in LLM response")
                extracted_relationships = []
                
            # Convert to ExtractedRelationship objects
            for rel_data in extracted_relationships[:max_relationships]:
                # Validate that both entities exist in our extracted entities
                source_text = rel_data.get('source', '')
                target_text = rel_data.get('target', '')
                
                if source_text in entity_texts and target_text in entity_texts:
                    relationship = ExtractedRelationship(
                        source=source_text,
                        target=target_text,
                        relation=rel_data.get('relation', 'discovered_relation'),
                        confidence=float(rel_data.get('confidence', 0.7)),
                        context=rel_data.get('context', ''),
                        metadata={
                            "extraction_method": "llm_semantic",
                            "agent1_guided": bool(predicted_types),
                            "llm_generated": True
                        }
                    )
                    relationships.append(relationship)
            
            logger.info(f"✅ LLM extracted {len(relationships)} semantic relationships")
                
        except Exception as e:
            logger.error(f"❌ LLM relationship extraction failed: {e}")
            # No fallback to primitive logic - LLM must work
    
    # Calculate confidence
    extraction_confidence = min((len(entities) + len(relationships)) / (max_entities + max_relationships), 1.0)
    
    # Generate signature
    signature = f"optimized_enhanced_me{max_entities}_mr{max_relationships}_ct{confidence_threshold:.2f}"
    if domain_analysis:
        signature += f"_vc{domain_analysis.characteristics.vocabulary_complexity:.2f}"
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Extract key concepts from entities (required field)
    extracted_concepts = []
    for entity in entities:
        if entity.text and len(entity.text.strip()) > 2:
            # Use entity text as concept, clean up
            concept = entity.text.strip().lower()
            if concept not in extracted_concepts and len(concept) > 3:
                extracted_concepts.append(concept)
    
    # Add relationship-based concepts
    for rel in relationships:
        if rel.relation and len(rel.relation.strip()) > 2:
            concept = rel.relation.strip().lower().replace('_', ' ')
            if concept not in extracted_concepts and len(concept) > 3:
                extracted_concepts.append(concept)
    
    # Limit concepts to reasonable number
    extracted_concepts = extracted_concepts[:15]
    
    return ExtractionResult(
        entities=entities,
        relationships=relationships,
        extraction_confidence=extraction_confidence,
        processing_signature=signature,
        processing_time=processing_time,
        extracted_concepts=extracted_concepts,
        quality_assessment={
            "extraction_method": "optimized_enhanced",
            "domain_analysis_used": domain_analysis is not None,
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "processing_optimized": True,
            "concepts_extracted": len(extracted_concepts)
        }
    )








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
            nodes_affected=0,  # No nodes affected when service unavailable
            edges_affected=0,  # No edges affected when service unavailable
            success=False,
            error_message="Azure Cosmos DB service not available",
        )

    try:
        cosmos_client = ctx.deps.cosmos_client
        nodes_created = 0
        edges_created = 0

        # Store entities as graph nodes
        for entity in extraction_result.entities:
            # Direct Gremlin query with proper partition key (skip generate_gremlin_query)
            # Escape single quotes and handle None values
            safe_text = (entity.text or "").replace("'", "\\'")[:100]
            safe_context = (entity.context or "").replace("'", "\\'")[:100]  
            safe_entity_type = (entity.type or "concept").replace("'", "\\'")  # Fixed field name
            
            entity_query = f"""
            g.V().has('text', '{safe_text}').fold().coalesce(
                unfold(),
                addV('entity')
                    .property('text', '{safe_text}')
                    .property('entity_type', '{safe_entity_type}')
                    .property('confidence', {entity.confidence})
                    .property('context', '{safe_context}')
                    .property('partitionKey', '{safe_entity_type}')
            )
            """

            result = await cosmos_client.execute_query(entity_query)
            if result:
                nodes_created += 1

        # Store relationships as graph edges  
        for relationship in extraction_result.relationships:
            # Escape single quotes and handle None values
            safe_source = (relationship.source or "").replace("'", "\\'")[:100]
            safe_target = (relationship.target or "").replace("'", "\\'")[:100]
            safe_relation = (relationship.relation or "discovered_relation").replace("'", "\\'")
            safe_context = (relationship.context or "").replace("'", "\\'")[:100]
            
            edge_query = f"""
            g.V().has('text', '{safe_source}').as('source')
             .V().has('text', '{safe_target}').as('target')
             .addE('{safe_relation}')
             .from('source').to('target')
             .property('confidence', {relationship.confidence})
             .property('context', '{safe_context}')
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
            operation_type="store_knowledge",
            nodes_affected=0,  # No nodes affected on error
            edges_affected=0,  # No edges affected on error
            success=False, 
            error_message=str(e)
        )


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
    content: str, use_domain_analysis: bool = True
) -> ExtractionResult:
    """
    Run knowledge extraction with Agent 1 → Agent 2 inter-agent communication.
    
    Uses TRUE LLM-based semantic extraction with Agent 1 guidance (no primitive logic).

    Args:
        content: Text content to extract from
        use_domain_analysis: Whether to use domain intelligence (recommended: True)
    """
    deps = await get_universal_deps()
    agent = await create_knowledge_extraction_agent()

    # Use TRUE LLM extraction with Agent 1 guidance (optimized version)
    result = await agent.run(
        f"Use the extract_entities_and_relationships tool to extract entities and relationships from: {content[:200]}...",
        deps=deps,
    )

    return result.output
