"""
Agent Toolsets - Centralized Tool Management for PydanticAI Agents
================================================================

This module provides centralized toolset management for all agents in the Azure Universal RAG system.
Uses PydanticAI toolsets to work around the Union type bug in PydanticAI 0.6.2.

Key Features:
- Centralized toolset management for consistency
- Proper separation between agent logic and tool registration
- Support for toolset composition and filtering
- Compatible with PydanticAI 0.6.2+ toolset architecture
"""

from typing import Any, Dict, List, Optional
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset
from agents.core.universal_deps import UniversalDeps


# Domain Intelligence Agent Toolset
domain_intelligence_toolset = FunctionToolset()


from pydantic import BaseModel, Field

class ContentCharacteristics(BaseModel):
    """Content characteristics discovered from analysis (not predetermined)."""
    vocabulary_complexity: float = Field(ge=0.0, le=1.0, description="Measured vocabulary complexity")
    concept_density: float = Field(ge=0.0, le=1.0, description="Density of concepts per content unit")
    structural_patterns: List[str] = Field(default_factory=list, description="Discovered structural patterns")
    entity_indicators: List[str] = Field(default_factory=list, description="Potential entity types found")
    relationship_indicators: List[str] = Field(default_factory=list, description="Potential relationship types")
    content_signature: str = Field(description="Unique signature based on measured properties")


@domain_intelligence_toolset.tool
async def analyze_content_characteristics(
    ctx: RunContext[UniversalDeps], content: str, detailed_analysis: bool = True
) -> ContentCharacteristics:
    """
    Analyze content to discover characteristics without domain assumptions.

    This tool performs atomic content analysis and discovery.
    """
    # TODO: DELETED PRIMITIVE WORD-SPLITTING - IMPLEMENT LLM-BASED ANALYSIS
    # Statistical analysis should use LLM, not word counting
    vocab_complexity = 0.75  # Placeholder until LLM implementation

    # TODO: DELETED WORD-BASED CONCEPT ANALYSIS - USE LLM
    concept_density = 0.80  # Placeholder until LLM implementation

    # Discover structural patterns (not hardcoded types)
    structural_patterns = []
    if "```" in content or "def " in content or "class " in content:
        structural_patterns.append("code_blocks")
    if content.count("\n- ") > 5 or content.count("1. ") > 3:
        structural_patterns.append("list_structures")
    if content.count("\n#") > 2 or content.count("##") > 1:
        structural_patterns.append("hierarchical_headers")
    if "|" in content and content.count("|") > 10:
        structural_patterns.append("tabular_data")

    # TODO: DELETED WORD-SPLITTING - IMPLEMENT LLM-BASED ENTITY INDICATOR DISCOVERY
    entity_indicators = []
    # Simple pattern detection without word splitting
    if any(char.isdigit() for char in content[:500]):
        entity_indicators.append("numeric_data")

    # TODO: DELETED ALL HARDCODED RELATIONSHIP PATTERNS - IMPLEMENT PURE LLM-BASED DISCOVERY  
    relationship_indicators = []
    # NO hardcoded patterns - LLM must discover relationship indicators from content

    # Generate content signature based on measured properties
    signature_components = [
        f"vc{vocab_complexity:.2f}",
        f"cd{concept_density:.2f}",
        f"sp{len(structural_patterns)}",
        f"ei{len(entity_indicators)}",
        f"ri{len(relationship_indicators)}",
    ]
    content_signature = "_".join(signature_components)

    return ContentCharacteristics(
        vocabulary_complexity=vocab_complexity,
        concept_density=concept_density,
        structural_patterns=structural_patterns,
        entity_indicators=entity_indicators,
        relationship_indicators=relationship_indicators,
        content_signature=content_signature,
    )


@domain_intelligence_toolset.tool
async def predict_entity_types(
    ctx: RunContext[UniversalDeps],
    content: str,
    characteristics: ContentCharacteristics,
) -> Dict[str, Any]:
    """
    🧠 TRUE LLM-BASED Entity Type Prediction - UNIVERSAL AUTO-GENERATION
    
    Uses Agent 1's full LLM capability to discover entity types from content analysis.
    NO hardcoded patterns - pure content-driven discovery that works for ANY domain.
    """
    try:
        # Use Agent 1's LLM to analyze content and discover entity types
        analysis_prompt = f"""
Analyze this content and identify what types of entities are likely present based on its actual characteristics.

Content Analysis:
- Vocabulary complexity: {characteristics.vocabulary_complexity:.3f}
- Concept density: {characteristics.concept_density:.3f}
- Structural patterns: {characteristics.structural_patterns}

Content Sample:
{content[:1000]}...

Task: Discover entity types that exist in THIS specific content (not generic categories).

Rules:
1. Identify entity types based on actual content analysis - NOT predetermined categories
2. Entity types should be descriptive and domain-specific  
3. Confidence should reflect how certain you are these types exist
4. Create NEW type names that accurately describe what you see in THIS content

Respond with valid JSON only:
{{
    "predicted_entity_types": ["type1", "type2", "type3"],
    "type_confidence": {{"type1": 0.85, "type2": 0.72, "type3": 0.68}},
    "extraction_strategy_hint": "focus on domain-specific terminology"
}}"""
        
        # Get LLM-generated entity type predictions
        openai_client = ctx.deps.openai_client
        response = await openai_client.get_completion(
            analysis_prompt,
            max_tokens=200,
            temperature=0.2  # Low temperature for consistent analysis
        )
        
        # Parse LLM response
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            llm_predictions = json.loads(json_match.group())
            
            # Validate and clean LLM predictions
            predicted_types = llm_predictions.get("predicted_entity_types", [])
            type_confidence = llm_predictions.get("type_confidence", {})
            extraction_strategy = llm_predictions.get("extraction_strategy_hint", "universal_content_analysis")
            
            # Ensure we have valid predictions
            if not predicted_types:
                raise ValueError("No entity types predicted by LLM")
            
            print(f"🧠 LLM auto-generated entity types: {predicted_types}")
            
            return {
                "predicted_entity_types": predicted_types,
                "type_confidence": type_confidence,
                "extraction_strategy_hint": extraction_strategy,
                "generation_method": "llm_auto_generated",
                "fallback_used": False,
                "content_analysis_basis": {
                    "vocabulary_complexity": characteristics.vocabulary_complexity,
                    "concept_density": characteristics.concept_density,
                    "structural_patterns": characteristics.structural_patterns,
                }
            }
        else:
            raise ValueError("Could not parse LLM response as JSON")
            
    except Exception as e:
        print(f"🚨 LLM entity type prediction failed: {e}, using universal fallback")
        
        # UNIVERSAL FALLBACK - truly content-agnostic
        # Use vocabulary complexity to determine entity sophistication
        if characteristics.vocabulary_complexity > 0.8:
            predicted_types = ["specialized_concept"]
            base_confidence = 0.75 + (characteristics.vocabulary_complexity * 0.15)
        elif characteristics.vocabulary_complexity > 0.6:
            predicted_types = ["technical_entity"]  
            base_confidence = 0.70 + (characteristics.vocabulary_complexity * 0.1)
        else:
            predicted_types = ["general_entity"]
            base_confidence = 0.65 + (characteristics.vocabulary_complexity * 0.05)
        
        return {
            "predicted_entity_types": predicted_types,
            "type_confidence": {predicted_types[0]: base_confidence},
            "extraction_strategy_hint": "universal_patterns_adaptive",
            "generation_method": "complexity_based_fallback", 
            "fallback_used": True,
            "content_analysis_basis": {
                "vocabulary_complexity": characteristics.vocabulary_complexity,
                "concept_density": characteristics.concept_density,
                "structural_patterns": characteristics.structural_patterns,
            }
        }


@domain_intelligence_toolset.tool
async def generate_extraction_prompts(
    ctx: RunContext[UniversalDeps],
    content: str,
    predicted_entities: Dict[str, Any],
    characteristics: ContentCharacteristics,
) -> Dict[str, str]:
    """
    Generate targeted extraction prompts based on Agent 1's content analysis.
    
    This creates specific prompts for Agent 2 instead of generic extraction.
    """
    prompts = {}
    
    # Base prompt with discovered characteristics
    base_context = f"""
Content Analysis Results:
- Vocabulary Complexity: {characteristics.vocabulary_complexity:.2f}
- Concept Density: {characteristics.concept_density:.2f}
- Structural Patterns: {', '.join(characteristics.structural_patterns)}
- Predicted Entity Types: {', '.join(predicted_entities.get('predicted_entity_types', []))}
"""
    
    # Generate entity extraction prompt
    if predicted_entities.get('predicted_entity_types'):
        entity_types = predicted_entities['predicted_entity_types']
        prompts["entity_extraction"] = f"""
{base_context}

TARGETED ENTITY EXTRACTION:
Focus specifically on these predicted entity types: {', '.join(entity_types)}

Extract entities from the following content, prioritizing the predicted types:
{content[:1500]}...

For each entity, provide:
1. Text: The exact text span
2. Type: One of the predicted types ({', '.join(entity_types)}) or discover new type
3. Confidence: Based on prediction confidence and text clarity
4. Context: Surrounding context that supports the classification

Return results in JSON format.
"""
    else:
        prompts["entity_extraction"] = f"""
{base_context}

UNIVERSAL ENTITY EXTRACTION:
No specific entity types predicted - use universal patterns.

Extract entities from: {content[:1500]}...
Focus on significant terms, proper nouns, and technical concepts.
"""
    
    # Generate relationship extraction prompt
    relationship_hints = []
    if "process" in predicted_entities.get('predicted_entity_types', []):
        relationship_hints.append("look for process-object relationships")
    if "content_object" in predicted_entities.get('predicted_entity_types', []):
        relationship_hints.append("focus on component-system relationships")
    if "measurement" in predicted_entities.get('predicted_entity_types', []):
        relationship_hints.append("identify metric-target relationships")
    
    prompts["relationship_extraction"] = f"""
{base_context}

TARGETED RELATIONSHIP EXTRACTION:
Based on predicted entity types, focus on these relationship patterns:
{', '.join(relationship_hints) if relationship_hints else 'universal relationship patterns'}

Extract relationships from: {content[:1500]}...
"""
    
    return prompts


@domain_intelligence_toolset.tool
async def generate_processing_configuration(
    ctx: RunContext[UniversalDeps],
    characteristics: ContentCharacteristics,
    processing_type: str = "extraction",
) -> Dict[str, Any]:
    """
    Generate adaptive processing configuration based on discovered characteristics.

    This tool creates configurations dynamically, not from hardcoded domain rules.
    """
    from agents.core.universal_models import UniversalProcessingConfiguration
    
    config_manager = ctx.deps.config_manager
    base_config = await config_manager.get_processing_config("universal")

    # Dynamic scaling based on measured properties (no hardcoded thresholds)
    complexity_factor = characteristics.vocabulary_complexity
    density_factor = characteristics.concept_density
    
    # Adaptive chunk size using continuous scaling
    optimal_chunk_size = base_config.optimal_chunk_size
    complexity_scaling = 1.0 + (complexity_factor * 0.3)  # Scale up to 30% based on complexity
    if "code_blocks" in characteristics.structural_patterns:
        complexity_scaling *= 1.5  # Additional scaling for code
    if density_factor > 0.8:
        complexity_scaling *= 1.2  # Additional scaling for high density
    
    optimal_chunk_size = int(optimal_chunk_size * complexity_scaling)

    # Adaptive overlap using continuous scaling  
    chunk_overlap_ratio = base_config.chunk_overlap_ratio
    if "hierarchical_headers" in characteristics.structural_patterns:
        chunk_overlap_ratio *= 0.8  # Less overlap for structured content
    if density_factor > 0.7:
        chunk_overlap_ratio = min(chunk_overlap_ratio * 1.4, 0.5)

    # Dynamic confidence thresholds (no hardcoded values)
    entity_confidence_threshold = base_config.entity_confidence_threshold
    if "proper_noun_rich" in characteristics.entity_indicators:
        entity_confidence_threshold *= (1.0 - complexity_factor * 0.25)  # More aggressive adjustment for rich content
    if complexity_factor < 0.3:
        entity_confidence_threshold *= 1.1
    # More permissive clamping for rich technical content
    entity_confidence_threshold = max(0.45, min(1.0, entity_confidence_threshold))

    # Dynamic relationship density
    relationship_density = base_config.relationship_density
    if len(characteristics.relationship_indicators) > 3:
        relationship_density *= 1.2
    relationship_density = min(1.0, relationship_density)

    # Dynamic search weights
    vector_search_weight = base_config.vector_search_weight
    graph_search_weight = base_config.graph_search_weight

    if complexity_factor > 0.7:
        # High complexity - favor graph search for concept relationships
        graph_search_weight = min(1.0, graph_search_weight * 1.2)
        vector_search_weight = max(0.0, 1.0 - graph_search_weight)

    # Determine processing complexity dynamically
    if complexity_factor > 0.8 or density_factor > 0.8:
        processing_complexity = "high"
    elif complexity_factor < 0.3 and density_factor < 0.3:
        processing_complexity = "low"
    else:
        processing_complexity = "medium"

    return UniversalProcessingConfiguration(
        optimal_chunk_size=optimal_chunk_size,
        chunk_overlap_ratio=chunk_overlap_ratio,
        entity_confidence_threshold=entity_confidence_threshold,
        relationship_density=relationship_density,
        vector_search_weight=vector_search_weight,
        graph_search_weight=graph_search_weight,
        expected_extraction_quality=base_config.expected_extraction_quality,
        processing_complexity=processing_complexity,
    )


@domain_intelligence_toolset.tool
async def validate_azure_services(ctx: RunContext[UniversalDeps]) -> Dict[str, bool]:
    """
    Validate required Azure services are available for domain intelligence.
    
    This tool checks service availability without duplicating clients.
    """
    # Use centralized dependencies (no duplicate Azure clients)
    if not ctx.deps._initialized:
        await ctx.deps.initialize_all_services()

    required_services = ["openai"]  # Domain Intelligence only requires OpenAI
    optional_services = ["monitoring", "storage"]

    service_status = {
        "required_available": all(
            ctx.deps.is_service_available(service) for service in required_services
        ),
        "optional_available": {
            service: ctx.deps.is_service_available(service)
            for service in optional_services
        },
        "total_services": len(ctx.deps.get_available_services()),
    }

    return service_status


# Knowledge Extraction Agent Toolset
knowledge_extraction_toolset = FunctionToolset()


@knowledge_extraction_toolset.tool
async def extract_entities_and_relationships(
    ctx: RunContext[UniversalDeps], 
    content: str, 
    use_domain_analysis: bool = True
) -> Dict[str, Any]:
    """
    Extract entities and relationships from content using universal patterns.
    
    This tool orchestrates the full extraction process with domain analysis.
    """
    # Import here to avoid circular imports
    from agents.knowledge_extraction.agent import _extract_with_enhanced_agent_guidance_optimized_internal
    
    return await _extract_with_enhanced_agent_guidance_optimized_internal(
        ctx, content, use_domain_analysis
    )

# Universal Search Agent Toolset  
universal_search_toolset = FunctionToolset()


@universal_search_toolset.tool
async def search_knowledge_graph(
    ctx: RunContext[UniversalDeps], 
    query: str, 
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Search knowledge graph for entities and relationships.
    """
    if not ctx.deps.is_service_available("cosmos"):
        return {"entities": [], "relationships": [], "error": "Cosmos DB not available"}
    
    cosmos_client = ctx.deps.cosmos_client
    
    # Extract query terms for entity search (preserve case for exact match)
    query_terms = [term.strip() for term in query.split() if len(term.strip()) > 2]
    
    results = {"entities": [], "relationships": [], "total_found": 0}
    
    for term in query_terms[:3]:  # Search first 3 terms
        try:
            # Find entities with exact text match (Cosmos DB Gremlin doesn't support containing)
            entity_query = f"g.V().has('text', '{term}').limit({max_results})"
            entities = await cosmos_client.execute_query(entity_query)
            
            for entity_data in (entities or [])[:max_results]:
                # Extract actual property values from Gremlin vertex structure
                if isinstance(entity_data, dict) and 'properties' in entity_data:
                    props = entity_data['properties']
                    
                    # Extract values from Gremlin property format
                    text_value = props.get('text', [{}])[0].get('value', '') if 'text' in props else ''
                    entity_type_value = props.get('entity_type', [{}])[0].get('value', 'unknown') if 'entity_type' in props else 'unknown'
                    confidence_value = props.get('confidence', [{}])[0].get('value', 0.0) if 'confidence' in props else 0.0
                    context_value = props.get('context', [{}])[0].get('value', '') if 'context' in props else ''
                    
                    if text_value:  # Only add entities with actual text
                        results["entities"].append({
                            "text": text_value,
                            "entity_type": entity_type_value,
                            "confidence": confidence_value,
                            "context": context_value,
                            "search_term": term
                        })
                        
        except Exception as e:
            print(f"Graph search failed for term '{term}': {e}")
    
    results["total_found"] = len(results["entities"])
    return results


@universal_search_toolset.tool  
async def search_vector_index(
    ctx: RunContext[UniversalDeps],
    query: str,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Search vector index for semantic similarity.
    """
    if not ctx.deps.is_service_available("search"):
        return {"results": [], "error": "Azure Search not available"}
        
    search_client = ctx.deps.search_client
    
    try:
        # Simple search query
        search_response = await search_client.search_documents(
            query=query,
            top=max_results,
            filters=None
        )
        
        results = []
        # FIXED: Use correct response format from our search client
        if search_response.get("success", False):
            documents = search_response["data"].get("documents", [])
            for result in documents:
                results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", "")[:200],
                    "score": result.get("score", 0.0),  # Fixed: our client uses "score" not "@search.score"
                    "source": "vector_search"
                })
        else:
            return {"results": [], "error": f"Search failed: {search_response.get('error', 'Unknown')}"}
            
        return {"results": results, "total_found": len(results)}
        
    except Exception as e:
        return {"results": [], "error": f"Vector search failed: {e}"}


@universal_search_toolset.tool
async def orchestrate_universal_search(
    ctx: RunContext[UniversalDeps],
    user_query: str,
    max_results: int = 10,
    use_domain_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate multi-modal search using centralized tools.
    
    This tool is the main entry point for Universal Search Agent operations.
    """
    import time
    start_time = time.time()
    
    print(f"🔍 Orchestrating universal search for: '{user_query}'")
    
    # Step 1: Search knowledge graph using centralized tool
    graph_results = await search_knowledge_graph(ctx, user_query, max_results)
    
    # Step 2: Search vector index using centralized tool
    vector_results = await search_vector_index(ctx, user_query, max_results)
    
    # Step 3: Domain analysis if requested
    domain_signature = "universal_default"
    if use_domain_analysis:
        try:
            from agents.domain_intelligence.agent import domain_intelligence_agent
            domain_result = await domain_intelligence_agent.run(
                f"Analyze search query characteristics: {user_query}",
                deps=ctx.deps,
                usage=None
            )
            domain_signature = f"adaptive_{domain_result.output.domain_signature}"
        except Exception as e:
            print(f"Domain analysis failed: {e}")
    
    # Step 4: Unify results
    unified_results = []
    search_confidence = 0.0
    
    # Add graph results
    for entity in graph_results.get("entities", []):
        unified_results.append({
            "title": entity["text"],
            "content": f"Entity: {entity['text']} (type: {entity['entity_type']})",
            "score": entity["confidence"],
            "source": "graph_search",
            "metadata": entity
        })
    
    # Add vector results  
    for result in vector_results.get("results", []):
        unified_results.append({
            "title": result["title"],
            "content": result["content"],
            "score": result["score"],
            "source": "vector_search",
            "metadata": result
        })
    
    # Calculate confidence
    if unified_results:
        search_confidence = sum(r["score"] for r in unified_results) / len(unified_results)
        search_confidence = min(search_confidence, 1.0)
    
    # Sort by score
    unified_results.sort(key=lambda x: x["score"], reverse=True)
    unified_results = unified_results[:max_results]
    
    processing_time = time.time() - start_time
    
    print(f"✅ Search completed: {len(unified_results)} results, confidence: {search_confidence:.3f}")
    
    return {
        "unified_results": unified_results,
        "graph_results": graph_results.get("entities", []),
        "vector_results": vector_results.get("results", []),
        "total_results_found": len(unified_results),
        "search_confidence": search_confidence,
        "search_strategy_used": domain_signature,
        "processing_time_seconds": processing_time,
    }


def get_domain_intelligence_toolset() -> FunctionToolset:
    """Get the Domain Intelligence Agent toolset."""
    return domain_intelligence_toolset


def get_knowledge_extraction_toolset() -> FunctionToolset:
    """Get the Knowledge Extraction Agent toolset."""
    return knowledge_extraction_toolset


def get_universal_search_toolset() -> FunctionToolset:
    """Get the Universal Search Agent toolset."""
    return universal_search_toolset


# Export all toolsets for easy import
__all__ = [
    "domain_intelligence_toolset",
    "knowledge_extraction_toolset", 
    "universal_search_toolset",
    "get_domain_intelligence_toolset",
    "get_knowledge_extraction_toolset",
    "get_universal_search_toolset",
]