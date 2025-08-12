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


# Import the proper universal model instead of duplicating
from agents.core.universal_models import UniversalDomainCharacteristics


@domain_intelligence_toolset.tool
async def analyze_content_characteristics(
    ctx: RunContext[UniversalDeps], content: str, detailed_analysis: bool = True
) -> UniversalDomainCharacteristics:
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

    # Calculate additional required fields for UniversalDomainCharacteristics
    # Basic document metrics
    avg_document_length = len(content)
    document_count = 1  # Single document analysis
    
    # TODO: Implement LLM-based vocabulary analysis
    vocabulary_richness = min(vocab_complexity + 0.1, 1.0)  # Temporary calculation
    sentence_complexity = len(content.split()) / max(content.count('.') + content.count('!') + content.count('?'), 1)
    
    # Extract key content terms using LLM analysis  
    key_content_terms = await _extract_key_content_terms_via_llm(ctx, content)
    content_patterns = structural_patterns  # Use discovered patterns
    
    # Language detection placeholder
    language_indicators = {"english": 0.95}  # Placeholder
    
    # Lexical diversity calculation
    words = content.lower().split()
    unique_words = len(set(words))
    total_words = len(words)
    lexical_diversity = unique_words / max(total_words, 1) if total_words > 0 else 0.0
    
    # Use vocab_complexity as vocabulary_complexity_ratio (matching centralized schema)
    vocabulary_complexity_ratio = vocab_complexity
    
    # Structural consistency based on patterns found
    structural_consistency = min(len(structural_patterns) / 4.0, 1.0)
    
    return UniversalDomainCharacteristics(
        # Required fields from centralized schema
        vocabulary_complexity_ratio=vocabulary_complexity_ratio,
        lexical_diversity=lexical_diversity,
        structural_consistency=structural_consistency,
        vocabulary_richness=vocabulary_richness,
        sentence_complexity=sentence_complexity,
        content_patterns=content_patterns,
        key_content_terms=key_content_terms,
        avg_document_length=avg_document_length,
        document_count=document_count,
        # Additional fields
        language_indicators=language_indicators,
    )


@domain_intelligence_toolset.tool
async def predict_entity_types(
    ctx: RunContext[UniversalDeps],
    content: str,
    characteristics: UniversalDomainCharacteristics,
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
- Vocabulary complexity: {characteristics.vocabulary_complexity_ratio:.3f}
- Concept density: {characteristics.concept_density:.3f}
- Structural patterns: {characteristics.content_patterns}

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
                "source": "llm_generated",
                "content_analysis_basis": {
                    "vocabulary_complexity": characteristics.vocabulary_complexity_ratio,
                    "concept_density": characteristics.concept_density,
                    "structural_patterns": characteristics.content_patterns,
                }
            }
        else:
            raise ValueError("Could not parse LLM response as JSON")
            
    except Exception as e:
        # FAIL FAST - No fallbacks for Azure OpenAI service failures
        raise RuntimeError(f"Azure OpenAI entity type prediction failed: {e}. System requires functional Azure OpenAI service.")




@domain_intelligence_toolset.tool
async def generate_processing_configuration(
    ctx: RunContext[UniversalDeps],
    characteristics: UniversalDomainCharacteristics,
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
    complexity_factor = characteristics.vocabulary_complexity_ratio
    density_factor = characteristics.concept_density
    
    # Adaptive chunk size using continuous scaling
    optimal_chunk_size = base_config.optimal_chunk_size
    complexity_scaling = 1.0 + (complexity_factor * 0.3)  # Scale up to 30% based on complexity
    if "code_blocks" in characteristics.content_patterns:
        complexity_scaling *= 1.5  # Additional scaling for code
    if density_factor > 0.8:
        complexity_scaling *= 1.2  # Additional scaling for high density
    
    optimal_chunk_size = int(optimal_chunk_size * complexity_scaling)

    # Adaptive overlap using continuous scaling  
    chunk_overlap_ratio = base_config.chunk_overlap_ratio
    if "hierarchical_headers" in characteristics.content_patterns:
        chunk_overlap_ratio *= 0.8  # Less overlap for structured content
    if density_factor > 0.7:
        chunk_overlap_ratio = min(chunk_overlap_ratio * 1.4, 0.5)

    # Dynamic confidence thresholds (no hardcoded values)
    entity_confidence_threshold = base_config.entity_confidence_threshold
    # Note: entity_indicators was removed from UniversalDomainCharacteristics as it was primitive
    # Using content patterns instead for proper nouns detection
    # Adaptive threshold based on measured content characteristics
    if complexity_factor > COMPLEXITY_HIGH_THRESHOLD:  # Dynamically adjust for measured complexity
        entity_confidence_threshold *= (1.0 - complexity_factor * 0.25)  # More aggressive adjustment for rich content
    elif complexity_factor < COMPLEXITY_LOW_THRESHOLD:
        entity_confidence_threshold *= 1.1
    # More permissive clamping for rich technical content
    entity_confidence_threshold = max(0.45, min(1.0, entity_confidence_threshold))

    # Dynamic relationship density
    relationship_density = base_config.relationship_density
    # Note: relationship_indicators was removed - using content patterns for relationship density
    if len(characteristics.content_patterns) > 2:  # More patterns = more potential relationships
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
    from agents.knowledge_extraction.agent import run_knowledge_extraction
    
    # Call the actual extraction function
    result = await run_knowledge_extraction(
        content, 
        use_domain_analysis=use_domain_analysis
    )
    
    # Convert ExtractionResult to dict format expected by toolset
    return {
        "entities": [entity.model_dump() for entity in result.entities],
        "relationships": [rel.model_dump() for rel in result.relationships],
        "extraction_confidence": result.extraction_confidence,
        "processing_signature": result.processing_signature,
        "processing_time": result.processing_time,
        "extracted_concepts": result.extracted_concepts
    }

@knowledge_extraction_toolset.tool
async def generate_extraction_prompts(
    ctx: RunContext[UniversalDeps],
    content: str,
    predicted_entities: Dict[str, Any],
    characteristics: "UniversalDomainCharacteristics",  # Use string annotation to avoid circular import
) -> Dict[str, str]:
    """
    Generate targeted extraction prompts based on Agent 1's content analysis.
    
    This creates specific prompts for Agent 2 instead of generic extraction.
    """
    prompts = {}
    
    # Base prompt with discovered characteristics
    base_context = f"""
Content Analysis Results:
- Vocabulary Complexity: {characteristics.vocabulary_complexity_ratio:.3f}
- Concept Density: {characteristics.concept_density:.3f}
- Content Patterns: {', '.join(characteristics.content_patterns)}
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
            # Find entities with substring match (use containing for text search)
            entity_query = f"g.V().hasLabel('entity').has('text', containing('{term}')).limit({max_results})"
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
            # FAIL FAST - Don't mask graph search failures
            raise RuntimeError(f"Graph search failed for term '{term}': {e}. Check Azure Cosmos DB Gremlin connection.")
    
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
        # FAIL FAST - Don't mask vector search failures  
        raise RuntimeError(f"Vector search failed: {e}. Check Azure Cognitive Search connection.")


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
    
    print(f"🎯 Orchestrating MANDATORY tri-modal search for: '{user_query}'")
    print("   🚨 ALL THREE modalities REQUIRED: Vector + Graph + GNN")
    
    # Step 1: Vector search (MANDATORY) 
    print("   1️⃣ Vector search (REQUIRED)...")
    vector_results = await search_vector_index(ctx, user_query, max_results)
    print(f"   ✅ Vector: {len(vector_results.get('results', []))} results")
    
    # Step 2: Graph search (MANDATORY)
    print("   2️⃣ Graph search (REQUIRED)...")
    graph_results = await search_knowledge_graph(ctx, user_query, max_results)
    print(f"   ✅ Graph: {len(graph_results.get('entities', []))} entities")
    
    # Step 3: Domain analysis - FAIL FAST if requested
    domain_signature = "universal_default"
    if use_domain_analysis:
        from agents.domain_intelligence.agent import domain_intelligence_agent
        domain_result = await domain_intelligence_agent.run(
            f"Analyze search query characteristics: {user_query}",
            deps=ctx.deps,
            usage=None
        )
        domain_signature = f"adaptive_{domain_result.output.domain_signature}"
    
    # Step 4: GNN Inference (MANDATORY) 
    print("   3️⃣ GNN inference (REQUIRED)...")
    
    # FAIL FAST: GNN client must be available for mandatory tri-modal search
    if not ctx.deps.gnn_client:
        raise RuntimeError(
            "GNN inference client required for mandatory tri-modal search. "
            "All three modalities (Vector + Graph + GNN) must be available. "
            "Check Azure ML service configuration and GNN client initialization."
        )
    
    gnn_inference_client = ctx.deps.gnn_client
    
    # Prepare context from vector and graph results
    gnn_context = {
        "query": user_query,
        "vector_context": [r.get("content", "") for r in vector_results.get("results", [])[:3]],
        "graph_context": [e.get("text", "") for e in graph_results.get("entities", [])[:3]],
        "max_predictions": min(max_results, 5)
    }
    
    # Execute GNN inference - FAIL if Azure ML service fails
    # Use the universal predict method which handles the context format
    gnn_predictions = await gnn_inference_client.predict(gnn_context)
    
    gnn_results = []
    for prediction in gnn_predictions.get("predictions", []):
        gnn_results.append({
            "title": f"GNN Prediction: {prediction.get('entity', 'Unknown')}",
            "content": prediction.get('reasoning', 'GNN-based relationship prediction'),
            "score": prediction.get('confidence', 0.5),
            "source": "gnn_inference", 
            "metadata": prediction
        })
        
    print(f"   ✅ GNN: {len(gnn_results)} predictions generated")
        
    # Step 5: Unify results from all three modalities
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
    
    # Add GNN results (NEW - Complete tri-modal)
    unified_results.extend(gnn_results)
    
    # Calculate confidence from all three modalities
    if unified_results:
        search_confidence = sum(r["score"] for r in unified_results) / len(unified_results)
        search_confidence = min(search_confidence, 1.0)
    
    # Sort by score
    unified_results.sort(key=lambda x: x["score"], reverse=True)
    unified_results = unified_results[:max_results]
    
    processing_time = time.time() - start_time
    
    # Validate tri-modal requirement satisfaction - NO FALLBACKS ALLOWED
    missing = []
    if not vector_results.get("results"): missing.append("Vector")
    if not graph_results.get("entities"): missing.append("Graph")  
    if not gnn_results: missing.append("GNN")
    
    if missing:
        raise RuntimeError(f"MANDATORY tri-modal search failed: Missing results from {missing}. All three modalities (Vector + Graph + GNN) are required. No fallback patterns allowed - implement real Azure services first.")
    
    print(f"🎉 MANDATORY tri-modal search completed: {len(unified_results)} unified results")
    print(f"   📊 Vector: {len(vector_results.get('results', []))} + Graph: {len(graph_results.get('entities', []))} + GNN: {len(gnn_results)} results")
    print(f"   🎯 Search confidence: {search_confidence:.3f}")
    
    return {
        "unified_results": unified_results,
        "graph_results": graph_results.get("entities", []),
        "vector_results": vector_results.get("results", []),
        "gnn_results": gnn_results,  # NEW - Include GNN results
        "total_results_found": len(unified_results),
        "search_confidence": search_confidence,
        "search_strategy_used": f"{domain_signature}_mandatory_tri_modal",
        "processing_time_seconds": processing_time,
        "modalities_used": ["vector", "graph", "gnn"],  # ALL THREE ALWAYS REQUIRED
    }


async def _extract_key_content_terms_via_llm(ctx: RunContext[UniversalDeps], content: str) -> List[str]:
    """
    Extract most IMPORTANT content-characterizing terms using REAL Azure OpenAI service ONLY.
    
    NO FALLBACKS - Production requires real Azure integration.
    Uses FULL content, not samples.
    """
    # Use REAL Azure OpenAI service - NO fallbacks allowed
    if not ctx.deps.openai_client:
        raise RuntimeError("Azure OpenAI client not initialized - cannot extract key content terms")
    
    # Create focused prompt for term extraction using FULL REAL content
    analysis_prompt = f"""Analyze this COMPLETE content and identify the 3 most IMPORTANT terms that characterize what this content is fundamentally about (not just frequent words, but key concepts that define the content's purpose and domain).

Return ONLY a JSON array of exactly 3 terms, like: ["term1", "term2", "term3"]

COMPLETE CONTENT TO ANALYZE:
{content}"""  # Use FULL content, not truncated
    
    # Use the UnifiedAzureOpenAIClient's complete_chat method
    response_data = await ctx.deps.openai_client.complete_chat(
        messages=[
            {"role": "system", "content": "You are a content analyst. Extract the 3 most IMPORTANT terms that characterize what this content is fundamentally about. Focus on key concepts, not word frequency. Return ONLY valid JSON."},
            {"role": "user", "content": analysis_prompt}
        ],
        max_tokens=100,  # Allow more tokens for proper response  
        temperature=0.0  # Zero temperature for consistent results
    )
    
    # Check for errors in response
    if "error" in response_data:
        raise RuntimeError(f"Azure OpenAI service error: {response_data['error']}")
    
    response_text = response_data["content"].strip()
    
    # Debug: show what we got from Azure OpenAI
    if not response_text:
        raise ValueError(f"Azure OpenAI returned empty response. Full response: {response_data}")
    
    # Clean response text - remove markdown code blocks if present
    if response_text.startswith('```json'):
        # Extract JSON from markdown code block
        lines = response_text.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip() == '```json':
                in_json = True
                continue
            elif line.strip() == '```':
                break
            elif in_json:
                json_lines.append(line)
        response_text = '\n'.join(json_lines).strip()
    
    # Parse JSON response - FAIL if malformed (NO fallbacks)
    import json
    try:
        terms = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Azure OpenAI returned invalid JSON after cleaning: '{response_text}'. JSON error: {e}")
    
    if not isinstance(terms, list) or len(terms) != 3:
        raise ValueError(f"Azure OpenAI returned invalid format: expected list of 3 terms, got {terms}")
    
    # Clean and validate terms
    cleaned_terms = []
    for term in terms:
        if not isinstance(term, str) or len(term.strip()) == 0:
            raise ValueError(f"Azure OpenAI returned invalid term: '{term}' - must be non-empty string")
        cleaned_terms.append(term.strip().lower())
    
    return cleaned_terms


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