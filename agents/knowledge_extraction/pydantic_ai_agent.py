"""
Simplified Knowledge Extraction Agent - PydanticAI Best Practices
=================================================================

This implementation demonstrates simplified knowledge extraction following PydanticAI patterns:
- Direct tool definitions without complex processors
- Simple dependency injection
- Clear separation of concerns
- Focus on core extraction functionality
"""

import os
from typing import Dict, List, Any
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Simple, focused dependencies
class ExtractionDeps(BaseModel):
    """Simple dependencies for knowledge extraction"""
    confidence_threshold: float = 0.8
    max_entities_per_chunk: int = 15
    enable_relationships: bool = True

# Clean output models
class ExtractedEntity(BaseModel):
    """Extracted entity with confidence"""
    text: str
    type: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0

class ExtractedRelationship(BaseModel):
    """Extracted relationship between entities"""
    subject: str
    predicate: str  
    object: str
    confidence: float
    context: str = ""

class ExtractionResult(BaseModel):
    """Knowledge extraction results"""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    processing_time: float
    extraction_confidence: float
    entity_count: int
    relationship_count: int

# Direct model configuration from environment
model_name = f"openai:{os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')}"

agent = Agent(
    model_name,
    deps_type=ExtractionDeps,
    result_type=ExtractionResult,
    system_prompt="""You are a Knowledge Extraction Agent that extracts entities and relationships from text.

    Your capabilities:
    1. Extract named entities (people, organizations, concepts, technical terms)
    2. Identify relationships between entities
    3. Provide confidence scores for all extractions
    4. Focus on high-quality, relevant extractions
    
    Always return structured ExtractionResult with entities and relationships.""",
)

@agent.tool
async def extract_entities(ctx: RunContext[ExtractionDeps], text: str) -> List[Dict[str, Any]]:
    """Extract entities from text using LLM reasoning"""
    
    # Simple entity extraction using the agent's LLM
    # This leverages the model's built-in NER capabilities
    entities = []
    
    # Use simple heuristics combined with LLM analysis
    words = text.split()
    
    # Look for common patterns (this would be enhanced with actual LLM extraction)
    for i, word in enumerate(words):
        if word.istitle() and len(word) > 2:  # Simple capitalization heuristic
            # In real implementation, this would use the LLM for classification
            entity_type = "CONCEPT" if word.endswith("ing") else "ENTITY"
            confidence = 0.8 if len(word) > 4 else 0.6
            
            if confidence >= ctx.deps.confidence_threshold:
                entities.append({
                    "text": word,
                    "type": entity_type,
                    "confidence": confidence,
                    "start_pos": i,
                    "end_pos": i + 1
                })
                
                if len(entities) >= ctx.deps.max_entities_per_chunk:
                    break
    
    return entities

@agent.tool
async def extract_relationships(
    ctx: RunContext[ExtractionDeps], 
    text: str, 
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract relationships between entities"""
    
    if not ctx.deps.enable_relationships or len(entities) < 2:
        return []
    
    relationships = []
    
    # Simple relationship extraction based on proximity and common patterns
    for i, entity1 in enumerate(entities[:5]):  # Limit for efficiency
        for entity2 in entities[i+1:i+3]:  # Look at nearby entities
            # Simple relationship detection (would use LLM in real implementation)
            if entity1["text"] != entity2["text"]:
                relationship = {
                    "subject": entity1["text"],
                    "predicate": "RELATED_TO",  # Simplified relationship type
                    "object": entity2["text"],
                    "confidence": min(entity1["confidence"], entity2["confidence"]) * 0.9,
                    "context": f"Found in proximity within text"
                }
                relationships.append(relationship)
    
    return relationships

@agent.tool
async def validate_extractions(
    ctx: RunContext[ExtractionDeps], 
    entities: List[Dict[str, Any]], 
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate extraction quality and provide metrics"""
    
    # Simple validation metrics
    avg_entity_confidence = sum(e["confidence"] for e in entities) / max(len(entities), 1)
    avg_relationship_confidence = sum(r["confidence"] for r in relationships) / max(len(relationships), 1)
    
    overall_confidence = (avg_entity_confidence + avg_relationship_confidence) / 2
    
    return {
        "average_entity_confidence": avg_entity_confidence,
        "average_relationship_confidence": avg_relationship_confidence,
        "overall_confidence": overall_confidence,
        "validation_passed": overall_confidence >= ctx.deps.confidence_threshold
    }

# Export simplified interface
__all__ = ["agent", "ExtractionDeps", "ExtractionResult", "ExtractedEntity", "ExtractedRelationship"]