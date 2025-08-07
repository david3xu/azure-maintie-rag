"""
Knowledge Extraction Agent - Integrated with Real Azure Infrastructure
======================================================================

This implementation integrates with real Azure services:
- Azure Cosmos DB Gremlin for knowledge graph storage
- Azure Blob Storage for document and chunk management
- Real Azure OpenAI for entity and relationship extraction
- Automated knowledge graph population
"""

import os
import time
import logging
from typing import Dict, List, Any
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load real Azure environment variables
load_dotenv("/workspace/azure-maintie-rag/.env")

# Import real infrastructure (Phase 2 Enhanced)
from infrastructure.azure_cosmos import SimpleCosmosGremlinClient
from infrastructure.azure_storage import SimpleStorageClient
from infrastructure.prompt_workflows.universal_prompt_generator import UniversalPromptGenerator
from infrastructure.azure_ml import GNNInferenceClient
from infrastructure.azure_monitoring import AppInsightsClient

logger = logging.getLogger(__name__)

# Enhanced dependencies with Phase 2 Azure infrastructure
class ExtractionDeps(BaseModel):
    """Dependencies for knowledge extraction with Phase 2 Azure services"""
    confidence_threshold: float = 0.8
    max_entities_per_chunk: int = 15
    max_relationships_per_chunk: int = 10
    enable_relationships: bool = True
    enable_graph_storage: bool = True
    enable_document_chunking: bool = True
    enable_gnn_training: bool = True    # Phase 2: GNN model training
    enable_monitoring: bool = True       # Phase 2: Performance monitoring
    
    # Phase 1 infrastructure clients
    cosmos_client: Any = None
    storage_client: Any = None
    prompt_generator: Any = None
    
    # Phase 2 infrastructure clients
    gnn_client: Any = None        # GNN training and inference
    monitoring_client: Any = None  # Performance tracking
    
    class Config:
        arbitrary_types_allowed = True

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

# Create REAL Azure OpenAI agent using proper PydanticAI configuration
# Following the same pattern as domain intelligence agent
azure_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY')
)

model = OpenAIModel(
    os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o'),
    provider=OpenAIProvider(openai_client=azure_client)
)

agent = Agent(
    model,
    deps_type=ExtractionDeps,
    output_type=ExtractionResult,
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
    """Extract entities from text using REAL Azure OpenAI and domain-specific prompts"""
    
    entities = []
    
    try:
        # Initialize prompt generator if needed
        if not ctx.deps.prompt_generator:
            ctx.deps.prompt_generator = UniversalPromptGenerator()
        
        logger.info(f"Extracting entities from {len(text)} characters of text")
        
        # Generate domain-specific entity extraction prompt
        extraction_prompt = await ctx.deps.prompt_generator.generate_entity_extraction_prompt(
            text=text,
            confidence_threshold=ctx.deps.confidence_threshold,
            max_entities=ctx.deps.max_entities_per_chunk
        )
        
        # Use the agent's Azure OpenAI model for real entity extraction
        entity_response = await azure_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o'),
            messages=[
                {"role": "system", "content": "You are an expert entity extraction system. Extract entities exactly as requested in the prompt."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        # Parse the LLM response to extract entities
        response_text = entity_response.choices[0].message.content
        
        # Process LLM response (this would typically parse JSON or structured output)
        import json
        try:
            # Assuming the LLM returns JSON format
            entity_data = json.loads(response_text)
            if isinstance(entity_data, list):
                for entity in entity_data:
                    if entity.get('confidence', 0) >= ctx.deps.confidence_threshold:
                        entities.append({
                            "text": entity.get('text', ''),
                            "type": entity.get('type', 'ENTITY'),
                            "confidence": entity.get('confidence', 0.0),
                            "start_pos": entity.get('start_pos', 0),
                            "end_pos": entity.get('end_pos', 0)
                        })
        except json.JSONDecodeError:
            # Fallback: parse text response for entities
            lines = response_text.strip().split('\n')
            for line in lines[:ctx.deps.max_entities_per_chunk]:
                if ':' in line:
                    entity_text = line.split(':')[0].strip()
                    if len(entity_text) > 2:
                        entities.append({
                            "text": entity_text,
                            "type": "EXTRACTED_ENTITY",
                            "confidence": 0.8,
                            "start_pos": 0,
                            "end_pos": len(entity_text)
                        })
        
        logger.info(f"Extracted {len(entities)} entities using Azure OpenAI")
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        # Fallback to basic heuristic extraction
        words = text.split()
        for i, word in enumerate(words[:ctx.deps.max_entities_per_chunk]):
            if word.istitle() and len(word) > 3:
                entities.append({
                    "text": word,
                    "type": "FALLBACK_ENTITY",
                    "confidence": 0.6,
                    "start_pos": i,
                    "end_pos": i + 1
                })
    
    return entities

@agent.tool
async def extract_relationships(
    ctx: RunContext[ExtractionDeps], 
    text: str, 
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract relationships between entities using REAL Azure OpenAI analysis"""
    
    if not ctx.deps.enable_relationships or len(entities) < 2:
        return []
    
    relationships = []
    
    try:
        # Initialize prompt generator if needed
        if not ctx.deps.prompt_generator:
            ctx.deps.prompt_generator = UniversalPromptGenerator()
        
        # Create entity list for relationship extraction
        entity_list = [e["text"] for e in entities]
        
        logger.info(f"Extracting relationships between {len(entity_list)} entities")
        
        # Generate domain-specific relationship extraction prompt
        relationship_prompt = await ctx.deps.prompt_generator.generate_relationship_extraction_prompt(
            text=text,
            entities=entity_list,
            max_relationships=ctx.deps.max_relationships_per_chunk
        )
        
        # Use Azure OpenAI for real relationship extraction
        relationship_response = await azure_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o'),
            messages=[
                {"role": "system", "content": "You are an expert relationship extraction system. Extract meaningful relationships between entities."},
                {"role": "user", "content": relationship_prompt}
            ],
            temperature=0.2  # Slightly higher for relationship creativity
        )
        
        # Parse the LLM response to extract relationships
        response_text = relationship_response.choices[0].message.content
        
        # Process LLM response for relationships
        import json
        try:
            # Assuming the LLM returns JSON format
            relationship_data = json.loads(response_text)
            if isinstance(relationship_data, list):
                for rel in relationship_data:
                    if rel.get('confidence', 0) >= ctx.deps.confidence_threshold:
                        relationships.append({
                            "subject": rel.get('subject', ''),
                            "predicate": rel.get('predicate', 'RELATES_TO'),
                            "object": rel.get('object', ''),
                            "confidence": rel.get('confidence', 0.0),
                            "context": rel.get('context', text[:100] + "...")
                        })
        except json.JSONDecodeError:
            # Fallback: parse text response for relationships
            lines = response_text.strip().split('\n')
            for line in lines[:ctx.deps.max_relationships_per_chunk]:
                if '->' in line or 'relates to' in line.lower():
                    parts = line.split('->')
                    if len(parts) == 2:
                        relationships.append({
                            "subject": parts[0].strip(),
                            "predicate": "RELATES_TO",
                            "object": parts[1].strip(),
                            "confidence": 0.7,
                            "context": text[:50] + "..."
                        })
        
        logger.info(f"Extracted {len(relationships)} relationships using Azure OpenAI")
        
    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        # Fallback: basic proximity-based relationships
        for i, entity1 in enumerate(entities[:3]):
            for entity2 in entities[i+1:i+2]:
                if entity1["text"] != entity2["text"]:
                    relationships.append({
                        "subject": entity1["text"],
                        "predicate": "CO_OCCURS_WITH",
                        "object": entity2["text"],
                        "confidence": 0.5,
                        "context": "Fallback proximity detection"
                    })
    
    return relationships

@agent.tool
async def store_in_knowledge_graph(
    ctx: RunContext[ExtractionDeps],
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    source_document: str = None
) -> Dict[str, Any]:
    """Store extracted knowledge in Azure Cosmos DB knowledge graph"""
    
    if not ctx.deps.enable_graph_storage:
        return {"stored": False, "reason": "Graph storage disabled"}
    
    stored_entities = 0
    stored_relationships = 0
    
    try:
        # Initialize Cosmos client if needed
        if not ctx.deps.cosmos_client:
            ctx.deps.cosmos_client = SimpleCosmosGremlinClient()
        
        logger.info(f"Storing {len(entities)} entities and {len(relationships)} relationships in knowledge graph")
        
        # Store entities as vertices
        for entity in entities:
            vertex_query = f"""
                g.V().has('entity_name', '{entity['text']}').fold().
                coalesce(unfold(), 
                    addV('entity').
                    property('entity_name', '{entity['text']}').
                    property('entity_type', '{entity['type']}').
                    property('confidence', {entity['confidence']}).
                    property('source_document', '{source_document or "unknown"}')
                )
            """
            
            await ctx.deps.cosmos_client.execute_query(vertex_query)
            stored_entities += 1
        
        # Store relationships as edges
        for rel in relationships:
            edge_query = f"""
                g.V().has('entity_name', '{rel['subject']}').as('a').
                V().has('entity_name', '{rel['object']}').as('b').
                coalesce(
                    __.select('a').outE('{rel['predicate']}').where(inV().as('b')),
                    addE('{rel['predicate']}').from('a').to('b').
                    property('confidence', {rel['confidence']}).
                    property('context', '{rel['context'][:100]}').
                    property('source_document', '{source_document or "unknown"}')
                )
            """
            
            await ctx.deps.cosmos_client.execute_query(edge_query)
            stored_relationships += 1
        
        logger.info(f"Successfully stored {stored_entities} entities and {stored_relationships} relationships")
        
        return {
            "stored": True,
            "entities_stored": stored_entities,
            "relationships_stored": stored_relationships,
            "total_stored": stored_entities + stored_relationships
        }
        
    except Exception as e:
        logger.error(f"Knowledge graph storage failed: {e}")
        return {
            "stored": False,
            "reason": f"Storage error: {str(e)}",
            "entities_attempted": len(entities),
            "relationships_attempted": len(relationships)
        }

@agent.tool
async def validate_extractions(
    ctx: RunContext[ExtractionDeps], 
    entities: List[Dict[str, Any]], 
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate extraction quality and provide metrics"""
    
    # Enhanced validation metrics
    if not entities and not relationships:
        return {
            "average_entity_confidence": 0.0,
            "average_relationship_confidence": 0.0,
            "overall_confidence": 0.0,
            "validation_passed": False,
            "warning": "No extractions found"
        }
    
    avg_entity_confidence = sum(e["confidence"] for e in entities) / max(len(entities), 1)
    avg_relationship_confidence = sum(r["confidence"] for r in relationships) / max(len(relationships), 1)
    
    # Weight entities and relationships differently
    if entities and relationships:
        overall_confidence = (avg_entity_confidence * 0.6 + avg_relationship_confidence * 0.4)
    elif entities:
        overall_confidence = avg_entity_confidence
    else:
        overall_confidence = avg_relationship_confidence
    
    return {
        "average_entity_confidence": avg_entity_confidence,
        "average_relationship_confidence": avg_relationship_confidence,
        "overall_confidence": overall_confidence,
        "validation_passed": overall_confidence >= ctx.deps.confidence_threshold,
        "entity_count": len(entities),
        "relationship_count": len(relationships)
    }

@agent.tool
async def train_gnn_model(
    ctx: RunContext[ExtractionDeps],
    extracted_entities: List[Dict[str, Any]],
    extracted_relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Train GNN model on extracted knowledge using Azure ML"""
    
    if not ctx.deps.enable_gnn_training:
        return {"trained": False, "reason": "GNN training disabled"}
    
    try:
        # Initialize GNN client if needed
        if not ctx.deps.gnn_client:
            ctx.deps.gnn_client = GNNInferenceClient()
        
        logger.info(f"Training GNN model on {len(extracted_entities)} entities and {len(extracted_relationships)} relationships")
        
        # Prepare training data for GNN
        training_data = {
            "nodes": [
                {
                    "id": idx,
                    "features": [
                        len(entity['text']),
                        entity['confidence'],
                        hash(entity['type']) % 1000  # Type encoding
                    ],
                    "label": entity['type']
                }
                for idx, entity in enumerate(extracted_entities)
            ],
            "edges": [
                {
                    "source": hash(rel['subject']) % len(extracted_entities),
                    "target": hash(rel['object']) % len(extracted_entities),
                    "features": [
                        rel['confidence'],
                        hash(rel['predicate']) % 1000  # Relation type encoding
                    ]
                }
                for rel in extracted_relationships
            ]
        }
        
        # Submit training job to Azure ML
        training_result = await ctx.deps.gnn_client.train_model(
            model_name="knowledge-extraction-gnn",
            training_data=training_data,
            config={
                "epochs": 100,
                "learning_rate": 0.001,
                "hidden_dim": 128,
                "num_layers": 3
            }
        )
        
        logger.info(f"GNN training completed: {training_result}")
        
        return {
            "trained": True,
            "model_name": training_result.get('model_name', ''),
            "training_accuracy": training_result.get('accuracy', 0.0),
            "training_loss": training_result.get('loss', 0.0),
            "training_time": training_result.get('training_time', 0),
            "model_size": training_result.get('model_size_mb', 0)
        }
        
    except Exception as e:
        logger.error(f"GNN training failed: {e}")
        return {
            "trained": False,
            "reason": f"Training error: {str(e)}"
        }

@agent.tool
async def track_extraction_performance(
    ctx: RunContext[ExtractionDeps],
    extraction_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Track extraction performance using Azure Application Insights"""
    
    if not ctx.deps.enable_monitoring:
        return {"tracked": False, "reason": "Monitoring disabled"}
    
    try:
        # Initialize monitoring client if needed
        if not ctx.deps.monitoring_client:
            ctx.deps.monitoring_client = AppInsightsClient()
        
        # Track extraction performance metrics
        performance_data = {
            "extraction_type": "knowledge_extraction",
            "text_length": extraction_metrics.get('text_length', 0),
            "entities_extracted": extraction_metrics.get('entity_count', 0),
            "relationships_extracted": extraction_metrics.get('relationship_count', 0),
            "processing_time": extraction_metrics.get('processing_time', 0),
            "average_entity_confidence": extraction_metrics.get('avg_entity_confidence', 0.0),
            "average_relationship_confidence": extraction_metrics.get('avg_relationship_confidence', 0.0),
            "graph_storage_enabled": extraction_metrics.get('graph_storage_enabled', False),
            "gnn_training_enabled": extraction_metrics.get('gnn_training_enabled', False)
        }
        
        # Send metrics to Azure Application Insights
        tracking_result = await ctx.deps.monitoring_client.track_custom_event(
            event_name="knowledge_extraction_completed",
            properties=performance_data,
            measurements={
                "processing_time_ms": extraction_metrics.get('processing_time', 0) * 1000,
                "entities_per_second": extraction_metrics.get('entity_count', 0) / max(extraction_metrics.get('processing_time', 1), 0.001),
                "relationships_per_second": extraction_metrics.get('relationship_count', 0) / max(extraction_metrics.get('processing_time', 1), 0.001),
                "overall_confidence": extraction_metrics.get('overall_confidence', 0.0)
            }
        )
        
        logger.info(f"Extraction performance tracked: {performance_data}")
        
        return {
            "tracked": True,
            "event_id": tracking_result.get('event_id', ''),
            "metrics_sent": len(performance_data),
            "measurements_sent": 4
        }
        
    except Exception as e:
        logger.error(f"Performance tracking failed: {e}")
        return {
            "tracked": False,
            "reason": f"Tracking error: {str(e)}"
        }

# Convenience function for knowledge extraction (Phase 2 Enhanced)
async def run_knowledge_extraction(
    text: str,
    confidence_threshold: float = 0.8,
    max_entities: int = 15,
    max_relationships: int = 10,
    enable_graph_storage: bool = False,
    enable_gnn_training: bool = False,    # Phase 2: GNN training
    enable_monitoring: bool = True,       # Phase 2: Performance tracking
    source_document: str = None
) -> ExtractionResult:
    """
    Run knowledge extraction with Phase 2 Azure infrastructure integration
    
    This function orchestrates the complete knowledge extraction pipeline:
    - Entity extraction using Azure OpenAI
    - Relationship extraction using Azure OpenAI 
    - Optional knowledge graph storage in Azure Cosmos DB
    - Optional GNN model training on extracted knowledge    # Phase 2
    - Performance tracking with Azure Application Insights  # Phase 2
    """
    start_time = time.time()
    
    # Create dependencies with Phase 2 infrastructure clients
    deps = ExtractionDeps(
        confidence_threshold=confidence_threshold,
        max_entities_per_chunk=max_entities,
        max_relationships_per_chunk=max_relationships,
        enable_graph_storage=enable_graph_storage,
        enable_gnn_training=enable_gnn_training,   # Phase 2
        enable_monitoring=enable_monitoring        # Phase 2
    )
    
    # Run the enhanced knowledge extraction agent
    result = await agent.run(
        f"Extract comprehensive knowledge from the provided text: {text[:100]}...",
        deps=deps
    )
    
    processing_time = time.time() - start_time
    
    # Extract data from agent result and enhance with timing
    extraction_data = result.data
    extraction_data.processing_time = processing_time
    
    # Phase 1: Store in knowledge graph if requested
    if enable_graph_storage and (extraction_data.entities or extraction_data.relationships):
        storage_result = await store_in_knowledge_graph(
            RunContext(deps=deps),
            [e.dict() for e in extraction_data.entities],
            [r.dict() for r in extraction_data.relationships],
            source_document
        )
        logger.info(f"Knowledge graph storage: {storage_result}")
    
    # Phase 2: Train GNN model if requested
    if enable_gnn_training and (extraction_data.entities or extraction_data.relationships):
        training_ctx = RunContext(deps=deps)
        training_result = await train_gnn_model(
            training_ctx,
            [e.dict() for e in extraction_data.entities],
            [r.dict() for r in extraction_data.relationships]
        )
        logger.info(f"GNN training: {training_result}")
    
    # Phase 2: Track performance if monitoring enabled
    if enable_monitoring:
        try:
            extraction_metrics = {
                "text_length": len(text),
                "entity_count": len(extraction_data.entities),
                "relationship_count": len(extraction_data.relationships),
                "processing_time": processing_time,
                "avg_entity_confidence": sum(e.confidence for e in extraction_data.entities) / max(len(extraction_data.entities), 1),
                "avg_relationship_confidence": sum(r.confidence for r in extraction_data.relationships) / max(len(extraction_data.relationships), 1),
                "overall_confidence": extraction_data.extraction_confidence,
                "graph_storage_enabled": enable_graph_storage,
                "gnn_training_enabled": enable_gnn_training
            }
            
            # Track performance using the agent's monitoring tool
            tracking_ctx = RunContext(deps=deps)
            await track_extraction_performance(tracking_ctx, extraction_metrics)
            
        except Exception as e:
            logger.warning(f"Performance tracking failed: {e}")
    
    logger.info(f"Phase 2 Knowledge extraction completed in {processing_time:.2f}s")
    
    return extraction_data

# Export enhanced interface
__all__ = [
    "agent", 
    "run_knowledge_extraction",
    "ExtractionDeps", 
    "ExtractionResult", 
    "ExtractedEntity", 
    "ExtractedRelationship"
]