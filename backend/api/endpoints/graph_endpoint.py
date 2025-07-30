"""
Knowledge Graph Demo API Endpoints
Specifically designed for supervisor demonstration of Azure knowledge graph capabilities
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Azure service components
# Using focused services instead of integrations
from api.dependencies import get_infrastructure_service
from services.infrastructure_service import InfrastructureService
from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient

# Import our knowledge graph operations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from services.graph_service import GraphService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/demo", tags=["knowledge-graph-demo"])


class KnowledgeGraphStatsResponse(BaseModel):
    """Knowledge graph statistics response"""
    success: bool
    graph_stats: Dict[str, Any]
    relationship_distribution: Dict[str, int]
    entity_types: Dict[str, int]
    connectivity_metrics: Dict[str, float]
    sample_relationships: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class GraphTraversalRequest(BaseModel):
    """Graph traversal request"""
    start_entity_type: str = Field(default="equipment", description="Starting entity type")
    target_entity_type: str = Field(default="action", description="Target entity type")
    max_hops: int = Field(default=3, description="Maximum hops for traversal")
    limit: int = Field(default=10, description="Maximum results to return")


class GraphTraversalResponse(BaseModel):
    """Graph traversal response"""
    success: bool
    traversal_query: str
    paths_found: int
    sample_paths: List[List[str]]
    performance: Dict[str, Any]


@router.get("/knowledge-graph/stats", response_model=KnowledgeGraphStatsResponse)
def get_knowledge_graph_stats(
    infrastructure: InfrastructureService = Depends(get_infrastructure_service)
) -> Dict[str, Any]:
    """
    Get comprehensive knowledge graph statistics for supervisor demo
    
    Demonstrates the scale and intelligence of our Azure Cosmos DB knowledge graph
    """
    logger.info("Getting knowledge graph statistics for supervisor demo")
    
    try:
        start_time = time.time()
        
        # Get Cosmos client from services manager
        cosmos_client = azure_services.get_service('cosmos')
        if not cosmos_client:
            raise HTTPException(status_code=500, detail="Cosmos DB service not available")
        
        # Get basic graph statistics using safe methods
        vertex_result = cosmos_client._execute_gremlin_query_safe('g.V().count()')
        vertex_count = vertex_result[0] if vertex_result and len(vertex_result) > 0 else 0
        
        edge_result = cosmos_client._execute_gremlin_query_safe('g.E().count()')
        edge_count = edge_result[0] if edge_result and len(edge_result) > 0 else 0
        
        # Get entity type distribution using safe method
        entity_type_query = "g.V().groupCount().by('entity_type')"
        entity_type_result = cosmos_client._execute_gremlin_query_safe(entity_type_query)
        entity_types = entity_type_result[0] if entity_type_result and len(entity_type_result) > 0 else {}
        
        # Get relationship type distribution using safe method
        relationship_query = "g.E().groupCount().by(label())"
        relationship_result = cosmos_client._execute_gremlin_query_safe(relationship_query)
        relationship_distribution = relationship_result[0] if relationship_result and len(relationship_result) > 0 else {}
        
        # Get sample relationships for demo using safe method
        sample_rel_query = 'g.E().limit(5).project("source", "target", "type").by(outV().values("text")).by(inV().values("text")).by(label())'
        sample_relationships = cosmos_client._execute_gremlin_query_safe(sample_rel_query)
        
        # Calculate connectivity metrics
        connectivity_ratio = edge_count / vertex_count if vertex_count > 0 else 0.0
        avg_degree = (edge_count * 2) / vertex_count if vertex_count > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "graph_stats": {
                "total_vertices": vertex_count,
                "total_edges": edge_count,
                "unique_entity_types": len(entity_types),
                "unique_relationship_types": len(relationship_distribution),
                "graph_density": connectivity_ratio,
                "is_highly_connected": connectivity_ratio > 10.0
            },
            "relationship_distribution": dict(sorted(
                relationship_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),  # Top 10 relationship types
            "entity_types": dict(sorted(
                entity_types.items(), 
                key=lambda x: x[1], 
                reverse=True
            )),
            "connectivity_metrics": {
                "connectivity_ratio": connectivity_ratio,
                "average_degree": avg_degree,
                "total_connections": edge_count
            },
            "sample_relationships": [
                {
                    "source": rel.get('source', 'N/A'),
                    "target": rel.get('target', 'N/A'),
                    "relationship_type": rel.get('type', 'N/A'),
                    "confidence": rel.get('properties', {}).get('confidence', [1.0])[0] if rel.get('properties', {}).get('confidence') else 1.0
                }
                for rel in sample_relationships
            ],
            "performance_metrics": {
                "query_time_seconds": processing_time,
                "queries_executed": 4,
                "azure_service": "Azure Cosmos DB Gremlin API"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get knowledge graph stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/traverse", response_model=GraphTraversalResponse)
def demonstrate_graph_traversal(
    request: GraphTraversalRequest,
    infrastructure: InfrastructureService = Depends(get_infrastructure_service)
) -> Dict[str, Any]:
    """
    Demonstrate multi-hop graph traversal for supervisor demo
    
    Shows intelligent pathfinding through the knowledge graph
    """
    logger.info(f"Demonstrating graph traversal: {request.start_entity_type} -> {request.target_entity_type}")
    
    try:
        start_time = time.time()
        
        # Initialize Gremlin client
        cosmos_client = AzureCosmosGremlinClient()
        cosmos_client._initialize_client()
        
        # Build traversal query
        traversal_query = f'''
        g.V().has('entity_type', '{request.start_entity_type}')
            .limit(3)
            .repeat(out().simplePath())
            .times({request.max_hops})
            .has('entity_type', '{request.target_entity_type}')
            .limit({request.limit})
            .path()
            .by('text')
        '''
        
        # Execute traversal
        traversal_results = cosmos_client.gremlin_client.submit(traversal_query).all().result()
        
        # Format paths for demo
        sample_paths = []
        for path in traversal_results[:5]:  # Show top 5 paths
            if isinstance(path, list):
                sample_paths.append(path)
            else:
                sample_paths.append([str(path)])
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "traversal_query": traversal_query.strip(),
            "paths_found": len(traversal_results),
            "sample_paths": sample_paths,
            "performance": {
                "execution_time_seconds": processing_time,
                "start_entity_type": request.start_entity_type,
                "target_entity_type": request.target_entity_type,
                "max_hops": request.max_hops,
                "azure_service": "Azure Cosmos DB Gremlin API"
            }
        }
        
    except Exception as e:
        logger.error(f"Graph traversal failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/maintenance-workflows")
def get_maintenance_workflows(
    limit: int = 10,
    infrastructure: InfrastructureService = Depends(get_infrastructure_service)
) -> Dict[str, Any]:
    """
    Demonstrate maintenance workflow discovery for supervisor demo
    
    Shows how the knowledge graph discovers equipment-component-action workflows
    """
    logger.info("Discovering maintenance workflows for supervisor demo")
    
    try:
        start_time = time.time()
        
        # Initialize Gremlin client
        cosmos_client = AzureCosmosGremlinClient()
        cosmos_client._initialize_client()
        
        # Equipment -> Component -> Action workflow discovery
        workflow_query = f'''
        g.V().has('entity_type', 'equipment')
            .limit(5)
            .as('equipment')
            .out()
            .has('entity_type', 'component')
            .as('component')
            .out()
            .has('entity_type', 'action')
            .as('action')
            .select('equipment', 'component', 'action')
            .by('text')
            .limit({limit})
        '''
        
        workflow_results = cosmos_client.gremlin_client.submit(workflow_query).all().result()
        
        # Format workflows for demo
        workflows = []
        for result in workflow_results:
            workflows.append({
                "equipment": result.get('equipment', 'N/A'),
                "component": result.get('component', 'N/A'),
                "action": result.get('action', 'N/A'),
                "workflow_chain": f"{result.get('equipment', 'N/A')} → {result.get('component', 'N/A')} → {result.get('action', 'N/A')}"
            })
        
        # Issue -> Action troubleshooting discovery
        troubleshooting_query = f'''
        g.V().has('entity_type', 'issue')
            .limit(3)
            .as('issue')
            .out()
            .has('entity_type', 'action')
            .as('action')
            .select('issue', 'action')
            .by('text')
            .limit({limit})
        '''
        
        troubleshooting_results = cosmos_client.gremlin_client.submit(troubleshooting_query).all().result()
        
        troubleshooting_workflows = []
        for result in troubleshooting_results:
            troubleshooting_workflows.append({
                "issue": result.get('issue', 'N/A'),
                "action": result.get('action', 'N/A'),
                "troubleshooting_chain": f"{result.get('issue', 'N/A')} → {result.get('action', 'N/A')}"
            })
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "maintenance_workflows": {
                "equipment_component_action_chains": workflows,
                "issue_action_troubleshooting": troubleshooting_workflows,
                "total_workflows_found": len(workflows) + len(troubleshooting_workflows)
            },
            "discovery_intelligence": {
                "preventive_maintenance_chains": len(workflows),
                "troubleshooting_workflows": len(troubleshooting_workflows),
                "graph_intelligence": "Multi-hop relationship discovery"
            },
            "performance": {
                "discovery_time_seconds": processing_time,
                "queries_executed": 2,
                "azure_service": "Azure Cosmos DB Gremlin API"
            }
        }
        
    except Exception as e:
        logger.error(f"Maintenance workflow discovery failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-flow/summary")
async def get_data_flow_summary() -> Dict[str, Any]:
    """
    Get complete data flow summary for supervisor demo
    
    Shows the entire pipeline from raw text to intelligent API endpoints
    """
    logger.info("Getting complete data flow summary for supervisor demo")
    
    try:
        # Dynamically find the most recent data files
        base_path = Path(__file__).parent.parent.parent
        
        # Find most recent extraction file
        extraction_outputs_dir = base_path / "data/extraction_outputs"
        extraction_file = None
        if extraction_outputs_dir.exists():
            extraction_files = list(extraction_outputs_dir.glob("*.json"))
            if extraction_files:
                extraction_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
        
        # Find most recent KG operations file
        kg_operations_dir = base_path / "data/kg_operations"
        kg_demo_file = None
        if kg_operations_dir.exists():
            kg_files = list(kg_operations_dir.glob("*.json"))
            if kg_files:
                kg_demo_file = max(kg_files, key=lambda x: x.stat().st_mtime)
        
        # Find most recent GNN metadata file
        gnn_training_dir = base_path / "data/gnn_training"
        gnn_metadata_file = None
        if gnn_training_dir.exists():
            gnn_files = list(gnn_training_dir.glob("gnn_metadata_*.json"))
            if gnn_files:
                gnn_metadata_file = max(gnn_files, key=lambda x: x.stat().st_mtime)
        
        # Get actual data counts from files
        pipeline_stages = {}
        
        # Stage 1: Raw text data
        raw_data_dir = base_path / "data/raw"
        raw_files_count = len(list(raw_data_dir.glob("*.md"))) if raw_data_dir.exists() else 0
        pipeline_stages["1_raw_text_data"] = {
            "description": "Raw data from source files",
            "source_files": raw_files_count,
            "data_type": "Raw text documents"
        }
        
        # Stage 2: LLM extraction (from actual extraction file)
        if extraction_file and extraction_file.exists():
            try:
                with open(extraction_file, 'r') as f:
                    extraction_data = json.load(f)
                    pipeline_stages["2_llm_extraction"] = {
                        "description": "Azure OpenAI knowledge extraction",
                        "service": "Azure OpenAI",
                        "entities_extracted": len(extraction_data.get("entities", [])),
                        "relationships_identified": len(extraction_data.get("relationships", [])),
                        "extraction_file": extraction_file.name
                    }
            except Exception:
                pipeline_stages["2_llm_extraction"] = {
                    "description": "Azure OpenAI knowledge extraction",
                    "status": "No extraction data available"
                }
        else:
            pipeline_stages["2_llm_extraction"] = {
                "description": "Azure OpenAI knowledge extraction",
                "status": "No extraction data available"
            }
        
        # Stage 3: Knowledge graph (from actual KG data)
        if kg_demo_file and kg_demo_file.exists():
            try:
                with open(kg_demo_file, 'r') as f:
                    kg_data = json.load(f)
                    graph_state = kg_data.get("graph_state", {})
                    pipeline_stages["3_knowledge_structure"] = {
                        "description": "Azure Cosmos DB knowledge graph",
                        "service": "Azure Cosmos DB Gremlin API",
                        "vertices_loaded": graph_state.get("vertices", 0),
                        "edges_loaded": graph_state.get("edges", 0),
                        "kg_file": kg_demo_file.name
                    }
            except Exception:
                pipeline_stages["3_knowledge_structure"] = {
                    "description": "Azure Cosmos DB knowledge graph",
                    "status": "No knowledge graph data available"
                }
        else:
            pipeline_stages["3_knowledge_structure"] = {
                "description": "Azure Cosmos DB knowledge graph",
                "status": "No knowledge graph data available"
            }
        
        # Stage 4: GNN training (from actual metadata)
        if gnn_metadata_file and gnn_metadata_file.exists():
            try:
                with open(gnn_metadata_file, 'r') as f:
                    gnn_data = json.load(f)
                    pipeline_stages["4_gnn_training"] = {
                        "description": "Graph Neural Network training",
                        "framework": "PyTorch Geometric",
                        "model_accuracy": gnn_data.get("test_accuracy", "Not available"),
                        "training_environment": "Azure ML",
                        "metadata_file": gnn_metadata_file.name
                    }
            except Exception:
                pipeline_stages["4_gnn_training"] = {
                    "description": "Graph Neural Network training",
                    "status": "No GNN training data available"
                }
        else:
            pipeline_stages["4_gnn_training"] = {
                "description": "Graph Neural Network training",
                "status": "No GNN training data available"
            }
        
        data_flow_summary = {
            "success": True,
            "pipeline_stages": pipeline_stages,
            "data_sources": {
                "raw_data_files": raw_files_count,
                "extraction_available": extraction_file is not None,
                "kg_data_available": kg_demo_file is not None,
                "gnn_metadata_available": gnn_metadata_file is not None
                }
            }
        
        return data_flow_summary
        
    except Exception as e:
        logger.error(f"Error getting data flow summary: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Unable to generate data flow summary from available data"
        }
