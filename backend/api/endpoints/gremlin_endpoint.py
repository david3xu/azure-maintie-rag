"""
Gremlin Demo API Endpoints
Real-time Azure Cosmos DB Gremlin queries for supervisor demonstration
Following REST API patterns similar to Gremlin chaos engineering API
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

# Azure service components
from services.graph_service import GraphService
from api.dependencies import get_azure_services

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/gremlin", tags=["gremlin-demo"])


class GremlinQueryRequest(BaseModel):
    """Gremlin query request"""
    query: str = Field(..., description="Gremlin query to execute")
    description: Optional[str] = Field(None, description="Description of the query")


class GremlinQueryResponse(BaseModel):
    """Gremlin query response"""
    success: bool
    query: str
    description: Optional[str]
    results: List[Any]
    execution_time_ms: float
    results_count: int
    timestamp: str


class GraphStatsResponse(BaseModel):
    """Graph statistics response"""
    success: bool
    vertices: int
    edges: int
    connectivity_ratio: float
    entity_types: Dict[str, int]
    relationship_types: Dict[str, int]
    execution_time_ms: float


@router.get("/graph/stats", response_model=GraphStatsResponse)
def get_graph_statistics(
    azure_services: AzureServicesManager = Depends(get_azure_services)
) -> Dict[str, Any]:
    """
    Get real-time graph statistics using Gremlin queries
    
    Executes multiple Gremlin queries to get comprehensive graph statistics
    """
    logger.info("Executing Gremlin queries for graph statistics")
    
    try:
        start_time = time.time()
        
        # Get Cosmos Gremlin client
        cosmos_client = azure_services.get_service('cosmos')
        if not cosmos_client:
            raise HTTPException(status_code=500, detail="Cosmos DB service not available")
        
        # Execute basic count queries
        vertex_count = cosmos_client.gremlin_client.submit('g.V().count()').all().result()[0]
        edge_count = cosmos_client.gremlin_client.submit('g.E().count()').all().result()[0]
        
        # Get entity type distribution
        entity_type_query = "g.V().groupCount().by('entity_type')"
        entity_type_result = cosmos_client.gremlin_client.submit(entity_type_query).all().result()
        entity_types = entity_type_result[0] if entity_type_result else {}
        
        # Get relationship type distribution  
        relationship_query = "g.E().groupCount().by(label())"
        relationship_result = cosmos_client.gremlin_client.submit(relationship_query).all().result()
        relationship_types = relationship_result[0] if relationship_result else {}
        
        # Calculate connectivity
        connectivity_ratio = edge_count / vertex_count if vertex_count > 0 else 0
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "success": True,
            "vertices": vertex_count,
            "edges": edge_count, 
            "connectivity_ratio": connectivity_ratio,
            "entity_types": dict(sorted(entity_types.items(), key=lambda x: x[1], reverse=True)),
            "relationship_types": dict(sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)),
            "execution_time_ms": execution_time
        }
        
    except Exception as e:
        logger.error(f"Graph statistics query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gremlin query failed: {str(e)}")


@router.post("/query/execute", response_model=GremlinQueryResponse)
def execute_gremlin_query(
    request: GremlinQueryRequest,
    azure_services: AzureServicesManager = Depends(get_azure_services)
) -> Dict[str, Any]:
    """
    Execute a custom Gremlin query for demonstration
    
    Allows supervisor to see real-time Gremlin query execution
    """
    logger.info(f"Executing custom Gremlin query: {request.query[:100]}...")
    
    try:
        start_time = time.time()
        
        # Get Cosmos Gremlin client
        cosmos_client = azure_services.get_service('cosmos')
        if not cosmos_client:
            raise HTTPException(status_code=500, detail="Cosmos DB service not available")
        
        # Execute the query
        results = cosmos_client.gremlin_client.submit(request.query).all().result()
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "success": True,
            "query": request.query,
            "description": request.description,
            "results": results[:50],  # Limit results for display
            "execution_time_ms": execution_time,
            "results_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Gremlin query execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gremlin query failed: {str(e)}")


@router.get("/traversal/equipment-to-actions")
def get_equipment_to_actions_traversal(
    limit: int = Query(default=10, description="Limit number of results"),
    azure_services: AzureServicesManager = Depends(get_azure_services)
) -> Dict[str, Any]:
    """
    Demonstrate equipment to actions traversal using real Gremlin queries
    
    Shows multi-hop reasoning: equipment → component → action
    """
    logger.info("Executing equipment to actions traversal")
    
    try:
        start_time = time.time()
        
        # Get Cosmos Gremlin client
        cosmos_client = azure_services.get_service('cosmos')
        if not cosmos_client:
            raise HTTPException(status_code=500, detail="Cosmos DB service not available")
        
        # Multi-hop traversal query
        traversal_query = f'''
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
        
        results = cosmos_client.gremlin_client.submit(traversal_query).all().result()
        
        # Format results for demo
        workflows = []
        for result in results:
            workflows.append({
                "equipment": result.get('equipment', 'N/A'),
                "component": result.get('component', 'N/A'),
                "action": result.get('action', 'N/A'),
                "workflow_chain": f"{result.get('equipment', 'N/A')} → {result.get('component', 'N/A')} → {result.get('action', 'N/A')}"
            })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "query_type": "Multi-hop traversal",
            "gremlin_query": traversal_query.strip(),
            "workflows_discovered": workflows,
            "workflows_count": len(workflows),
            "execution_time_ms": execution_time,
            "demo_insight": "Shows intelligent pathfinding through knowledge graph"
        }
        
    except Exception as e:
        logger.error(f"Equipment to actions traversal failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Traversal query failed: {str(e)}")


@router.get("/analysis/top-connected-entities") 
def get_top_connected_entities(
    limit: int = Query(default=10, description="Number of top entities to return"),
    azure_services: AzureServicesManager = Depends(get_azure_services)
) -> Dict[str, Any]:
    """
    Find most connected entities using Gremlin degree centrality
    
    Shows graph analytics using real Gremlin queries
    """
    logger.info("Finding top connected entities")
    
    try:
        start_time = time.time()
        
        # Get Cosmos Gremlin client
        cosmos_client = azure_services.get_service('cosmos')
        if not cosmos_client:
            raise HTTPException(status_code=500, detail="Cosmos DB service not available")
        
        # Degree centrality query
        centrality_query = f'''
        g.V()
            .project('entity', 'entity_type', 'in_degree', 'out_degree', 'total_degree')
            .by('text')
            .by('entity_type')
            .by(inE().count())
            .by(outE().count())
            .by(bothE().count())
            .order()
            .by(select('total_degree'), Order.desc)
            .limit({limit})
        '''
        
        results = cosmos_client.gremlin_client.submit(centrality_query).all().result()
        
        # Format results
        top_entities = []
        for result in results:
            top_entities.append({
                "entity": result.get('entity', 'N/A'),
                "entity_type": result.get('entity_type', 'N/A'),
                "in_degree": result.get('in_degree', 0),
                "out_degree": result.get('out_degree', 0),
                "total_degree": result.get('total_degree', 0)
            })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "query_type": "Degree centrality analysis",
            "gremlin_query": centrality_query.strip(),
            "top_connected_entities": top_entities,
            "execution_time_ms": execution_time,
            "demo_insight": "Identifies most important entities in maintenance knowledge graph"
        }
        
    except Exception as e:
        logger.error(f"Top connected entities query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Centrality query failed: {str(e)}")


@router.get("/search/entity-neighborhood")
def search_entity_neighborhood(
    entity_text: str = Query(..., description="Entity text to search for"),
    hops: int = Query(default=2, description="Number of hops for neighborhood"),
    azure_services: AzureServicesManager = Depends(get_azure_services)
) -> Dict[str, Any]:
    """
    Search entity neighborhood using Gremlin traversal
    
    Shows semantic search with graph expansion
    """
    logger.info(f"Searching neighborhood for entity: {entity_text}")
    
    try:
        start_time = time.time()
        
        # Get Cosmos Gremlin client
        cosmos_client = azure_services.get_service('cosmos')
        if not cosmos_client:
            raise HTTPException(status_code=500, detail="Cosmos DB service not available")
        
        # Neighborhood search query
        neighborhood_query = f'''
        g.V().has('text', containing('{entity_text.lower()}'))
            .limit(1)
            .as('center')
            .repeat(both().simplePath())
            .times({hops})
            .as('neighbor')
            .select('center', 'neighbor')
            .by(valueMap())
            .limit(20)
        '''
        
        results = cosmos_client.gremlin_client.submit(neighborhood_query).all().result()
        
        # Format results
        center_entity = None
        neighbors = []
        
        for result in results:
            center_data = result.get('center', {})
            neighbor_data = result.get('neighbor', {})
            
            if not center_entity and center_data:
                center_entity = {
                    "text": center_data.get('text', ['N/A'])[0] if center_data.get('text') else 'N/A',
                    "entity_type": center_data.get('entity_type', ['N/A'])[0] if center_data.get('entity_type') else 'N/A'
                }
            
            if neighbor_data:
                neighbors.append({
                    "text": neighbor_data.get('text', ['N/A'])[0] if neighbor_data.get('text') else 'N/A',
                    "entity_type": neighbor_data.get('entity_type', ['N/A'])[0] if neighbor_data.get('entity_type') else 'N/A'
                })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "query_type": "Entity neighborhood search",
            "gremlin_query": neighborhood_query.strip(),
            "search_entity": entity_text,
            "center_entity": center_entity,
            "neighbors": neighbors[:10],  # Limit for display
            "neighbors_found": len(neighbors),
            "execution_time_ms": execution_time,
            "demo_insight": f"Shows {hops}-hop neighborhood expansion from '{entity_text}'"
        }
        
    except Exception as e:
        logger.error(f"Entity neighborhood search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Neighborhood search failed: {str(e)}")


@router.get("/demo/predefined-queries")
def get_predefined_demo_queries() -> Dict[str, Any]:
    """
    Get predefined Gremlin queries for supervisor demonstration
    
    Provides ready-to-use queries that showcase different graph capabilities
    """
    logger.info("Getting predefined demo queries")
    
    return {
        "success": True,
        "predefined_queries": {
            "basic_statistics": {
                "vertices_count": "g.V().count()",
                "edges_count": "g.E().count()",
                "entity_types": "g.V().groupCount().by('entity_type')",
                "relationship_types": "g.E().groupCount().by(label())"
            },
            "graph_traversals": {
                "equipment_components": "g.V().has('entity_type', 'equipment').limit(3).out().has('entity_type', 'component').values('text')",
                "issue_actions": "g.V().has('entity_type', 'issue').limit(5).out().has('entity_type', 'action').values('text')",
                "multi_hop_paths": "g.V().has('entity_type', 'equipment').limit(2).repeat(out().simplePath()).times(3).has('entity_type', 'action').path().by('text')"
            },
            "graph_analytics": {
                "high_degree_entities": "g.V().project('entity', 'degree').by('text').by(bothE().count()).order().by(select('degree'), Order.desc).limit(10)",
                "entity_neighborhoods": "g.V().has('text', containing('air')).both().limit(10).valueMap()",
                "relationship_patterns": "g.E().sample(10).project('source', 'relationship', 'target').by(outV().values('text')).by(label()).by(inV().values('text'))"
            }
        },
        "demo_workflow": [
            "Start with basic statistics to show graph scale",
            "Use traversals to demonstrate multi-hop reasoning", 
            "Apply analytics to show graph intelligence",
            "Try custom queries for specific use cases"
        ],
        "supervisor_talking_points": [
            "Real Gremlin queries against Azure Cosmos DB",
            "Sub-second query performance at scale",
            "Multi-hop reasoning with complex traversals",
            "Graph analytics for maintenance intelligence"
        ]
    }