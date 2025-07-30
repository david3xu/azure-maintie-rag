#!/usr/bin/env python3
"""
GNN Enhanced Query API Endpoint
Provides enhanced query processing with GNN integration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

# Import GNN services from our architecture
from services.gnn_service import GNNService

router = APIRouter()

# Global GNN service instance
gnn_service = None

class GNNQueryRequest(BaseModel):
    query: str = Field(..., description="Query to process")
    entities: List[str] = Field(default=[], description="Entities to analyze")
    domain: str = Field(default="maintenance", description="Domain for analysis")
    max_hops: int = Field(3, description="Maximum hops for reasoning")
    include_embeddings: bool = Field(False, description="Include semantic embeddings in response")

class GNNQueryResponse(BaseModel):
    query: str
    domain: str
    entities_analyzed: int
    entities_found: int
    processing_time: float
    entity_coverage: float
    related_entities_count: int
    embeddings: Optional[Dict[str, Any]] = None
    related_entities: Optional[Dict[str, List[str]]] = None
    error: Optional[str] = None

def get_gnn_service() -> GNNService:
    """Get or initialize GNN service"""
    global gnn_service

    if gnn_service is None:
        gnn_service = GNNService()

    return gnn_service

@router.post("/api/v1/query/gnn-enhanced", response_model=GNNQueryResponse)
async def gnn_enhanced_query(request: GNNQueryRequest):
    """
    Enhanced query processing with GNN integration

    This endpoint provides:
    - Entity analysis using trained GNN model
    - Multi-hop entity relationship discovery
    - Semantic embedding generation
    - Graph-aware query processing
    """

    start_time = time.time()

    try:
        # Get GNN service
        service = get_gnn_service()

        # Use entities from request, or try to extract from query if empty
        entities_to_analyze = request.entities
        if not entities_to_analyze and request.query:
            # Simple entity extraction - in real implementation, use NLP
            entities_to_analyze = [word for word in request.query.split() if len(word) > 2]

        # Perform GNN analysis
        result = await service.analyze_query_entities(entities_to_analyze, request.domain)

        return GNNQueryResponse(
            query=request.query,
            domain=request.domain,
            entities_analyzed=result.get('entities_analyzed', 0),
            entities_found=result.get('entities_found', 0),
            processing_time=time.time() - start_time,
            entity_coverage=result.get('entity_coverage', 0.0),
            related_entities_count=result.get('total_related_entities', 0),
            embeddings=result.get('embeddings') if request.include_embeddings else None,
            related_entities=result.get('related_entities'),
            error=result.get('error')
        )

    except Exception as e:
        return GNNQueryResponse(
            query=request.query,
            domain=request.domain,
            entities_analyzed=0,
            entities_found=0,
            processing_time=time.time() - start_time,
            entity_coverage=0.0,
            related_entities_count=0,
            error=str(e)
        )

@router.get("/api/v1/gnn/status")
async def gnn_status():
    """Get GNN service status"""
    service = get_gnn_service()

    try:
        # Try to get a model to check availability
        model = await service.get_model("maintenance")
        model_available = model is not None
        
        return {
            "gnn_available": service is not None,
            "model_loaded": model_available,
            "service_initialized": True,
            "default_domain": "maintenance",
            "status": "operational" if model_available else "model_not_loaded"
        }
    except Exception as e:
        return {
            "gnn_available": False,
            "model_loaded": False,
            "service_initialized": False,
            "error": str(e),
            "status": "error"
        }

@router.post("/api/v1/gnn/analyze")
async def analyze_entities(entities: List[str], domain: str = "maintenance", include_embeddings: bool = False):
    """Analyze entities using GNN"""
    service = get_gnn_service()

    try:
        result = await service.analyze_query_entities(entities, domain)
        
        response = {
            "entities_analyzed": result.get('entities_analyzed', 0),
            "entities_found": result.get('entities_found', 0),
            "entity_coverage": result.get('entity_coverage', 0.0),
            "related_entities": result.get('related_entities', {}),
            "total_related": result.get('total_related_entities', 0),
            "domain": domain
        }
        
        if include_embeddings:
            response["embeddings"] = result.get('embeddings', {})
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity analysis failed: {str(e)}")

@router.post("/api/v1/gnn/related")
async def find_related_entities(entities: List[str], domain: str = "maintenance", hops: int = 2):
    """Find entities related to the given entities through graph traversal"""
    service = get_gnn_service()

    try:
        related = await service.inference_service.find_related_entities(entities, domain, hops)
        
        return {
            "input_entities": entities,
            "domain": domain,
            "hops": hops,
            "related_entities": related,
            "total_related": sum(len(rels) for rels in related.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Related entity search failed: {str(e)}")

@router.get("/api/v1/gnn/domains")
async def get_available_domains():
    """Get available GNN domains"""
    return {
        "domains": ["maintenance"],  # Can be expanded based on available models
        "default_domain": "maintenance",
        "description": "Available domains for GNN analysis"
    }
