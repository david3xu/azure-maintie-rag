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

# Import GNN integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.integrate_gnn_with_api import GNNEnhancedAPI

router = APIRouter()

# Global GNN service instance
gnn_api = None

class GNNQueryRequest(BaseModel):
    query: str = Field(..., description="Query to process")
    use_gnn: bool = Field(True, description="Whether to use GNN enhancement")
    max_hops: int = Field(3, description="Maximum hops for reasoning")
    include_embeddings: bool = Field(False, description="Include semantic embeddings in response")

class GNNQueryResponse(BaseModel):
    query: str
    enhanced: bool
    gnn_confidence: float
    processing_time: float
    entities_found: int
    reasoning_paths: int
    enhanced_query: Optional[Dict[str, Any]] = None
    reasoning_results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

def get_gnn_service() -> GNNEnhancedAPI:
    """Get or initialize GNN service"""
    global gnn_api

    if gnn_api is None:
        gnn_api = GNNEnhancedAPI()
        try:
            gnn_api.initialize_gnn()
        except Exception as e:
            print(f"❌ GNN initialization failed: {e}")
            gnn_api = None

    return gnn_api

@router.post("/api/v1/query/gnn-enhanced", response_model=GNNQueryResponse)
async def gnn_enhanced_query(request: GNNQueryRequest):
    """
    Enhanced query processing with GNN integration

    This endpoint provides:
    - Entity classification using trained GNN model
    - Enhanced multi-hop reasoning with confidence scores
    - Semantic embedding generation
    - Graph-aware query processing
    """

    start_time = time.time()

    try:
        # Get GNN service
        gnn_service = get_gnn_service()

        if not gnn_service or not gnn_service.initialized:
            return GNNQueryResponse(
                query=request.query,
                enhanced=False,
                gnn_confidence=0.0,
                processing_time=time.time() - start_time,
                entities_found=0,
                reasoning_paths=0,
                error="GNN service not available"
            )

        # Process query with GNN enhancement
        result = gnn_service.enhanced_query_processing(
            request.query,
            use_gnn=request.use_gnn
        )

        # Count entities and reasoning paths
        entities_found = len(result.get('enhanced_query', {}).get('extracted_entities', []))
        reasoning_paths = sum(
            len(reasoning.get('paths', []))
            for reasoning in result.get('reasoning_results', [])
        )

        return GNNQueryResponse(
            query=request.query,
            enhanced=result.get('enhanced', False),
            gnn_confidence=result.get('gnn_confidence', 0.0),
            processing_time=time.time() - start_time,
            entities_found=entities_found,
            reasoning_paths=reasoning_paths,
            enhanced_query=result.get('enhanced_query'),
            reasoning_results=result.get('reasoning_results')
        )

    except Exception as e:
        return GNNQueryResponse(
            query=request.query,
            enhanced=False,
            gnn_confidence=0.0,
            processing_time=time.time() - start_time,
            entities_found=0,
            reasoning_paths=0,
            error=str(e)
        )

@router.get("/api/v1/gnn/status")
async def gnn_status():
    """Get GNN service status"""
    gnn_service = get_gnn_service()

    return {
        "gnn_available": gnn_service is not None and gnn_service.initialized,
        "model_loaded": gnn_service.model is not None if gnn_service else False,
        "entities_loaded": len(gnn_service.entity_embeddings) if gnn_service else 0,
        "classes_available": len(gnn_service.class_names) if gnn_service else 0,
        "device": str(gnn_service.device) if gnn_service else "unknown"
    }

@router.post("/api/v1/gnn/classify")
async def classify_entity(entity: str, context: str = ""):
    """Classify a single entity using GNN"""
    gnn_service = get_gnn_service()

    if not gnn_service or not gnn_service.initialized:
        raise HTTPException(status_code=503, detail="GNN service not available")

    try:
        classification = gnn_service.gnn_service.classify_entity(entity, context)
        return classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/api/v1/gnn/reasoning")
async def gnn_reasoning(start_entity: str, end_entity: str, max_hops: int = 3):
    """Perform GNN-enhanced reasoning between two entities"""
    gnn_service = get_gnn_service()

    if not gnn_service or not gnn_service.initialized:
        raise HTTPException(status_code=503, detail="GNN service not available")

    try:
        paths = gnn_service.gnn_service.gnn_enhanced_multi_hop_reasoning(
            start_entity, end_entity, max_hops
        )

        return {
            "start_entity": start_entity,
            "end_entity": end_entity,
            "max_hops": max_hops,
            "paths_found": len(paths),
            "paths": paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@router.get("/api/v1/gnn/classes")
async def get_gnn_classes():
    """Get available GNN classification classes"""
    gnn_service = get_gnn_service()

    if not gnn_service or not gnn_service.initialized:
        raise HTTPException(status_code=503, detail="GNN service not available")

    return {
        "classes": gnn_service.gnn_service.class_names,
        "num_classes": len(gnn_service.gnn_service.class_names)
    }
