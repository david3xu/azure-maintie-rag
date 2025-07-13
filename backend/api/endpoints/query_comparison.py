"""
Query comparison endpoint for A/B testing
Compares multi-modal vs structured RAG approaches side-by-side
Provides performance metrics and quality comparison
"""

import logging
import time
import sys
import os
from typing import Dict, List, Any

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.pipeline.enhanced_rag import get_rag_instance
from src.models.maintenance_models import RAGResponse
from api.models.query_models import ComparisonRequest


logger = logging.getLogger(__name__)

# Create router for comparison endpoint
router = APIRouter()


def get_rag_system():
    """Dependency to get RAG system instance"""
    rag_system = get_rag_instance()
    # Check if both RAG components are initialized
    if (not hasattr(rag_system, 'multi_modal_rag') or not rag_system.multi_modal_rag.components_initialized or
        not hasattr(rag_system, 'structured_rag') or not rag_system.structured_rag.components_initialized):
        raise HTTPException(
            status_code=503,
            detail="RAG system components not initialized. Please wait for system startup to complete."
        )
    return rag_system


@router.post("/", response_model=Dict[str, Any])
async def compare_retrieval_methods(
    request: ComparisonRequest,
    rag_system=Depends(get_rag_system)
) -> Dict[str, Any]:
    """
    Compare both retrieval methods side-by-side

    Runs the same query through both:
    1. Original multi-modal retrieval (3 API calls)
    2. Optimized structured retrieval (1 API call)

    Returns performance metrics and quality comparison for A/B testing.
    """

    start_time = time.time()

    try:
        logger.info(f"Comparing retrieval methods for query: {request.query}")

        # Method 1: Original multi-modal
        multi_modal_start = time.time()
        multi_modal_response = rag_system.process_query(
            query=request.query,
            max_results=request.max_results,
            include_explanations=request.include_explanations,
            enable_safety_warnings=request.enable_safety_warnings
        )
        multi_modal_time = time.time() - multi_modal_start

        # Method 2: Optimized structured
        optimized_start = time.time()
        optimized_response = await rag_system.process_query_optimized(
            query=request.query,
            max_results=request.max_results,
            include_explanations=request.include_explanations,
            enable_safety_warnings=request.enable_safety_warnings
        )
        optimized_time = time.time() - optimized_start

        # Calculate performance improvements
        time_improvement = ((multi_modal_time - optimized_time) / multi_modal_time) * 100
        speedup_factor = multi_modal_time / optimized_time if optimized_time > 0 else 0

        # Compare response quality metrics
        quality_comparison = {
            "confidence_score": {
                "multi_modal": multi_modal_response.confidence_score,
                "optimized": optimized_response.confidence_score,
                "difference": optimized_response.confidence_score - multi_modal_response.confidence_score
            },
            "search_results_count": {
                "multi_modal": len(multi_modal_response.search_results),
                "optimized": len(optimized_response.search_results)
            },
            "safety_warnings_count": {
                "multi_modal": len(multi_modal_response.safety_warnings),
                "optimized": len(optimized_response.safety_warnings)
            }
        }

        comparison_result = {
            "query": request.query,
            "timestamp": time.time(),
            "performance": {
                "multi_modal": {
                    "processing_time": multi_modal_time,
                    "method": "multi_modal_retrieval",
                    "api_calls_estimated": 3  # Original method uses 3 vector searches
                },
                "optimized": {
                    "processing_time": optimized_time,
                    "method": "optimized_structured_rag",
                    "api_calls_estimated": 1  # New method uses 1 vector search
                },
                "improvement": {
                    "time_reduction_percent": time_improvement,
                    "speedup_factor": speedup_factor,
                    "api_calls_reduction": 2  # 3 -> 1 calls
                }
            },
            "quality_comparison": quality_comparison,
            "responses": {
                "multi_modal": {
                    "response": multi_modal_response.generated_response[:500] + "..." if len(multi_modal_response.generated_response) > 500 else multi_modal_response.generated_response,
                    "sources": multi_modal_response.sources[:3],  # First 3 sources
                    "safety_warnings": multi_modal_response.safety_warnings
                },
                "optimized": {
                    "response": optimized_response.generated_response[:500] + "..." if len(optimized_response.generated_response) > 500 else optimized_response.generated_response,
                    "sources": optimized_response.sources[:3],  # First 3 sources
                    "safety_warnings": optimized_response.safety_warnings
                }
            },
            "recommendation": {
                "use_optimized": optimized_time < multi_modal_time and optimized_response.confidence_score >= multi_modal_response.confidence_score * 0.95,
                "reason": f"Optimized method is {speedup_factor:.1f}x faster with {quality_comparison['confidence_score']['difference']:+.3f} confidence difference"
            }
        }

        total_time = time.time() - start_time
        logger.info(f"Comparison completed in {total_time:.2f}s - Optimized is {speedup_factor:.1f}x faster")

        return comparison_result

    except Exception as e:
        logger.error(f"Error in method comparison: {e}", exc_info=True)
        processing_time = time.time() - start_time

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Method comparison failed",
                "message": str(e),
                "query": request.query,
                "processing_time": processing_time,
                "suggestions": [
                    "Try individual endpoints: POST /api/v1/query/multi-modal or POST /api/v1/query/structured",
                    "Check if the system is fully initialized",
                    "Verify both methods are available"
                ]
            }
        )